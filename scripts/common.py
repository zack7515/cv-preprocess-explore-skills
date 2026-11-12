"""Shared helpers: goal file, dataset split, scoring, contact sheets, external eval."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ops import IMAGE_EXTS, load_image, quick_metrics

GOAL_DEFAULTS = {
    "task": "",
    "input_dir": "",
    "output": {"format": "png", "mode": "keep"},  # mode: keep | gray | color
    "checks": {},
    "primary": "proxy",            # proxy | eval_cmd
    "eval_cmd": None,              # e.g. "python my_eval.py --images {output_dir}"
    "eval_higher_is_better": True,
    "eval_timeout_s": 1800,
    "min_keep_rate": 0.9,
    "holdout_fraction": 0.2,
    "max_versions": 10,
    "patience": 3,
    "min_improvement": 0.5,
    "sheet_count": 8,
}

# check name -> (metric, kind). kind: "range" expects [lo, hi], "min"/"max" a number
CHECKS = {
    "brightness": ("brightness", "range"),
    "contrast_min": ("contrast", "min"),
    "sharpness_min": ("sharpness", "min"),
    "noise_max": ("noise", "max"),
    "clip_dark_max": ("clip_dark", "max"),
    "clip_bright_max": ("clip_bright", "max"),
    "min_side": ("_min_side", "min"),
    "max_side": ("_max_side", "max"),
    "size": ("_size", "eq"),         # [width, height]
    "channels": ("channels", "eq"),  # 1 or 3
}


def load_goal(workdir: Path) -> dict:
    p = workdir / "goal.json"
    if not p.exists():
        raise SystemExit(f"missing {p}. Create it first (see references/evaluation.md).")
    raw = json.loads(p.read_text(encoding="utf-8"))
    goal = {**GOAL_DEFAULTS, **raw}
    unknown = set(goal["checks"]) - set(CHECKS)
    if unknown:
        raise SystemExit(f"goal.json: unknown checks {sorted(unknown)}; allowed {sorted(CHECKS)}")
    if goal["primary"] == "eval_cmd" and not goal["eval_cmd"]:
        raise SystemExit("goal.json: primary=eval_cmd but eval_cmd is empty")
    if goal["primary"] == "eval_cmd" and "min_improvement" not in raw:
        raise SystemExit("goal.json: primary=eval_cmd needs an explicit min_improvement on the eval score's "
                         "scale, e.g. 0.005 for a 0-1 score, 0.5 for a 0-100 score")
    inp = Path(goal["input_dir"])
    if not inp.is_absolute():
        inp = (workdir / inp).resolve()
    goal["input_dir"] = str(inp)
    return goal


def list_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _key(f: Path, root: Path) -> str:
    rel = str(f.relative_to(root)).replace("\\", "/")
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()


def select(files: list[Path], root: Path, split: str, frac: float) -> list[Path]:
    """Deterministic split: files ordered by hash of their relative path, the first
    round(frac*n) are holdout. Same folder contents -> same split every run.
    Fewer than 10 images: no holdout (everything is explore)."""
    if split == "all":
        return files
    if len(files) < 10 or frac <= 0:
        return [] if split == "holdout" else files
    ranked = sorted(files, key=lambda f: _key(f, root))
    hold = set(ranked[:max(1, round(frac * len(files)))])
    return [f for f in files if (f in hold) == (split == "holdout")]


def metrics_for(img: np.ndarray) -> dict:
    m = quick_metrics(img)
    m["_min_side"] = min(m["width"], m["height"])
    m["_max_side"] = max(m["width"], m["height"])
    m["_size"] = [m["width"], m["height"]]
    return m


def check_image(m: dict, checks: dict) -> dict:
    """Return {check_name: bool} for every configured check."""
    out = {}
    for name, target in checks.items():
        key, kind = CHECKS[name]
        v = m[key]
        if kind == "range":
            out[name] = target[0] <= v <= target[1]
        elif kind == "min":
            out[name] = v >= target
        elif kind == "max":
            out[name] = v <= target
        else:
            out[name] = (list(v) if isinstance(v, (list, tuple)) else v) == target
    return out


def proxy_score(per_image: list[dict], checks: dict) -> tuple[float, dict]:
    """Score 0-100 = mean fraction of checks each kept image passes; plus per-check pass rates."""
    if not checks or not per_image:
        return 0.0, {}
    rates = {c: 0 for c in checks}
    total = 0.0
    for r in per_image:
        res = r["checks"]
        total += sum(res.values()) / len(res)
        for c, ok in res.items():
            rates[c] += ok
    n = len(per_image)
    return round(100 * total / n, 2), {c: round(v / n, 3) for c, v in rates.items()}


def summarize(per_image: list[dict]) -> dict:
    keys = ["brightness", "contrast", "sharpness", "noise", "clip_dark", "clip_bright", "colorfulness"]
    out = {}
    for k in keys:
        vals = np.array([r["metrics"][k] for r in per_image], dtype=np.float64)
        if vals.size:
            out[k] = {"p10": round(float(np.percentile(vals, 10)), 2),
                      "median": round(float(np.median(vals)), 2),
                      "p90": round(float(np.percentile(vals, 90)), 2)}
    return out


def run_eval_cmd(cmd: str, input_dir: Path, output_dir: Path, list_file: Path, timeout: int) -> dict:
    """Run the user's downstream evaluator. It must print a JSON object with "score"
    as the last JSON line on stdout. Placeholders: {output_dir} {input_dir} {list_file}."""
    q = lambda p: shlex.quote(str(p)) if not _is_windows() else f'"{p}"'
    full = cmd  # replace, not str.format: commands may contain literal braces
    for k, v in {"output_dir": output_dir, "input_dir": input_dir, "list_file": list_file}.items():
        full = full.replace("{" + k + "}", q(v))
    proc = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                if "score" in data:
                    return data
            except json.JSONDecodeError:
                pass
    tail = (proc.stdout[-800:] + "\n" + proc.stderr[-800:]).strip()
    raise RuntimeError(f"eval_cmd gave no JSON line with 'score' (exit {proc.returncode}):\n{tail}")


def _is_windows() -> bool:
    import os
    return os.name == "nt"


def _thumb(img: np.ndarray, h: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = h / img.shape[0]
    w = max(1, min(int(img.shape[1] * s), h * 3))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def contact_sheet(pairs: list[tuple[str, np.ndarray, np.ndarray]], out: Path, h: int = 180) -> None:
    """Rows of [before | after] with the file name. For humans (and vision models)."""
    rows = []
    for name, before, after in pairs:
        a, b = _thumb(before, h), _thumb(after, h)
        gap = np.full((h, 8, 3), 255, np.uint8)
        row = np.hstack([a, gap, b])
        label = np.full((22, row.shape[1], 3), 255, np.uint8)
        cv2.putText(label, re.sub(r"[^\x20-\x7e]", "?", name)[:80], (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.vstack([label, row]))
    if not rows:
        return
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    from ops import save_image
    save_image(out, np.vstack(rows), quality=85)


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_safe(path: Path):
    try:
        return load_image(path)
    except Exception:
        return None
