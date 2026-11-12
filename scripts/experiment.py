"""Run one exploration version: apply a pipeline config to the explore split,
score it, write a before/after sheet, and append to the log.

    python experiment.py --workdir W --baseline                 # v0: no processing
    python experiment.py --workdir W --config W/configs/v1.json --note "CLAHE clip 2"
    python experiment.py --workdir W --status                   # leaderboard + stop advice

Config file:
    {"steps": [{"op": "clahe", "clip": 2.0}, ...],
     "filter": {"min_side": 64}}        # optional: reject unusable originals

Runs are immutable: a name that already exists is refused. The holdout split is
never touched here (export_pipeline.py evaluates it once at the end).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CHECKS, check_image, contact_sheet, list_images, load_goal, load_safe,  # noqa: E402
                    metrics_for, proxy_score, read_jsonl, run_eval_cmd, select, summarize)
from ops import apply_pipeline, save_image, validate_steps  # noqa: E402


def finalize_mode(img, mode):
    import cv2
    if mode == "gray" and img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "color" and img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def process_split(goal, steps, filt, files, root, out_dir, base_dir, save=True, sheet_n=0):
    per_image, rejected, errors, pairs, t_total = [], [], [], [], 0.0
    ext = "." + goal["output"]["format"].lstrip(".")
    for f in files:
        rel = str(f.relative_to(root)).replace("\\", "/")
        orig = load_safe(f)
        if orig is None:
            rejected.append({"file": rel, "reason": "unreadable"})
            continue
        if filt:
            res = check_image(metrics_for(orig), filt)
            failed = [k for k, ok in res.items() if not ok]
            if failed:
                rejected.append({"file": rel, "reason": "filter:" + ",".join(failed)})
                continue
        t0 = time.perf_counter()
        try:
            out = finalize_mode(apply_pipeline(orig, steps, base_dir), goal["output"]["mode"])
        except Exception as e:  # noqa: BLE001
            errors.append({"file": rel, "error": f"{type(e).__name__}: {e}"})
            continue
        t_total += time.perf_counter() - t0
        m = metrics_for(out)
        per_image.append({"file": rel, "metrics": m, "checks": check_image(m, goal["checks"])})
        if save:
            save_image(out_dir / Path(rel).with_suffix(ext), out)
        if len(pairs) < sheet_n:
            pairs.append((rel, orig, out))
    ms = round(1000 * t_total / max(1, len(per_image)), 1)
    return per_image, rejected, errors, pairs, ms


JUMP_WARN = 15  # proxy points gained in one version that smell like metric gaming


def keep_rate_of(per_image, rejected, n):
    """Kept / readable. Unreadable files are reported by inspect, not held against a version."""
    readable = n - sum(r["reason"] == "unreadable" for r in rejected)
    return round(len(per_image) / max(1, readable), 3)


def gain(a, b, goal):
    if b is None:
        return None
    d = a - b
    return round(d if (goal["primary"] == "proxy" or goal["eval_higher_is_better"]) else -d, 3)


def status(work: Path, goal: dict, quiet=False) -> dict:
    runs = [r for r in read_jsonl(work / "runs.jsonl") if r["split"] == "explore"]
    valid = [r for r in runs if r["valid"]]
    base = next((r for r in runs if r["name"] == "v0"), None)
    best, streak = None, 0
    for r in valid:
        if best is None or gain(r["score"], best["score"], goal) >= goal["min_improvement"]:
            best, streak = r, 0
        elif r["name"] != "v0":
            streak += 1
    tried = len([r for r in runs if r["name"] != "v0"])
    stop = []
    if tried >= goal["max_versions"]:
        stop.append(f"version budget used ({tried}/{goal['max_versions']})")
    if streak >= goal["patience"]:
        stop.append(f"{streak} versions in a row without >= {goal['min_improvement']} gain")
    info = {"tried": tried, "best": best and best["name"], "best_score": best and best["score"],
            "best_proxy": best and best["proxy_score"],
            "baseline_score": base and base["score"], "no_gain_streak": streak, "stop": stop}
    if not quiet:
        order = sorted(valid, key=lambda r: r["score"], reverse=goal["primary"] == "proxy" or goal["eval_higher_is_better"])
        print("leaderboard (explore split, valid runs):")
        for r in order[:10]:
            print(f"  {r['name']:<10} score={r['score']:<9} keep={r['keep_rate']:<6} {r['ms_per_img']}ms/img  {r['note'][:60]}")
        bad = [r["name"] for r in runs if not r["valid"]]
        if bad:
            print(f"invalid runs (errors or keep_rate < {goal['min_keep_rate']}): {', '.join(bad)}")
        print(f"tried {tried}/{goal['max_versions']}  best={info['best']}  no-gain streak={streak}/{goal['patience']}")
        print("STOP: " + "; ".join(stop) if stop else "continue exploring")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--config")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--note", default="")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="do not write output images (proxy only)")
    ap.add_argument("--force", action="store_true", help="run past max_versions (only with user approval)")
    a = ap.parse_args()

    work = Path(a.workdir).resolve()
    goal = load_goal(work)
    if a.status:
        status(work, goal)
        return

    if a.baseline:
        cfg, name = {"steps": []}, "v0"
        a.note = a.note or "baseline: originals, no processing"
    elif a.config:
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
        name = a.name or Path(a.config).stem
    else:
        sys.exit("give --baseline, --config or --status")
    steps, filt = cfg.get("steps", []), cfg.get("filter", {})
    validate_steps(steps, work)
    bad = set(filt) - set(CHECKS)
    if bad:
        sys.exit(f"unknown filter checks {sorted(bad)}; allowed {sorted(CHECKS)}")

    run_dir = work / "runs" / name
    if run_dir.exists():
        sys.exit(f"run '{name}' already exists. Runs are immutable; use a new name.")
    if not a.baseline and not any(r["name"] == "v0" for r in read_jsonl(work / "runs.jsonl")):
        print("warning: no baseline yet. Run --baseline first so gains are measurable.")
    prev = status(work, goal, quiet=True)
    if not a.baseline and prev["tried"] >= goal["max_versions"] and not a.force:
        sys.exit(f"STOP: version budget used ({prev['tried']}/{goal['max_versions']}). Export the best version "
                 "with export_pipeline.py. --force only if the user explicitly approved more versions.")

    root = Path(goal["input_dir"])
    files = select(list_images(root), root, "explore", goal["holdout_fraction"])
    if not files:
        sys.exit(f"no images in {root}")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    img_dir = run_dir / "images"
    need_save = not a.no_save or goal["primary"] == "eval_cmd"

    per_image, rejected, errors, pairs, ms = process_split(
        goal, steps, filt, files, root, img_dir, work, save=need_save, sheet_n=goal["sheet_count"])
    proxy, rates = proxy_score(per_image, goal["checks"])

    ev = None
    if goal["eval_cmd"]:
        lst = run_dir / "files.txt"
        lst.write_text("\n".join(str(p) for p in sorted(img_dir.rglob("*")) if p.is_file()), encoding="utf-8")
        try:
            ev = run_eval_cmd(goal["eval_cmd"], root, img_dir, lst, goal["eval_timeout_s"])
        except Exception as e:  # noqa: BLE001
            errors.append({"file": "*", "error": f"eval_cmd: {e}"})
    score = (ev or {}).get("score") if goal["primary"] == "eval_cmd" else proxy
    keep_rate = keep_rate_of(per_image, rejected, len(files))
    valid = score is not None and not errors and keep_rate >= goal["min_keep_rate"]

    if pairs:
        contact_sheet(pairs, run_dir / "sheet.jpg")
    if a.no_save and img_dir.exists() and goal["primary"] != "eval_cmd":
        shutil.rmtree(img_dir)

    rec = {
        "name": name, "split": "explore", "time": dt.datetime.now().isoformat(timespec="seconds"),
        "note": a.note, "steps": steps, "filter": filt, "score": score, "proxy_score": proxy,
        "eval": ev, "pass_rates": rates, "keep_rate": keep_rate, "n": len(files),
        "rejected": rejected, "errors": errors[:20], "ms_per_img": ms, "valid": valid,
        "summary": summarize(per_image) if per_image else {},
    }
    with open(work / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (run_dir / "result.json").write_text(json.dumps({**rec, "images": per_image}, ensure_ascii=False, indent=1), encoding="utf-8")

    d_base = gain(score, prev["baseline_score"], goal) if score is not None else None
    d_best = gain(score, prev["best_score"], goal) if score is not None else None
    log = work / "log.md"
    if not log.exists():
        log.write_text("| version | score | vs baseline | vs best | keep | ms/img | valid | change |\n"
                       "|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"| {name} | {score} | {d_base} | {d_best} | {keep_rate} | {ms} | {'yes' if valid else 'NO'} | "
                 f"{a.note.replace('|', '/')} |\n")

    # console summary for the agent
    print(f"== {name}  score={score}  (proxy={proxy}{', eval=' + str(ev.get('score')) if ev else ''})")
    print(f"vs baseline: {d_base}   vs previous best ({prev['best']}): {d_best}")
    print(f"images: {len(per_image)} kept / {len(files)}  keep_rate={keep_rate}  {ms} ms/img  valid={valid}")
    if prev["best_proxy"] is not None and proxy - prev["best_proxy"] > JUMP_WARN:
        print(f"WARNING: proxy jumped {proxy - prev['best_proxy']:.1f} points over {prev['best']} in one version. "
              "Likely metric gaming (binarize, heavy blur/NLM, over-crop, over-sharpen). Check the sheet, "
              "or compare 'median after' with the previous best and explain why this is a real improvement.")
    if rates:
        worst = sorted(rates.items(), key=lambda kv: kv[1])
        print("check pass rates (worst first): " + "  ".join(f"{k}={v}" for k, v in worst))
    s = rec["summary"]
    if s:
        print("median after: " + "  ".join(f"{k}={v['median']}" for k, v in s.items()))
    if rejected:
        print(f"rejected {len(rejected)}: " + ", ".join(f"{r['file']}({r['reason']})" for r in rejected[:5]))
    if errors:
        print(f"ERRORS {len(errors)}: " + "; ".join(f"{e['file']}: {e['error']}" for e in errors[:3]))
    if pairs:
        print(f"sheet: {run_dir / 'sheet.jpg'}")
    after = status(work, goal, quiet=True)
    print("STOP: " + "; ".join(after["stop"]) if after["stop"] else
          f"budget {after['tried']}/{goal['max_versions']}, no-gain streak {after['no_gain_streak']}/{goal['patience']}")


if __name__ == "__main__":
    main()
