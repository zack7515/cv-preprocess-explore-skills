"""Profile an image folder before exploring.

    python inspect_dataset.py <input_dir> --workdir <workdir> [--preset photo|ocr|detect|vlm]

Writes <workdir>/inspect.json, <workdir>/inspect_sheet.jpg (worst examples per
issue) and <workdir>/goal.draft.json (preset checks; adapt them to the user's goal).
Never modifies the input folder.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import contact_sheet, list_images, load_safe, metrics_for, summarize  # noqa: E402

FLAGS = {  # heuristic, measured at 512px max side
    "dark": lambda m: m["brightness"] < 60,
    "bright": lambda m: m["brightness"] > 200,
    "low_contrast": lambda m: m["contrast"] < 30,
    "blurry": lambda m: m["sharpness"] < 50,
    "noisy": lambda m: m["noise"] > 3,
    "small": lambda m: m["_min_side"] < 224,
    "grayscale": lambda m: m["channels"] == 1 or m["colorfulness"] < 3,
    "clipped": lambda m: m["clip_dark"] > 0.1 or m["clip_bright"] > 0.1,
}

SEVERITY = {  # sort key per flag, smaller = worse; picks the sheet examples
    "dark": lambda m: m["brightness"],
    "bright": lambda m: -m["brightness"],
    "low_contrast": lambda m: m["contrast"],
    "blurry": lambda m: m["sharpness"],
    "noisy": lambda m: -m["noise"],
    "small": lambda m: m["_min_side"],
    "grayscale": lambda m: m["colorfulness"],
    "clipped": lambda m: -max(m["clip_dark"], m["clip_bright"]),
}

_PHOTO = {"brightness": [80, 190], "contrast_min": 40, "sharpness_min": 100, "noise_max": 2.0,
          "clip_dark_max": 0.05, "clip_bright_max": 0.05}
PRESETS = {  # starting checks + output mode per task (references/evaluation.md explains them)
    "photo": (_PHOTO, "keep"),
    "ocr": ({"brightness": [190, 250], "contrast_min": 40, "noise_max": 1.5, "clip_dark_max": 0.05}, "gray"),
    "detect": ({**_PHOTO, "size": [640, 640]}, "keep"),  # set size to the detector's input
    "vlm": ({"brightness": [60, 220], "clip_dark_max": 0.1, "clip_bright_max": 0.1, "max_side": 2048}, "keep"),
}


def _bits(s) -> int:
    return int("".join("1" if b else "0" for b in (s[:, 1:] > s[:, :-1]).flatten()), 2)


def dhash(img) -> tuple[int, int]:
    """256-bit dHash of the image and of its high-pass detail. Lighting gradients
    make unrelated images share the first hash, so duplicates must match both."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 16)
    small = lambda x: cv2.resize(x, (17, 16), interpolation=cv2.INTER_AREA)
    return _bits(small(g)), _bits(small(np.abs(hp)))


def hdist(a, b) -> int:
    return max(bin(a[0] ^ b[0]).count("1"), bin(a[1] ^ b[1]).count("1"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--dup-distance", type=int, default=10, help="max differing bits of 256")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="photo", help="starting checks for goal.draft.json")
    a = ap.parse_args()
    root, work = Path(a.input_dir).resolve(), Path(a.workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    files = list_images(root)
    if not files:
        sys.exit(f"no images found in {root}")

    rows, unreadable, hashes, imgs_by_name = [], [], [], {}
    for f in files:
        img = load_safe(f)
        rel = str(f.relative_to(root)).replace("\\", "/")
        if img is None:
            unreadable.append(rel)
            continue
        m = metrics_for(img)
        flags = [k for k, fn in FLAGS.items() if fn(m)]
        rows.append({"file": rel, "format": f.suffix.lower().lstrip("."), "metrics": m, "flags": flags})
        hashes.append((rel, dhash(img)))

    # near-duplicates
    dups, seen = [], set()
    if len(hashes) <= 5000:
        for i, (ni, hi) in enumerate(hashes):
            if ni in seen:
                continue
            group = [nj for nj, hj in hashes[i + 1:] if nj not in seen and hdist(hi, hj) <= a.dup_distance]
            if group:
                dups.append([ni] + group)
                seen.update(group)

    flag_count = collections.Counter(fl for r in rows for fl in r["flags"])
    sizes = [(r["metrics"]["width"], r["metrics"]["height"]) for r in rows]
    report = {
        "input_dir": str(root),
        "total_files": len(files),
        "readable": len(rows),
        "unreadable": unreadable,
        "formats": dict(collections.Counter(r["format"] for r in rows)),
        "channels": dict(collections.Counter(str(r["metrics"]["channels"]) for r in rows)),
        "size": {
            "min_side_p10": int(np.percentile([min(s) for s in sizes], 10)),
            "min_side_median": int(np.median([min(s) for s in sizes])),
            "max_side_median": int(np.median([max(s) for s in sizes])),
            "aspect_median": round(float(np.median([w / h for w, h in sizes])), 3),
            "distinct_sizes": len(set(sizes)),
        },
        "metrics": summarize(rows),
        "flags": dict(flag_count.most_common()),
        "near_duplicates": dups,
        "images": rows,
    }
    (work / "inspect.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # sheet: up to 2 worst examples per frequent flag
    pairs = []
    for fl, _ in flag_count.most_common(5):
        for r in sorted((r for r in rows if fl in r["flags"]), key=lambda r: SEVERITY[fl](r["metrics"]))[:2]:
            img = load_safe(root / r["file"])
            pairs.append((f"[{fl}] {r['file']}", img, img))
    if pairs:
        contact_sheet(pairs, work / "inspect_sheet.jpg", h=140)

    med = report["metrics"]
    checks, mode = PRESETS[a.preset]
    draft = {  # preset defaults; adapt every value to the user's goal
        "task": "<restate the user's goal in one sentence, plus any assumptions you made>",
        "input_dir": str(root),
        "output": {"format": "png", "mode": mode},
        "checks": dict(checks),
        "primary": "proxy",
        "eval_cmd": None,
        "min_keep_rate": 0.9,
        "holdout_fraction": 0.2 if len(rows) >= 10 else 0,
        "max_versions": 10,
        "patience": 3,
        "min_improvement": 0.5,  # proxy scale 0-100; with eval_cmd set it on the eval score's scale
    }
    (work / "goal.draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")

    # concise console summary (the agent reads this, not the full JSON)
    print(f"images: {len(rows)} readable / {len(files)} files  unreadable: {len(unreadable)}")
    print(f"formats: {report['formats']}  channels: {report['channels']}")
    print(f"size: {report['size']}")
    for k, v in med.items():
        print(f"{k:13s} p10={v['p10']:<9} median={v['median']:<9} p90={v['p90']}")
    print(f"flags: {report['flags'] or 'none'}")
    print(f"near-duplicate groups: {len(dups)}")
    print(f"goal.draft.json uses preset '{a.preset}': adapt checks to the user's goal before saving as goal.json")
    print(f"wrote: inspect.json, goal.draft.json{', inspect_sheet.jpg' if pairs else ''} in {work}")


if __name__ == "__main__":
    main()
