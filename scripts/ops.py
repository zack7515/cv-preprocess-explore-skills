"""Preprocessing op library + pipeline runner (OpenCV + NumPy only).

Images are NumPy uint8 arrays: HxWx3 (BGR) or HxW (gray).
A pipeline is a list of steps: {"op": "<name>", <params...>, "if": {...optional}}.

    python ops.py --list            # show every op and its parameters

This file is copied unchanged into the exported deliverable, so it must not
depend on anything else in the skill.
"""
from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_SIDE = 512  # metrics are measured at this max side so sizes are comparable
METRICS = ("brightness", "contrast", "sharpness", "noise", "clip_dark", "clip_bright",
           "colorfulness", "width", "height", "channels")  # usable in a step's "if"


# ---------------------------------------------------------------- I/O
def load_image(path) -> np.ndarray | None:
    """Read an image (unicode-safe on Windows). EXIF orientation is applied.
    Returns BGR or gray uint8, or None if unreadable."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype != np.uint8:  # 16-bit PNG/TIFF
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, int(img.max())))
    if img.ndim == 3 and img.shape[2] == 4:  # composite alpha on white
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    # IMREAD_UNCHANGED skips EXIF rotation; re-decode colour JPEGs with it applied
    if Path(str(path)).suffix.lower() in {".jpg", ".jpeg"} and img.ndim == 3:
        rot = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if rot is not None:
            img = rot
    return img


def save_image(path, img: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


# ---------------------------------------------------------------- helpers
def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _odd(k):
    k = int(k)
    return k if k % 2 == 1 else k + 1


def _on_luma(img, fn):
    """Apply fn to luminance only, keeping colour."""
    if img.ndim == 2:
        return fn(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = fn(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _metric_view(img):
    g = _gray(img)
    h, w = g.shape
    s = METRIC_SIDE / max(h, w)
    if s < 1:
        g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return g


def quick_metrics(img) -> dict:
    """Cheap per-image metrics. Used by conditional steps and by evaluation."""
    g = _metric_view(img)
    gf = g.astype(np.float32)
    h, w = g.shape
    # Immerkaer noise estimate
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(gf, -1, k)[1:-1, 1:-1]
    noise = math.sqrt(math.pi / 2) * float(np.abs(conv).sum()) / (6 * max(1, (w - 2) * (h - 2)))
    if img.ndim == 3:
        b, gg, r = [c.astype(np.float32) for c in cv2.split(img)]
        rg, yb = r - gg, 0.5 * (r + gg) - b
        colorful = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    else:
        colorful = 0.0
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "channels": 1 if img.ndim == 2 else int(img.shape[2]),
        "brightness": round(float(gf.mean()), 2),
        "contrast": round(float(gf.std()), 2),
        "sharpness": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2),
        "noise": round(noise, 3),
        "clip_dark": round(float((g <= 3).mean()), 4),
        "clip_bright": round(float((g >= 252).mean()), 4),
        "colorfulness": round(colorful, 2),
    }


# ---------------------------------------------------------------- ops
# Every op: fn(img, **params) -> img. Docstring first line is shown by --list.

def to_gray(img):
    """Convert to single-channel grayscale."""
    return _gray(img)


def to_color(img):
    """Ensure 3-channel BGR (gray is replicated)."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def resize(img, max_side=None, width=None, height=None, upscale=True):
    """Resize by max_side, or to exact width/height (one given keeps aspect)."""
    h, w = img.shape[:2]
    if max_side:
        s = max_side / max(h, w)
        nw, nh = round(w * s), round(h * s)
    elif width and height:
        nw, nh = int(width), int(height)
    elif width:
        nw, nh = int(width), round(h * width / w)
    elif height:
        nw, nh = round(w * height / h), int(height)
    else:
        return img
    if not upscale and (nw > w or nh > h):
        return img
    interp = cv2.INTER_AREA if nw * nh < w * h else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, nw), max(1, nh)), interpolation=interp)


def letterbox(img, width=640, height=640, color=114):
    """Fit inside width x height keeping aspect, pad the rest with color."""
    h, w = img.shape[:2]
    s = min(width / w, height / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    top, left = (height - nh) // 2, (width - nw) // 2
    val = color if img.ndim == 2 else (color, color, color)
    return cv2.copyMakeBorder(r, top, height - nh - top, left, width - nw - left, cv2.BORDER_CONSTANT, value=val)


def pad_square(img, color=0):
    """Pad the short side so the image becomes square."""
    h, w = img.shape[:2]
    m = max(h, w)
    return letterbox(img, m, m, color) if h != w else img


def center_crop(img, frac=0.9):
    """Keep the central frac of width and height."""
    h, w = img.shape[:2]
    ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
    y, x = (h - ch) // 2, (w - cw) // 2
    return img[y:y + ch, x:x + cw]


def trim_border(img, thresh=12, margin=2):
    """Crop away uniform borders (scanner edges, black/white frames)."""
    g = _gray(img).astype(np.int16)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = int(np.median(border))
    mask = (np.abs(g - bg) > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pts = cv2.findNonZero(mask)
    if pts is None:
        return img
    x, y, w, h = cv2.boundingRect(pts)
    H, W = g.shape
    if w * h < 0.05 * W * H:  # refuse to crop to a speck
        return img
    x0, y0 = max(0, x - margin), max(0, y - margin)
    return img[y0:min(H, y + h + margin), x0:min(W, x + w + margin)]


def rotate(img, angle=0.0, color=255):
    """Rotate by angle degrees (counter-clockwise), expanding the canvas."""
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    val = color if img.ndim == 2 else (color, color, color)
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=val)


def deskew(img, max_angle=10.0, step=0.5, color=255):
    """Straighten text/line content by projection-profile search (documents, labels)."""
    g = _metric_view(img)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw.mean() > 127:  # light-on-dark content: invert so content is foreground
        bw = 255 - bw
    best, best_a = -1.0, 0.0
    for a in np.arange(-max_angle, max_angle + 1e-9, step):
        r = rotate(bw, float(a), color=0)
        score = float(np.var(r.sum(axis=1, dtype=np.float64)))
        if score > best:
            best, best_a = score, float(a)
    return rotate(img, best_a, color=color)


def doc_warp(img, min_area=0.2):
    """Find the largest 4-corner contour (paper, screen, card) and flatten it."""
    g = _gray(img)
    s = 800 / max(g.shape)
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    s = min(s, 1.0)
    edges = cv2.Canny(cv2.GaussianBlur(small, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_min = min_area * small.shape[0] * small.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < area_min:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32) / s
            sm, df = pts.sum(1), np.diff(pts, axis=1).ravel()
            tl, br, tr, bl = pts[np.argmin(sm)], pts[np.argmax(sm)], pts[np.argmin(df)], pts[np.argmax(df)]
            w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            if w < 10 or h < 10:
                continue
            M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
            return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)
    return img


def gamma(img, g=1.0):
    """Gamma correction. g<1 brightens, g>1 darkens."""
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def auto_gamma(img, target=128):
    """Choose gamma so mean luminance moves to target."""
    m = float(_gray(img).mean())
    if m <= 1 or m >= 254:
        return img
    g = math.log(target / 255.0) / math.log(m / 255.0)  # (m/255)^g == target/255
    return gamma(img, float(np.clip(g, 0.3, 3.0)))


def auto_levels(img, low_pct=1.0, high_pct=99.0):
    """Stretch luminance so the given percentiles map to 0 and 255."""
    def f(ch):
        lo, hi = np.percentile(ch, [low_pct, high_pct])
        if hi - lo < 5:
            return ch
        return np.clip((ch.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return _on_luma(img, f)


def clahe(img, clip=2.0, tile=8):
    """Local contrast equalisation (CLAHE) on luminance."""
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return _on_luma(img, c.apply)


def gray_world(img):
    """White balance by the gray-world assumption (removes colour casts)."""
    if img.ndim == 2:
        return img
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    f *= means.mean() / np.maximum(means, 1e-3)
    return np.clip(f, 0, 255).astype(np.uint8)


def remove_shadow(img, k=21):
    """Flatten uneven lighting by dividing out a blurred background (documents, flat objects)."""
    def f(ch):
        bg = cv2.medianBlur(cv2.dilate(ch, np.ones((7, 7), np.uint8)), _odd(k))
        out = cv2.divide(ch, bg, scale=255)
        return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return _on_luma(img, f)  # luminance only: per-channel normalising shifts colours


def denoise_nlm(img, h=7, h_color=7, template=7, search=21):
    """Non-local-means denoise (strong, slow). h = strength."""
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, float(h), int(template), int(search))
    return cv2.fastNlMeansDenoisingColored(img, None, float(h), float(h_color), int(template), int(search))


def median(img, k=3):
    """Median blur; good for salt-and-pepper noise."""
    return cv2.medianBlur(img, _odd(k))


def bilateral(img, d=7, sigma_color=50, sigma_space=50):
    """Edge-preserving smoothing."""
    return cv2.bilateralFilter(img, int(d), float(sigma_color), float(sigma_space))


def gaussian(img, k=3):
    """Gaussian blur."""
    return cv2.GaussianBlur(img, (_odd(k), _odd(k)), 0)


def unsharp(img, amount=1.0, sigma=1.0):
    """Unsharp-mask sharpening."""
    blur = cv2.GaussianBlur(img, (0, 0), float(sigma))
    return cv2.addWeighted(img, 1 + float(amount), blur, -float(amount), 0)


def binarize(img, method="otsu", block=31, C=10):
    """Black/white output. method: otsu | adaptive."""
    g = _gray(img)
    if method == "adaptive":
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, _odd(block), float(C))
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def morph(img, op="close", k=3, iterations=1):
    """Morphology: open | close | erode | dilate."""
    kern = np.ones((int(k), int(k)), np.uint8)
    if op == "erode":
        return cv2.erode(img, kern, iterations=int(iterations))
    if op == "dilate":
        return cv2.dilate(img, kern, iterations=int(iterations))
    code = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}[op]
    return cv2.morphologyEx(img, code, kern, iterations=int(iterations))


OPS = {f.__name__: f for f in [
    to_gray, to_color, resize, letterbox, pad_square, center_crop, trim_border,
    rotate, deskew, doc_warp, gamma, auto_gamma, auto_levels, clahe, gray_world,
    remove_shadow, denoise_nlm, median, bilateral, gaussian, unsharp, binarize, morph,
]}


# ---------------------------------------------------------------- pipeline
def _cond_ok(img, cond: dict | None) -> bool:
    """{"metric": "brightness", "lt": 80} / {"gt": ..} / {"between": [a, b]}"""
    if not cond:
        return True
    v = quick_metrics(img)[cond["metric"]]
    if "lt" in cond and not v < cond["lt"]:
        return False
    if "gt" in cond and not v > cond["gt"]:
        return False
    if "between" in cond and not (cond["between"][0] <= v <= cond["between"][1]):
        return False
    return True


def _load_custom(step: dict, base_dir: Path):
    path = (base_dir / step["module"]).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, step["fn"])


def validate_steps(steps: list, base_dir: Path = Path(".")) -> None:
    for i, st in enumerate(steps):
        name = st.get("op")
        cond = st.get("if")
        if cond is not None and (not isinstance(cond, dict) or cond.get("metric") not in METRICS
                                 or not set(cond) - {"metric"} or set(cond) - {"metric", "lt", "gt", "between"}):
            raise ValueError(f'step {i}: bad "if" {cond}; use {{"metric": <one of {", ".join(METRICS)}>, '
                             f'"lt"/"gt": n or "between": [a, b]}}')
        if name == "custom":
            if not (base_dir / st.get("module", "")).is_file():
                raise ValueError(f"step {i}: custom module not found: {st.get('module')}")
            continue
        if name not in OPS:
            raise ValueError(f"step {i}: unknown op '{name}'. Available: {', '.join(OPS)}")
        sig = inspect.signature(OPS[name])
        allowed = set(list(sig.parameters)[1:])
        extra = set(st) - allowed - {"op", "if"}
        if extra:
            raise ValueError(f"step {i} ({name}): unknown params {sorted(extra)}; allowed {sorted(allowed)}")


def apply_pipeline(img: np.ndarray, steps: list, base_dir: Path = Path(".")) -> np.ndarray:
    """Run steps in order. base_dir resolves custom modules."""
    for st in steps:
        if not _cond_ok(img, st.get("if")):
            continue
        params = {k: v for k, v in st.items() if k not in {"op", "if", "module", "fn"}}
        fn = _load_custom(st, Path(base_dir)) if st["op"] == "custom" else OPS[st["op"]]
        img = fn(img, **params)
        if img is None or img.size == 0:
            raise RuntimeError(f"op {st['op']} returned an empty image")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def list_ops() -> str:
    lines = []
    for name, fn in OPS.items():
        params = [str(p) for p in list(inspect.signature(fn).parameters.values())[1:]]
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        lines.append(f"{name}({', '.join(params)})  - {doc}")
    lines.append('custom(module="custom_ops/x.py", fn="name", ...)  - your own fn(img, **params) -> img')
    lines.append('any step may add "if": {"metric": "brightness", "lt": 80}  (metrics: ' + " ".join(METRICS) + ')')
    return "\n".join(lines)


if __name__ == "__main__":
    if "--list" in sys.argv:
        print(list_ops())
    else:
        print(__doc__)
