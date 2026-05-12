"""
Training data preprocessing pipeline for FreshlyFishy.

Converts raw fish images into background-removed, CLAHE-enhanced crops
that match the inference pipeline exactly — eliminating train/inference
distribution mismatch.

Directory layout expected:
    raw_dataset/
        fresh/      *.jpg / *.jpeg / *.png
        not_fresh/  *.jpg / *.jpeg / *.png

Output:
    processed_dataset/
        fresh/      *_processed.jpg
        not_fresh/  *_processed.jpg

Usage:
    pip install rembg ultralytics opencv-python numpy pillow onnxruntime
    python prepare_training_data.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Try to import rembg; guide user if missing ───────────────────────────────
try:
    from rembg import remove as rembg_remove
    from PIL import Image
    import io
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print(
        "[WARNING] rembg not installed.  Falling back to GrabCut.\n"
        "  Install with:  pip install rembg onnxruntime\n"
    )

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_PATH   = "models/yolo_detection_model.pt"
RAW_DIR     = Path("raw_dataset")
OUT_DIR     = Path("processed_dataset")
IMG_SIZE    = 224           # must match classifier input
CLASSES     = ["fresh", "not_fresh"]
EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Bbox padding: expand YOLO crop by this fraction before masking.
# Ensures the full fish (fins, tail) is included even when YOLO crops tight.
BBOX_PAD = 0.08


# ── YOLO setup ────────────────────────────────────────────────────────────────
print("Loading YOLO detector …")
yolo = YOLO(YOLO_PATH)
print("YOLO loaded ✅")


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_fish_bbox(image: np.ndarray):
    """
    Run YOLO at high resolution and return the best whole-fish bbox.
    Falls back to the full image if nothing is detected.
    """
    h, w = image.shape[:2]

    for imgsz, conf in [(1280, 0.15), (640, 0.08)]:
        results = yolo(image, imgsz=imgsz, conf=conf, verbose=False)[0]
        if len(results.boxes) == 0:
            continue

        # score = conf × area; prefer large confident boxes
        best = max(
            results.boxes,
            key=lambda b: float(b.conf) * (
                (float(b.xyxy[0][2]) - float(b.xyxy[0][0])) *
                (float(b.xyxy[0][3]) - float(b.xyxy[0][1]))
            ),
        )
        x1, y1, x2, y2 = map(int, best.xyxy[0])

        # pad the crop slightly so fins/tail are included
        pad_x = int((x2 - x1) * BBOX_PAD)
        pad_y = int((y2 - y1) * BBOX_PAD)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # sanity: ignore tiny boxes (eye-only detections)
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < (h * w * 0.04):
            continue

        return x1, y1, x2, y2

    # fallback: use full image
    return 0, 0, w, h


# ── Background removal ────────────────────────────────────────────────────────

def remove_bg_rembg(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Use U²-Net (via rembg) to produce a pixel-perfect fish mask.

    rembg returns an RGBA PIL image; we extract the alpha channel as mask.
    Quality is far superior to GrabCut, especially on complex backgrounds.
    """
    # convert BGR → PIL RGB for rembg
    pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    result_pil = rembg_remove(pil_img)          # returns RGBA PIL image
    alpha = np.array(result_pil)[:, :, 3]       # alpha channel = foreground mask
    return (alpha > 128).astype(np.uint8) * 255


def remove_bg_grabcut(crop_bgr: np.ndarray) -> np.ndarray:
    """
    GrabCut fallback.  Works well enough on uniform-colour backgrounds
    (green tray, white styrofoam).  Less reliable than rembg on complex scenes.
    """
    h, w = crop_bgr.shape[:2]
    if h < 32 or w < 32:
        return np.full((h, w), 255, dtype=np.uint8)

    mask       = np.zeros((h, w), np.uint8)
    bgd_model  = np.zeros((1, 65), np.float64)
    fgd_model  = np.zeros((1, 65), np.float64)

    margin_x = max(4, w // 10)
    margin_y = max(4, h // 10)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    try:
        cv2.grabCut(crop_bgr, mask, rect, bgd_model, fgd_model, 5,
                    cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)

        if fg.sum() < (h * w * 0.05 * 255):
            raise ValueError("near-empty mask")

        return fg
    except Exception:
        return np.full((h, w), 255, dtype=np.uint8)


def get_fish_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Route to rembg if available, else GrabCut."""
    if REMBG_AVAILABLE:
        return remove_bg_rembg(crop_bgr)
    return remove_bg_grabcut(crop_bgr)


# ── Post-masking strategy ─────────────────────────────────────────────────────
#
# Two options — choose ONE and use it consistently at inference too:
#
#   STRATEGY = "zero"   →  background becomes solid black
#   STRATEGY = "blur"   →  background is heavily blurred (default)
#
# "blur" is recommended if you are not 100 % sure rembg masks are clean,
# because some edge pixels will remain and solid-black edges create artifacts.
# "zero" gives the cleanest GradCAM but requires near-perfect masks.
#
STRATEGY = "blur"   # change to "zero" if rembg masks are high quality


def apply_strategy(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if STRATEGY == "zero":
        result = crop_bgr.copy()
        result[mask == 0] = 0
        return result
    else:  # blur
        blurred = cv2.GaussianBlur(crop_bgr, (51, 51), 0)
        result = crop_bgr.copy()
        result[mask == 0] = blurred[mask == 0]
        return result


# ── CLAHE ─────────────────────────────────────────────────────────────────────

def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Full pipeline per image ───────────────────────────────────────────────────

def process_image(src: Path, dst: Path) -> bool:
    """
    Full preprocessing pipeline:
      raw image
        → YOLO detect + padded crop
        → background removal (rembg or GrabCut)
        → apply masking strategy (blur/zero)
        → CLAHE contrast enhancement
        → resize to (224, 224)
        → save as JPEG
    Returns True on success.
    """
    try:
        image = load_image(src)

        # 1. Detect and crop
        x1, y1, x2, y2 = detect_fish_bbox(image)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            print(f"  [SKIP] empty crop: {src.name}")
            return False

        # 2. Background mask
        mask = get_fish_mask(crop)

        # 3. Apply masking strategy
        processed = apply_strategy(crop, mask)

        # 4. CLAHE
        processed = apply_clahe(processed)

        # 5. Resize
        processed = cv2.resize(processed, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_LANCZOS4)

        # 6. Save
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True

    except Exception as exc:
        print(f"  [ERROR] {src.name}: {exc}")
        return False


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    if not RAW_DIR.exists():
        print(f"[ERROR] Raw dataset directory not found: {RAW_DIR.resolve()}")
        print("Create it with subfolders:  fresh/  and  not_fresh/")
        sys.exit(1)

    algo = "rembg (U²-Net)" if REMBG_AVAILABLE else "GrabCut (fallback)"
    print(f"\nBackground removal algorithm : {algo}")
    print(f"Masking strategy             : {STRATEGY}")
    print(f"Output size                  : {IMG_SIZE}×{IMG_SIZE}\n")

    total, ok, fail = 0, 0, 0

    for cls in CLASSES:
        src_dir = RAW_DIR / cls
        dst_dir = OUT_DIR / cls

        if not src_dir.exists():
            print(f"[SKIP] Class folder not found: {src_dir}")
            continue

        images = [p for p in src_dir.iterdir() if p.suffix.lower() in EXTENSIONS]
        print(f"Processing [{cls}] — {len(images)} images …")

        for src_path in images:
            dst_path = dst_dir / (src_path.stem + "_processed.jpg")
            total += 1
            success = process_image(src_path, dst_path)
            if success:
                ok += 1
                print(f"  ✅ {src_path.name}")
            else:
                fail += 1

    print(f"\nDone.  {ok}/{total} processed successfully, {fail} failed.")
    print(f"Processed dataset saved to: {OUT_DIR.resolve()}")
    print("\nNext steps:")
    print("  1. Visually inspect a sample of processed_dataset/ images.")
    print("  2. Retrain EfficientNetV2S on processed_dataset/.")
    print("  3. Update inference pipeline to use the same masking strategy.")


if __name__ == "__main__":
    main()
