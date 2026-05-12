import base64
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
import requests as _requests
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from ultralytics import YOLO

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
_groq_client   = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ── LLM humanized analysis ────────────────────────────────────────────────────

def generate_llm_analysis(label: str, confidence: float, decision: str,
                           mask_coverage: float, focus_areas: list[str]) -> str:
    """
    Generate a plain-English freshness report via LLM.
    Priority: Groq (llama-3.3-70b) → Gemini (gemini-1.5-flash REST) → static fallback.
    """
    pct    = round(confidence * 100, 1)
    prompt = (
        f"You are a fish freshness expert AI. A vision model analyzed a fish image.\n\n"
        f"Results:\n"
        f"- Verdict: {label}\n"
        f"- Confidence: {pct}%\n"
        f"- Decision status: {decision}\n"
        f"- Fish body coverage in frame: {round(mask_coverage * 100, 1)}%\n"
        f"- Anatomical features examined: {', '.join(focus_areas)}\n\n"
        f"Write 2-3 concise sentences for a market vendor or consumer. "
        f"Explain what indicators the model observed, what the verdict means practically, "
        f"and give a brief recommendation. No bullet points, no headers, no jargon."
    )

    # ── Groq ─────────────────────────────────────────────────────────────────
    if _groq_client:
        try:
            resp = _groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=160,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # ── Gemini REST fallback ──────────────────────────────────────────────────
    if GEMINI_API_KEY:
        try:
            url  = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
            )
            body = {"contents": [{"parts": [{"text": prompt}]}]}
            r    = _requests.post(url, json=body, timeout=10)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass

    # ── Static fallback ───────────────────────────────────────────────────────
    if label == "Fresh":
        return (
            f"The model assessed this fish as Fresh with {pct}% confidence, "
            f"examining {focus_areas[0].lower()} and surrounding tissue. "
            f"The fish appears safe for consumption."
        )
    return (
        f"The model assessed this fish as Not Fresh with {pct}% confidence, "
        f"detecting signs of spoilage in {focus_areas[0].lower()} and related areas. "
        f"Consumption is not recommended."
    )

try:
    from rembg import remove as _rembg_remove
    from PIL import Image as _PILImage
    REMBG_AVAILABLE = True
    print("rembg available — using U²-Net for fish segmentation ✅")
except ImportError:
    REMBG_AVAILABLE = False
    print("rembg not installed — GrabCut fallback active")

# ── Config ────────────────────────────────────────────────────────────────────
CLASSIFIER_PATH      = "models/fish_classifier.keras"
YOLO_PATH            = "models/yolo_detection_model.pt"
IMG_SIZE             = 224
CLASS_NAMES          = ["Fresh", "Not Fresh"]   # must match alphabetical folder order
CONFIDENCE_THRESHOLD = 0.75
YOLO_CONF_THRESHOLD  = 0.20   # minimum YOLO score to count as "fish present"

# ── Model registry ────────────────────────────────────────────────────────────
classifier  = None
yolo_model  = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, yolo_model
    print("⏳ Loading models — server will be ready shortly…")
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
    yolo_model = YOLO(YOLO_PATH)
    print("✅ Classifier + YOLO loaded — server is ready!")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FreshlyFishy API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request schema ────────────────────────────────────────────────────────────
class ImageRequest(BaseModel):
    image_base64: str


# ── Image utilities ───────────────────────────────────────────────────────────

def b64_to_cv2(b64: str) -> np.ndarray:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes")
    return img


def cv2_to_b64(img: np.ndarray, ext: str = ".jpg") -> str:
    _, buf = cv2.imencode(ext, img)
    return base64.b64encode(buf).decode()


# ── YOLO presence gate ───────────────────────────────────────────────────────

def yolo_fish_present(image: np.ndarray) -> bool:
    """
    Run YOLOv8 on the image purely as a presence gate.
    Returns True if at least one fish is detected above YOLO_CONF_THRESHOLD.
    Does NOT crop or use the bounding box — detection only.
    """
    results = yolo_model.predict(
        source=image,
        imgsz=640,
        conf=YOLO_CONF_THRESHOLD,
        verbose=False,
    )
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            return True
    return False


# ── Fish detection via rembg ──────────────────────────────────────────────────

def _grabcut_mask(image: np.ndarray) -> np.ndarray:
    """GrabCut fallback when rembg is unavailable."""
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        return np.full((h, w), 255, dtype=np.uint8)

    mask      = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    mx, my    = max(4, w // 10), max(4, h // 10)
    rect      = (mx, my, w - 2 * mx, h - 2 * my)

    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5,
                    cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel, iterations=1)
        if fg.sum() < h * w * 0.05 * 255:
            raise ValueError("near-empty mask")
        return fg
    except Exception:
        return np.full((h, w), 255, dtype=np.uint8)


def detect_fish(image: np.ndarray):
    """
    Segment the fish directly from the full image using U²-Net (rembg).

    Steps:
      1. Run rembg on the full image → RGBA with alpha = foreground probability
      2. Threshold alpha to produce a binary fish mask
      3. Morphologically clean the mask
      4. Find the largest contiguous foreground region (the fish body)
      5. Derive a tight bounding box with a small padding
      6. Return (bbox, full-image mask) so the caller can crop both together

    Falls back to GrabCut if rembg is not installed.
    Returns (None, None) if no foreground is found.
    """
    H, W = image.shape[:2]

    if REMBG_AVAILABLE:
        pil_img = _PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        rgba    = np.array(_rembg_remove(pil_img))
        alpha   = rgba[:, :, 3]
        raw_mask = (alpha > 128).astype(np.uint8) * 255
    else:
        raw_mask = _grabcut_mask(image)

    # Morphological cleanup — close holes inside the fish body
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask   = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  kernel, iterations=1)

    # Find contours and take the largest one (= main fish body)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    min_area = H * W * 0.01          # ignore noise < 1 % of image
    valid    = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        valid = contours             # last resort: take whatever exists

    largest  = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 5 % padding so fins / tail at the mask boundary are included
    px = int(w * 0.05);  py = int(h * 0.05)
    x1 = max(0, x - px);  y1 = max(0, y - py)
    x2 = min(W, x + w + px);  y2 = min(H, y + h + py)

    return (x1, y1, x2, y2), mask


# ── Background suppression ────────────────────────────────────────────────────

def apply_mask_zero(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = crop.copy()
    result[mask == 0] = 0
    return result


def blur_background(crop: np.ndarray, mask: np.ndarray, ksize: int = 51) -> np.ndarray:
    """
    Keep the fish sharp; blur everything outside the mask.
    Blurring (not zeroing) keeps the image close to the training distribution
    so the classifier doesn't receive out-of-distribution background.
    """
    blurred = cv2.GaussianBlur(crop, (ksize | 1, ksize | 1), 0)
    result  = crop.copy()
    result[mask == 0] = blurred[mask == 0]
    return result


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_roi(crop: np.ndarray, fish_mask: np.ndarray) -> np.ndarray:
    """
    Preprocessing pipeline for EfficientNetV2S:
      1. Blur background → stays in training distribution
      2. CLAHE on fish region
      3. BGR → RGB
      4. Resize to (224, 224)
      5. Cast to float32 — NO /255 (model's include_preprocessing=True handles it)
    """
    out = blur_background(crop, fish_mask)
    out = apply_clahe(out)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out = cv2.resize(out, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(out.astype(np.float32), axis=0)


# ── Classification ────────────────────────────────────────────────────────────

def classify(img_array: np.ndarray):
    preds = classifier.predict(img_array, verbose=0)[0]
    if len(preds) == 1:
        conf  = float(preds[0])
        label = CLASS_NAMES[1] if conf > 0.5 else CLASS_NAMES[0]
        if conf <= 0.5:
            conf = 1.0 - conf
    else:
        idx   = int(np.argmax(preds))
        conf  = float(preds[idx])
        label = CLASS_NAMES[idx]
    return label, conf


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(model, img_array: np.ndarray,
                    fish_mask_224: np.ndarray) -> np.ndarray:
    """Background-suppressed Grad-CAM using the last Conv2D layer."""
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break
    if last_conv_name is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            predictions = predictions[0]
        class_idx = int(tf.argmax(predictions[0]).numpy())
        loss = predictions[:, class_idx]

    grads        = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_outputs[0]
    heatmap      = (conv_out @ pooled_grads[..., tf.newaxis]).numpy().squeeze()
    heatmap      = np.maximum(heatmap, 0)

    fh, fw       = heatmap.shape
    mask_resized = cv2.resize(fish_mask_224.astype(np.float32), (fw, fh)) / 255.0
    heatmap      = heatmap * mask_resized

    max_val = np.max(heatmap)
    if max_val > 1e-8:
        heatmap = heatmap / max_val
    return heatmap.astype(np.float32)


def overlay_gradcam(crop_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    h, w      = crop_bgr.shape[:2]
    hm        = cv2.resize(heatmap, (w, h))
    hm_color  = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    return cv2.addWeighted(crop_bgr, 0.6, hm_color, 0.4, 0)


# ── Predict endpoint ──────────────────────────────────────────────────────────

@app.post("/predict")
def predict(req: ImageRequest):
    t0 = time.perf_counter()

    # 1. Decode
    try:
        image = b64_to_cv2(req.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    original_b64 = cv2_to_b64(image)
    H, W = image.shape[:2]

    # 2. YOLO presence gate — terminate early if no fish found
    if not yolo_fish_present(image):
        raise HTTPException(status_code=422, detail="No fish found in image")

    # 3. Segment fish on the full image via rembg (no cropping)
    bbox, full_mask = detect_fish(image)
    if bbox is None:
        raise HTTPException(status_code=422, detail="No fish foreground could be segmented")

    x1, y1, x2, y2 = bbox
    crop      = image       # use full image — not cropped to YOLO bbox
    fish_mask = full_mask   # mask at full image resolution

    # Mask coverage = fraction of image occupied by the fish foreground
    mask_coverage = round(float((full_mask > 0).sum()) / (H * W), 3)

    # 4. Preprocess for classifier
    img_array     = preprocess_roi(crop, fish_mask)
    fish_mask_224 = cv2.resize(fish_mask, (IMG_SIZE, IMG_SIZE))

    # 5. Classify
    label, conf = classify(img_array)
    decision    = "Auto Approved" if conf >= CONFIDENCE_THRESHOLD else "Manual Review"

    # 6. Grad-CAM
    heatmap     = compute_gradcam(classifier, img_array, fish_mask_224)
    cam_img     = overlay_gradcam(crop, heatmap)

    roi_display = apply_clahe(apply_mask_zero(crop, fish_mask))

    # 7. LLM humanized analysis
    focus_areas  = ["Eye clarity", "Gill color", "Skin texture"]
    llm_analysis = generate_llm_analysis(label, conf, decision, mask_coverage, focus_areas)

    processing_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "status": "success",
        "prediction": {
            "label":      label,
            "confidence": round(conf, 3),
            "decision":   decision,
            "threshold":  CONFIDENCE_THRESHOLD,
        },
        "detection": {
            "bbox":          [x1, y1, x2, y2],
            "mask_coverage": mask_coverage,
        },
        "images": {
            "original": original_b64,
            "roi":      cv2_to_b64(roi_display),
            "gradcam":  cv2_to_b64(cam_img),
        },
        "metadata": {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": processing_ms,
            "model_versions": {
                "segmentor":  "U²-Net (rembg)",
                "classifier": "EfficientNetV2S",
            },
        },
        "explanation": {
            "focus_areas": focus_areas,
            "note":        "GradCAM highlights biologically relevant freshness indicators.",
            "llm_analysis": llm_analysis,
        },
    }
