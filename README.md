<div align="center">

# FreshlyFishy

### AI-Powered Fish Freshness Detection

*Two-stage computer vision pipeline — YOLOv8 detection → EfficientNetV2S classification → GradCAM explainability*

**Team:** Koustav Manna · Satyam Singh

---

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.4-00FFFF?style=flat&logo=ultralytics&logoColor=black)](https://ultralytics.com)
[![Next.js](https://img.shields.io/badge/Next.js-16.2-000000?style=flat&logo=nextdotjs&logoColor=white)](https://nextjs.org)

</div>

---

## Overview

FreshlyFishy is a real-time fish freshness analysis system built on a two-stage deep learning pipeline. Given a photograph of a fish — from a webcam, a phone camera, or a file upload — the system:

1. **Validates** that a fish is actually present using YOLOv8 as a lightweight presence gate — if no fish is detected the request is immediately rejected with a clear error; no further compute is spent.
2. **Segments** the fish body at pixel level on the full image using U²-Net (rembg) to produce a precise foreground mask.
3. **Classifies** freshness on the full (unsegmented) image using a fine-tuned EfficientNetV2S model trained on 22,000 fish images, with background blurring applied via the U²-Net mask.
4. **Explains** the decision through background-suppressed Grad-CAM, highlighting the exact anatomical features (eyes, gills, skin texture) that drove the prediction.
5. **Humanises** the result via an LLM (Groq `llama-3.3-70b` → Gemini `gemini-1.5-flash` fallback) that converts the raw model output into plain-English advice written for a market vendor or consumer.

The result is returned in under 5 seconds with a **Fresh / Not Fresh** verdict, a calibrated confidence score, and three visualisation images — all rendered in a modern immersive web UI with a full landing page, animated pipeline visualisation, and an interactive Grad-CAM demo.

The UI also enforces a **human-in-the-loop safety gate**: if the model's confidence falls below **70%**, the result is flagged as unreliable and a "Human Intervention Required" warning is surfaced to the user, preventing automated decisions on low-confidence predictions.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        INPUT IMAGE                           │
│              (webcam frame · uploaded photo)                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│          STAGE 1 — PRESENCE GATE  (YOLOv8m)                  │
│                                                              │
│  • Single pass  (imgsz=640, conf=0.20)                       │
│  • Checks for at least one fish detection                    │
│  • No cropping — bounding box is NOT used downstream         │
│                                                              │
│  Fish found?  NO  → 422 "No fish found in image"  (stop)    │
│              YES  → continue to Stage 2                      │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│          STAGE 2 — SEGMENTATION  (U²-Net / rembg)            │
│                                                              │
│  • U²-Net salient-object segmentation on the FULL image      │
│  • Alpha-channel thresholding → binary fish mask             │
│  • Morphological close + open → clean foreground             │
│                                                              │
│  Output: pixel-accurate fish mask at full image resolution   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│          STAGE 3 — PREPROCESSING                             │
│                                                              │
│  • Background blur   (preserves training distribution)       │
│  • CLAHE contrast enhancement on the fish region             │
│  • Resize to 224 × 224                                       │
│                                                              │
│  Output: (1, 224, 224, 3) float32 tensor                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│          STAGE 4 — CLASSIFICATION  (EfficientNetV2S)         │
│                                                              │
│  • Runs on the FULL image (background blurred via U²-Net     │
│    mask, CLAHE applied to fish region)                       │
│  • ImageNet-pretrained backbone + custom classification head │
│  • Softmax output: [P(fresh), P(not_fresh)]                  │
│  • confidence ≥ 0.70 → Auto Approved / Manual Review        │
│  • confidence < 0.70 → Human Intervention Required          │
│                                                              │
│  Output: label · confidence · decision                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│    STAGE 5 — EXPLAINABILITY  (Background-Suppressed Grad-CAM)│
│                                                              │
│  • Gradients w.r.t. last Conv2D → class activation map      │
│  • Element-wise multiply by fish mask → background zeroed   │
│  • Jet colormap overlay on the fish crop                     │
│                                                              │
│  Output: heatmap highlighting eyes · gills · skin texture   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│    STAGE 6 — LLM HUMANISATION  (Groq / Gemini)               │
│                                                              │
│  • label + confidence + decision + focus_areas → prompt      │
│  • Primary: Groq  llama-3.3-70b-versatile                    │
│  • Fallback: Gemini gemini-1.5-flash (REST)                  │
│  • Static fallback if both APIs are unavailable              │
│                                                              │
│  Output: 2-3 sentence plain-English report for end user      │
└──────────────────────────────────────────────────────────────┘
```

---

## How the Models Work

### YOLOv8 — Real-Time Fish Detection

YOLOv8 (You Only Look Once, version 8) is a single-stage object detector built by Ultralytics. Unlike two-stage detectors (e.g. Faster R-CNN) that first propose candidate regions then classify them, YOLO processes the entire image in a single forward pass — which is why it can run at 135+ FPS.

**Architecture:**

```
Input image (1280×1280)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  BACKBONE — CSPDarknet                                    │
│                                                           │
│  A deep convolutional network with Cross-Stage Partial    │
│  (CSP) connections. CSP splits each stage's feature map   │
│  into two paths, processes one, then concatenates both.   │
│  This halves gradient duplication and speeds training.    │
│                                                           │
│  Produces multi-scale feature maps at 3 strides:         │
│  80×80 (small objects) · 40×40 (medium) · 20×20 (large)  │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  NECK — Path Aggregation Network (PANet)                  │
│                                                           │
│  Fuses the three scale-levels bidirectionally:            │
│  • Top-down path: deep semantics flow to shallow layers   │
│  • Bottom-up path: fine spatial detail flows upward       │
│  Result: each scale level sees both "what" and "where"    │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  HEAD — Anchor-Free Decoupled Detection                   │
│                                                           │
│  YOLOv8 dropped anchor boxes entirely. For every cell in  │
│  the feature grid, the head directly regresses:           │
│  • (x, y, w, h) as distance from the cell centre         │
│  • Class probability (using Distribution Focal Loss)      │
│                                                           │
│  A separate classification branch and regression branch   │
│  operate in parallel — "decoupled" — improving both       │
│  localisation and classification accuracy vs v5/v6.       │
└───────────────────────────────────────────────────────────┘
```

**In FreshlyFishy:** The model runs a high-resolution pass (imgsz=1280) first for accuracy, falls back to TTA and tiled SAHI passes for hard cases, and outputs a single tight bounding box per fish. That crop — and nothing else — is passed to the segmentor.

---

### rembg / U²-Net — Pixel-Level Fish Segmentation

rembg is a background removal library that wraps **U²-Net**, a nested U-Net architecture designed for *salient object detection* — finding the visually dominant object in a scene and producing a per-pixel foreground mask.

**Architecture:**

```
Input fish crop (arbitrary resolution)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  U²-Net — Nested U-Structure                              │
│                                                           │
│  A standard U-Net has one encoder-decoder. U²-Net nests   │
│  a full U-Net inside every encoder and decoder block.     │
│  Each nested block is called an RSU (Residual U-block):   │
│                                                           │
│       Input → [small U-Net with 4–7 depth levels] → Out  │
│                                                           │
│  These RSU blocks capture both fine local detail and      │
│  large global context within a single feature stage.      │
│                                                           │
│  The outer U-Net has 6 encoder + 6 decoder stages.        │
│  Skip connections carry spatial detail across the         │
│  encoder-decoder gap at every scale.                      │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  OUTPUT — Side Supervision Fusion                         │
│                                                           │
│  Each decoder stage produces its own saliency map.        │
│  All six maps are upsampled to input resolution and        │
│  summed into a final probability map.                     │
│  Sigmoid activation → pixel values in [0, 1]             │
│                                                           │
│  rembg thresholds this at 127/255, applies morphological  │
│  close + open, and returns an RGBA image with the         │
│  background set to transparent (alpha = 0).               │
└───────────────────────────────────────────────────────────┘
```

**In FreshlyFishy:** The alpha channel is extracted, thresholded to binary, and used as a mask in two ways — (1) the background is Gaussian-blurred rather than zeroed before classification, and (2) the same mask suppresses the Grad-CAM heatmap so attention stays on the fish body.

---

### EfficientNetV2S — Freshness Classification

EfficientNetV2S is a compact, fast convolutional network from Google Brain that achieves state-of-the-art accuracy through *compound scaling* — simultaneously widening, deepening, and increasing the input resolution by a mathematically derived ratio rather than scaling one dimension at a time.

**Architecture:**

```
Input (224×224×3, pre-normalised by include_preprocessing=True)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│  STEM — 3×3 Conv, stride 2 → 24 channels                 │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  EARLY STAGES — Fused-MBConv blocks                       │
│                                                           │
│  In EfficientNetV1, early layers used depthwise-separable │
│  convolutions (MBConv). V2 replaces the first few stages  │
│  with Fused-MBConv: a single 3×3 conv + pointwise conv.  │
│                                                           │
│  Why: on modern accelerators (TPUs/GPUs), fused ops run   │
│  faster than depthwise-separable at small feature maps.   │
│                                                           │
│  Stages 1–3: Fused-MBConv, expansion ratio 1–4           │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  LATER STAGES — MBConv blocks with SE attention           │
│                                                           │
│  Each block: pointwise expand → depthwise 3×3 → SE → pw  │
│                                                           │
│  Squeeze-and-Excitation (SE): global-average-pools the    │
│  feature map to a vector, passes it through two small     │
│  dense layers, and uses the output to re-weight each      │
│  channel — the network learns which feature maps matter   │
│  most for the current input.                              │
│                                                           │
│  Stages 4–6: MBConv, expansion ratio 4–6                 │
└───────────────────────┬───────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│  CUSTOM CLASSIFICATION HEAD (FreshlyFishy)                │
│                                                           │
│  GlobalAveragePooling2D                                   │
│        ↓                                                  │
│  BatchNormalization                                       │
│        ↓                                                  │
│  Dense(512, activation='gelu')                            │
│        ↓                                                  │
│  Dense(256, activation='gelu')                            │
│        ↓                                                  │
│  Dense(2,  activation='softmax')  → [P(fresh), P(not)]   │
│                                                           │
│  GELU (Gaussian Error Linear Unit) is smoother than ReLU  │
│  and works better with BatchNorm in fine-tuning regimes.  │
└───────────────────────────────────────────────────────────┘
```

**Training strategy:** Phase 1 freezes the backbone and trains only the head (25 epochs, lr=3e-4) to warm up the random weights without destroying ImageNet features. Phase 2 unfreezes the top 60 layers and jointly fine-tunes with a cosine-decayed learning rate (5e-5 → 0) to specialise the high-level features for fish texture.

**In FreshlyFishy:** The classifier receives a 224×224 crop of the segmented fish (background blurred, CLAHE applied) and outputs a softmax probability. The frontend applies a two-tier confidence gate:

| Confidence | Decision | UI treatment |
|---|---|---|
| ≥ 70% and high certainty | Auto Approved | Green verdict card |
| ≥ 70% but borderline | Manual Review | Yellow badge |
| < 70% | **Human Intervention Required** | Amber warning banner — result must not be acted on without expert review |

The 70% threshold was chosen because below this level the softmax probability distributions are wide enough that the model's own uncertainty renders the classification unreliable for safety-critical food-quality decisions.

---

## Tech Stack

### Backend

| Component | Technology | Version |
|---|---|---|
| API server | FastAPI + Uvicorn | 0.135 / 0.43 |
| Fish detection | YOLOv8m (Ultralytics) | 8.4.33 |
| Background segmentation | rembg (U²-Net) | 2.0.74 |
| ONNX runtime (rembg backend) | onnxruntime | 1.24+ |
| Freshness classification | EfficientNetV2S (TensorFlow / Keras) | 2.21.0 |
| LLM humanisation (primary) | Groq — `llama-3.3-70b-versatile` | groq ≥ 0.9 |
| LLM humanisation (fallback) | Google Gemini — `gemini-1.5-flash` | REST API |
| Environment config | python-dotenv | 1.0+ |
| Image processing | OpenCV · NumPy · Pillow | |
| Dependency management | uv | |

### Frontend

| Component | Technology | Version |
|---|---|---|
| Framework | Next.js (App Router) | 16.2 |
| UI library | React | 19.2 |
| Animations | Framer Motion | 12.x |
| Icons | Lucide React | 1.x |
| Styling | Tailwind CSS v4 | |
| Font | Geist Sans / Geist Mono | |

---

## Repository Structure

```
freshlyfishy2/
├── backend/
│   ├── models/
│   │   ├── yolo_detection_model.pt          ← trained YOLOv8 fish detector
│   │   └── fish_classifier.keras            ← fine-tuned EfficientNetV2S
│   ├── main.py                              ← FastAPI app & full inference pipeline
│   ├── prepare_training_data.py             ← offline preprocessing for training data
│   ├── fish-classifier-final.ipynb          ← EfficientNetV2S training (Kaggle)
│   ├── build_yolo_notebook.py               ← YOLOv8 training notebook builder
│   ├── pyproject.toml                       ← Python dependencies (uv)
│   └── start.bat                            ← one-click server start (Windows)
│
└── frontend/
    └── src/app/
        ├── page.tsx                         ← landing page + scanner app (single file)
        │                                      sections: Hero · Scanner Visual · Metrics
        │                                               Pipeline Flow · GradCAM Demo · App
        ├── layout.tsx                       ← root layout
        └── globals.css                      ← Tailwind + custom animations (ocean theme)
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) package manager — `pip install uv`
- A CUDA-capable GPU is recommended for fast inference; CPU works but is slower (~20 s/image)

---

### 1 — Backend

```bash
cd backend

# Install all Python dependencies
uv sync

# ONNX runtime is required by rembg (choose one)
uv pip install onnxruntime         # CPU — works on any machine
uv pip install onnxruntime-gpu     # NVIDIA GPU — requires matching CUDA/cuDNN
```

**Start the server**

```bash
# Windows — double-click or run in any terminal
start.bat

# Any platform
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> **First startup takes 30–60 seconds.** TensorFlow and the U²-Net ONNX model both initialise on first load. Wait for:
>
> ```
> ✅ Classifier loaded — server is ready!
> INFO:     Application startup complete.
> INFO:     Uvicorn running on http://0.0.0.0:8000
> ```
>
> Then open the frontend.

---

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev               # http://localhost:3000  (development)

# or for production
npm run build && npm start
```

---

## LLM Integration — Humanised Analysis

After the vision pipeline produces a verdict, FreshlyFishy passes the structured result to a large language model that writes a short plain-English report for a market vendor or consumer. The report is returned as `explanation.llm_analysis` in every `/predict` response.

### How it works

```
label + confidence + decision + mask_coverage + focus_areas
                        │
                        ▼
            ┌───────────────────────┐
            │   generate_llm_analysis()             │
            │                                       │
            │  1. Try Groq                          │
            │     model: llama-3.3-70b-versatile    │
            │     max_tokens: 160  temp: 0.7        │
            │                                       │
            │  2. Fallback: Gemini REST             │
            │     model: gemini-1.5-flash           │
            │                                       │
            │  3. Static fallback (no API keys)     │
            └───────────────────────┘
                        │
                        ▼
        2-3 sentence plain-English freshness report
```

### Prompt design

The prompt positions the LLM as a fish freshness expert and provides all structured outputs from the vision pipeline:

```
You are a fish freshness expert AI. A vision model analyzed a fish image.

Results:
- Verdict: Fresh
- Confidence: 91.2%
- Decision status: Auto Approved
- Fish body coverage in frame: 38.0%
- Anatomical features examined: Eye clarity, Gill color, Skin texture

Write 2-3 concise sentences for a market vendor or consumer. Explain what
indicators the model observed, what the verdict means practically, and give
a brief recommendation. No bullet points, no headers, no jargon.
```

### Setup

Create a `.env` file in `backend/` with one or both API keys:

```env
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
```

- If **both** keys are present, Groq is tried first (lower latency); Gemini is the fallback.
- If **neither** key is set, a deterministic static summary is returned — the pipeline still works fully, just without LLM-generated prose.
- Get a free Groq key at [console.groq.com](https://console.groq.com)
- Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com)

### Human intervention gate

The frontend enforces an independent confidence threshold on the vision model output:

| Model confidence | Decision label | UI treatment |
|---|---|---|
| ≥ 70%, high certainty | Auto Approved | Green verdict card |
| ≥ 70%, borderline | Manual Review | Yellow badge |
| **< 70%** | **Human Intervention Required** | Amber warning banner — LLM analysis is still shown but the result must not be acted on without expert review |

The LLM analysis is always generated and displayed regardless of confidence level — it provides context even for uncertain results. The amber banner and overridden decision badge make it unambiguous that the model's confidence was insufficient for an automated call.

---

## API Reference

### `POST /predict`

Analyses a fish image and returns a freshness verdict with full visualisations.

**Request body**

```json
{
    "image_base64": "<base64-encoded image — with or without data-URL prefix>"
}
```

**Successful response**

```json
{
    "status": "success",
    "prediction": {
        "label":      "Fresh",
        "confidence": 0.912,
        "decision":   "Auto Approved",
        "threshold":  0.75
    },
    "detection": {
        "bbox":          [42, 110, 580, 430],
        "mask_coverage": 0.38
    },
    "images": {
        "original": "<base64 JPEG>",
        "roi":      "<base64 JPEG — segmented fish crop>",
        "gradcam":  "<base64 JPEG — attention heatmap overlay>"
    },
    "metadata": {
        "timestamp":          "2025-04-04T10:22:05.123Z",
        "processing_time_ms": 1840,
        "model_versions": {
            "segmentor":  "U²-Net (rembg)",
            "classifier": "EfficientNetV2S"
        }
    },
    "explanation": {
        "focus_areas": ["Eye clarity", "Gill color", "Skin texture"],
        "note": "GradCAM highlights biologically relevant freshness indicators.",
        "llm_analysis": "The model examined the eye clarity, gill coloration, and skin texture of this fish and found all indicators consistent with a fresh catch. With 91.2% confidence the fish is safe for consumption or sale. It can be refrigerated and used within 1-2 days without concern."
    }
}
```

**Error responses**

| HTTP Status | Condition |
|---|---|
| `400` | Image cannot be decoded (corrupt file or unsupported format) |
| `422` | No fish foreground detected in the image |
| `500` | Internal model error |

Interactive Swagger docs are available at **`http://localhost:8000/docs`** once the server is running.

---

## Model Training

### Classifier — `fish-classifier-final.ipynb`

Designed to run on **Kaggle** with a T4 GPU.

| Parameter | Value |
|---|---|
| Dataset | 9,000 fresh · 13,000 not fresh (22,000 total raw images) |
| Backbone | EfficientNetV2S, ImageNet pretrained, `include_preprocessing=True` |
| Head | GAP → BatchNorm → Dense 512 GELU → Dense 256 GELU → Softmax 2 |
| Class weights | `total / (n_classes × n_per_class)` — corrects for 13k/9k imbalance |
| Augmentation | RandomFlip · RandomRotation ±10° · RandomZoom · RandomTranslation · RandomBrightness ±25% · RandomContrast |
| Phase 1 | 25 epochs, frozen backbone, AdamW lr=3e-4, label smoothing 0.05 |
| Phase 2 | 25 epochs, top-60 layers unfrozen, cosine decay 5e-5 → 0 |
| Monitored metric | `val_AUC` (more robust than val_loss on imbalanced data) |
| Evaluation | Confusion matrix · per-class F1 · ROC-AUC · GradCAM verification |
| Output | `/kaggle/working/fish_classifier.keras` |

After training: download the file and replace `backend/models/fish_classifier.keras`.

---

### Detector — `build_yolo_notebook.py`

Designed to run on **Kaggle** with a T4 GPU.

YOLOv8 is the **primary detection and cropping stage** of the FreshlyFishy pipeline. Its job is to locate the fish in the input image and return a tight bounding box, isolating the subject from complex backgrounds before the segmentation and classification stages run.

The notebook uses **YOLOWorld** (open-vocabulary, built into Ultralytics) to auto-annotate all 22,000 existing classification images with bounding boxes — eliminating the need for any manual annotation.

| Parameter | Value |
|---|---|
| Auto-labeler | `yolov8l-worldv2.pt` · class prompt: `"fish"` · conf ≥ 0.20 |
| Label quality filters | Min bbox area ≥ 3% of image · aspect ratio ≥ 1.2 · max 6 fish/image |
| Base model | `yolov8l.pt` (COCO pretrained, 25M parameters) |
| Image size | 1280 px |
| Epochs | 100 with early stopping (patience 20) |
| Augmentation | Mosaic · copy-paste · random flip/rotation ±45° · scale · HSV jitter |
| Loss weights | box=7.5 · cls=0.5 · dfl=1.5 |
| Evaluation | mAP@50 · mAP@50-95 · Precision · Recall · PR curve |
| Output | `/kaggle/working/yolo_fish_detector_best.pt` |

After training:
1. Download `yolo_fish_detector_best.pt`
2. Rename to `yolo_detection_model.pt`
3. Replace `backend/models/yolo_detection_model.pt`

---

## Model Performance

### YOLOv8 Detection Results

Evaluated on **641 images / 866 instances** from the held-out test split.

| Metric | Value | Interpretation |
|---|---|---|
| Precision | **0.956** | 95.6% of detections are true fish — almost no false alarms |
| Recall | **0.875** | Model finds 87.5% of all fish present in the scene |
| mAP@50 | **0.938** | Primary accuracy metric — industry-level bounding-box quality |
| mAP@50-95 | **0.757** | Strong performance even under strict IoU thresholds |
| Inference speed | **7.4 ms / image** | ~135 FPS throughput — real-time capable |

**Strengths:** Very high precision means the detector almost never fires on non-fish objects. The high mAP@50 confirms bounding boxes are tight and well-positioned.

**Note on recall:** The slightly lower recall (87.5%) means the detector occasionally misses small or partially visible fish. Because YOLO is the first gate in the pipeline — a miss here means no crop, no classification — but the miss rate is low enough that overall system reliability remains high.

<img src="backend/images/metrics/WhatsApp Image 2026-04-04 at 9.56.39 PM.jpeg" alt="YOLO detection metrics — precision, recall, mAP curves" width="800"/>

### EfficientNetV2S Classifier Results

Evaluated on the held-out test split.

| Metric | Value |
|---|---|
| Test accuracy | **94.378%** |
| ROC-AUC | 0.982 |
| Weighted F1 | 0.914 |

<img src="backend/images/metrics/model accuracy.jpeg" alt="EfficientNetV2S training accuracy and loss curves" width="800"/>

---

## Dataset

The classification dataset is publicly available on Kaggle:

> **`satyamsingh0912 / fish-classifier-dataset`**

```
final_dataset/
├── train/
│   ├── fresh/         ~7,200 images
│   └── not_fresh/     ~10,400 images
├── val/
│   ├── fresh/
│   └── not_fresh/
└── test/
    ├── fresh/
    └── not_fresh/
```

The same image set (resplit) is used by the YOLOv8 training notebook after auto-labeling with YOLOWorld.

---

## Key Design Decisions

**Why background-suppressed Grad-CAM?**
Standard Grad-CAM on fish images activates heavily on tank walls, ice, and packaging — not the fish itself. Multiplying the heatmap element-wise by the U²-Net foreground mask zeros all background activations, forcing the explanation to reflect only the anatomical indicators (eye clarity, gill color, skin texture) that are biologically meaningful for freshness assessment.

**Why blur the background instead of zeroing it?**
Zeroing the background (setting pixels to black) creates a strong out-of-distribution signal that degrades classifier confidence. Gaussian blur keeps the non-fish region close to the training distribution while still reducing its saliency to the model.

**Why add an LLM layer on top of the vision pipeline?**
The raw outputs of a computer vision model — a label, a probability, and a heatmap — are meaningful to an ML engineer but opaque to a fish market vendor or a consumer deciding whether to buy. The LLM translation layer converts those structured signals into a short, natural-language recommendation that non-technical users can act on immediately. Critically, the LLM receives only the *already-computed* vision outputs as its input; it does not see or re-analyse the image. This keeps the LLM in a strict "interpreter" role rather than a decision-maker, so its output cannot override or inflate the confidence gate.

**Why Groq (llama-3.3-70b) as the primary LLM?**
Groq's inference hardware delivers very low latency on large models, keeping the LLM step well within the overall 5-second response budget. The 70B parameter Llama 3.3 model produces fluent, contextually appropriate prose from the short structured prompt. Gemini 1.5 Flash is kept as a REST-based fallback in case the Groq service is unavailable, and a deterministic static template is the final fallback so the API never fails due to an LLM outage.

**Why use YOLOv8 only as a presence gate and not for cropping?**
The original design cropped the image to the YOLO bounding box before classification. This caused a subtle problem: EfficientNetV2S was trained on full fish images, so a tightly cropped patch is a mild out-of-distribution input. By keeping the full image and using U²-Net's pixel mask only to suppress the background via blurring, the classifier receives an input much closer to its training distribution. YOLO's role becomes a cheap, fast binary filter — it prevents the more expensive rembg + EfficientNetV2S + GradCAM chain from ever running on images that contain no fish, saving significant compute for obviously invalid inputs.

**Why a dedicated YOLO gate at all rather than relying on rembg to signal "no fish"?**
rembg (U²-Net) will always return *some* foreground mask, even on an image containing only a hand, a label, or an empty plate — it finds the most salient object regardless of class. Without a class-aware gate, those images would pass through the full pipeline and receive nonsensical freshness predictions. YOLO was trained specifically on fish, so it provides the semantic check that rembg cannot.

**Why is the human intervention threshold set at 70% confidence?**
Below 70% the softmax output distributions are wide — the model is in genuine doubt between classes. At these confidence levels small perturbations (lighting, angle, compression artefacts) can flip the verdict, making any automated decision unsafe for a food-quality context. Rather than silently returning an unreliable result, the UI renders an amber warning banner and overrides the decision badge to "Human Intervention Needed", ensuring a qualified person reviews the image before any action is taken. Above 70% the model's track record (94.38% test accuracy) justifies automated trust.
