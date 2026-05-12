"""Generates yolo-fish-detector.ipynb (run once, then delete this script)."""
import json, textwrap

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(src).lstrip("\n"),
    }

def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(src).lstrip("\n"),
    }

cells = []

# ── 0 ─ Imports ────────────────────────────────────────────────────────────
cells.append(code("""
    import os, math, shutil, yaml, random, time
    from pathlib import Path
    from collections import defaultdict

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    from ultralytics import YOLO, YOLOWorld
    import torch

    print(f"ultralytics version  : {__import__('ultralytics').__version__}")
    print(f"torch version        : {torch.__version__}")
    print(f"CUDA available       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU                  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM                 : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""))

# ── 1 ─ Configuration ──────────────────────────────────────────────────────
cells.append(code("""
    # ── UPDATE THIS PATH ──────────────────────────────────────────────────────
    DATA_DIR = "/kaggle/input/datasets/satyamsingh0912/fish-classifier-dataset/final_dataset"
    # ─────────────────────────────────────────────────────────────────────────

    WORK_DIR    = Path("/kaggle/working")
    DS_DIR      = WORK_DIR / "yolo_dataset"   # final YOLO dataset
    FLAT_DIR    = WORK_DIR / "_flat_images"   # scratch: all images in one folder

    # ── Auto-labeling ─────────────────────────────────────────────────────────
    LABEL_CONF       = 0.20     # YOLOWorld confidence threshold
    LABEL_IOU        = 0.45     # NMS IoU for auto-labeling
    LABEL_IMGSZ      = 1280     # high res catches small fish better
    MIN_BBOX_AREA    = 0.03     # bbox must cover ≥ 3 % of image (rejects eye-only boxes)
    MIN_ASPECT_RATIO = 1.2      # longer side / shorter side ≥ 1.2 (fish are elongated)
    MAX_FISH_PER_IMG = 6        # ignore images with implausible number of fish

    # ── Dataset split ─────────────────────────────────────────────────────────
    TRAIN_FRAC = 0.80
    VAL_FRAC   = 0.10
    # test = remainder
    SEED       = 42

    # ── Training ──────────────────────────────────────────────────────────────
    BASE_MODEL  = "yolov8l.pt"  # large: best accuracy on T4; swap to yolov8m.pt if OOM
    EPOCHS      = 100
    TRAIN_IMGSZ = 1280
    BATCH       = 8             # 4 for l on <10 GB VRAM, 8 for m

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
"""))

# ── 2 ─ Collect images ─────────────────────────────────────────────────────
cells.append(code("""
    # Collect every image from the classification dataset regardless of split or class
    raw_root = Path(DATA_DIR)
    all_images = sorted([
        p for p in raw_root.rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    ])
    print(f"Total images discovered: {len(all_images):,}")

    # Count per class (for reference — detection training ignores class labels)
    by_class = defaultdict(int)
    for p in all_images:
        by_class[p.parent.name] += 1
    for cls, n in sorted(by_class.items()):
        print(f"  {cls}: {n:,}")
"""))

# ── 3 ─ Copy to flat scratch folder ───────────────────────────────────────
cells.append(code("""
    # Flatten into one folder (avoid name collisions by prefixing class)
    FLAT_DIR.mkdir(parents=True, exist_ok=True)
    flat_paths = []
    seen_names = set()

    for src in all_images:
        stem = f"{src.parent.name}__{src.stem}"
        dst  = FLAT_DIR / (stem + src.suffix.lower())
        if dst.name in seen_names:
            stem += f"_{hash(str(src)) & 0xFFFF:04x}"
            dst = FLAT_DIR / (stem + src.suffix.lower())
        seen_names.add(dst.name)
        if not dst.exists():
            shutil.copy2(src, dst)
        flat_paths.append(dst)

    print(f"Flat image folder    : {FLAT_DIR}")
    print(f"Images copied        : {len(flat_paths):,}")
"""))

# ── 4 ─ Auto-label with YOLOWorld ─────────────────────────────────────────
cells.append(code("""
    # YOLOWorld is an open-vocabulary detector built into ultralytics.
    # We use it to detect "fish" in every image WITHOUT manual annotations.
    # It is much faster than GroundingDINO (~30 ms/image on T4 vs ~150 ms).

    print("Loading YOLOWorld labeler …")
    labeler = YOLOWorld("yolov8l-worldv2.pt")
    labeler.set_classes(["fish"])
    print("YOLOWorld ready.  Starting auto-labeling …")

    RAW_LABELS_DIR = WORK_DIR / "_raw_labels"
    RAW_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    kept, rejected_conf, rejected_shape, no_det = 0, 0, 0, 0
    label_records = []   # list of (image_path, [(x_c, y_c, w, h), ...])

    t0 = time.time()
    for idx, img_path in enumerate(flat_paths):
        if idx % 500 == 0 and idx > 0:
            elapsed = time.time() - t0
            eta = elapsed / idx * (len(flat_paths) - idx)
            print(f"  {idx:>6}/{len(flat_paths)}  "
                  f"kept={kept}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = labeler.predict(
            source=img,
            conf=LABEL_CONF,
            iou=LABEL_IOU,
            imgsz=LABEL_IMGSZ,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            no_det += 1
            continue

        # Convert detections to YOLO normalised format and apply quality filters
        good_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1

            # Filter 1: bbox must cover enough of the image
            area_ratio = (bw * bh) / (w * h)
            if area_ratio < MIN_BBOX_AREA:
                rejected_shape += 1
                continue

            # Filter 2: fish bodies are elongated, not square
            longer  = max(bw, bh)
            shorter = min(bw, bh)
            if (longer / (shorter + 1e-6)) < MIN_ASPECT_RATIO:
                rejected_shape += 1
                continue

            # Normalise to YOLO format
            x_c = (x1 + x2) / 2 / w
            y_c = (y1 + y2) / 2 / h
            nw  = bw / w
            nh  = bh / h
            good_boxes.append((x_c, y_c, nw, nh))

        if len(good_boxes) == 0 or len(good_boxes) > MAX_FISH_PER_IMG:
            rejected_conf += 1
            continue

        # Write YOLO label file
        label_path = RAW_LABELS_DIR / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            for x_c, y_c, nw, nh in good_boxes:
                f.write(f"0 {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}\\n")

        label_records.append((img_path, good_boxes))
        kept += 1

    total_elapsed = time.time() - t0
    print(f"\\nAuto-labeling complete in {total_elapsed/60:.1f} minutes")
    print(f"  Kept (good labels)      : {kept:,}")
    print(f"  Rejected (no detection) : {no_det:,}")
    print(f"  Rejected (shape filter) : {rejected_shape:,}")
    print(f"  Total processed         : {len(flat_paths):,}")
    print(f"\\nLabel quality: {kept/len(flat_paths)*100:.1f}% of images had a valid fish box")
"""))

# ── 5 ─ Visualise sample labels ────────────────────────────────────────────
cells.append(code("""
    def draw_boxes(img_bgr, yolo_boxes, color=(0, 255, 0), thickness=3):
        \"\"\"Draw YOLO-normalised boxes on an image.  Returns RGB copy.\"\"\"
        img = img_bgr.copy()
        h, w = img.shape[:2]
        for x_c, y_c, bw, bh in yolo_boxes:
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sample = random.sample(label_records, min(12, len(label_records)))
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for ax, (img_path, boxes) in zip(axes.flat, sample):
        img = cv2.imread(str(img_path))
        ax.imshow(draw_boxes(img, boxes))
        ax.set_title(f"{img_path.stem[:30]}\\n{len(boxes)} fish", fontsize=8)
        ax.axis("off")
    plt.suptitle("Auto-label quality check — verify boxes cover whole fish bodies",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print("If boxes look wrong (eye-only, background), reduce LABEL_CONF or increase MIN_BBOX_AREA.")
"""))

# ── 6 ─ Build YOLO dataset structure ───────────────────────────────────────
cells.append(code("""
    # Shuffle and split the labeled images into train / val / test
    random.seed(SEED)
    shuffled = label_records.copy()
    random.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    splits  = {
        "train": shuffled[:n_train],
        "val"  : shuffled[n_train : n_train + n_val],
        "test" : shuffled[n_train + n_val:],
    }
    for split, records in splits.items():
        print(f"  {split:5s}: {len(records):,} images")

    # Create directory tree
    for split in splits:
        (DS_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DS_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy images and labels
    for split, records in splits.items():
        img_dst_dir = DS_DIR / "images" / split
        lbl_dst_dir = DS_DIR / "labels" / split
        for img_path, _ in records:
            shutil.copy2(img_path, img_dst_dir / img_path.name)
            lbl_src = RAW_LABELS_DIR / (img_path.stem + ".txt")
            shutil.copy2(lbl_src, lbl_dst_dir / (img_path.stem + ".txt"))

    print(f"\\nDataset written to: {DS_DIR}")

    # data.yaml
    data_yaml = {
        "path"  : str(DS_DIR),
        "train" : "images/train",
        "val"   : "images/val",
        "test"  : "images/test",
        "nc"    : 1,
        "names" : {0: "fish"},
    }
    yaml_path = DS_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"data.yaml written  : {yaml_path}")

    # Quick sanity count
    for split in splits:
        n_imgs = len(list((DS_DIR / "images" / split).iterdir()))
        n_lbls = len(list((DS_DIR / "labels" / split).iterdir()))
        print(f"  {split:5s}: {n_imgs} images, {n_lbls} labels")
"""))

# ── 7 ─ Analyse label statistics ───────────────────────────────────────────
cells.append(code("""
    # Visualise bbox area and aspect ratio distributions to confirm data health
    areas, aspects = [], []
    for img_path, boxes in label_records:
        for x_c, y_c, bw, bh in boxes:
            areas.append(bw * bh)
            longer  = max(bw, bh)
            shorter = min(bw, bh) + 1e-6
            aspects.append(longer / shorter)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(areas, bins=60, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].axvline(MIN_BBOX_AREA, color="red", linestyle="--", label=f"Min area={MIN_BBOX_AREA}")
    axes[0].set_title("Bbox area (fraction of image)")
    axes[0].set_xlabel("Area ratio"); axes[0].legend()

    axes[1].hist(aspects, bins=60, color="seagreen", edgecolor="white", linewidth=0.5)
    axes[1].axvline(MIN_ASPECT_RATIO, color="red", linestyle="--",
                    label=f"Min aspect={MIN_ASPECT_RATIO}")
    axes[1].set_title("Bbox aspect ratio (longer / shorter)")
    axes[1].set_xlabel("Aspect ratio"); axes[1].legend()

    fish_per_img = [len(boxes) for _, boxes in label_records]
    axes[2].hist(fish_per_img, bins=range(1, MAX_FISH_PER_IMG + 2),
                 color="coral", edgecolor="white", linewidth=0.5, align="left")
    axes[2].set_title("Fish per image")
    axes[2].set_xlabel("# fish in label")
    axes[2].set_xticks(range(1, MAX_FISH_PER_IMG + 1))

    plt.suptitle("Label Distribution Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print(f"Median bbox area   : {np.median(areas)*100:.1f}% of image")
    print(f"Median aspect ratio: {np.median(aspects):.2f}")
    print(f"Single-fish images : {sum(1 for x in fish_per_img if x==1)/len(fish_per_img)*100:.1f}%")
"""))

# ── 8 ─ Train YOLOv8 ───────────────────────────────────────────────────────
cells.append(code("""
    model = YOLO(BASE_MODEL)   # loads COCO-pretrained weights

    print("=" * 70)
    print(f"Training YOLOv8l   imgsz={TRAIN_IMGSZ}   epochs={EPOCHS}   batch={BATCH}")
    print("=" * 70)

    train_results = model.train(
        data    = str(DS_DIR / "data.yaml"),
        epochs  = EPOCHS,
        imgsz   = TRAIN_IMGSZ,
        batch   = BATCH,
        device  = 0,
        workers = 4,

        # ── Augmentation (tuned for fish images) ─────────────────────────────
        fliplr       = 0.5,     # fish face left or right equally
        flipud       = 0.3,     # fish photographed upside-down is common
        degrees      = 45.0,    # fish orientation varies widely
        translate    = 0.2,     # random position shift
        scale        = 0.9,     # simulate different distances
        shear        = 5.0,     # slight shear distortion
        perspective  = 0.0005,  # mild perspective warp
        hsv_h        = 0.015,   # slight hue shift (water tint)
        hsv_s        = 0.70,    # strong saturation jitter (flash/market lighting)
        hsv_v        = 0.40,    # brightness variation
        mosaic       = 1.0,     # mosaic augmentation (multiple fish at once)
        mixup        = 0.10,    # mild mixup
        copy_paste   = 0.30,    # paste fish onto new backgrounds
        erasing      = 0.40,    # random erasing (occlusion robustness)
        close_mosaic = 10,      # disable mosaic in last 10 epochs for stability

        # ── Optimiser ────────────────────────────────────────────────────────
        optimizer      = "AdamW",
        lr0            = 0.001,
        lrf            = 0.01,   # final lr = lr0 * lrf
        momentum       = 0.937,
        weight_decay   = 0.0005,
        warmup_epochs  = 3,
        warmup_momentum= 0.8,

        # ── Loss weights (emphasise localisation for precise bbox) ────────────
        box    = 7.5,   # box regression loss weight (default=7.5, keep high)
        cls    = 0.5,   # classification loss weight (1 class → lower is fine)
        dfl    = 1.5,   # distribution focal loss weight

        # ── Misc ─────────────────────────────────────────────────────────────
        patience   = 20,          # early stopping
        save       = True,
        save_period= 10,
        plots      = True,
        amp        = True,        # automatic mixed precision
        project    = str(WORK_DIR),
        name       = "fish_detector",
        exist_ok   = True,
    )

    best_model_path = WORK_DIR / "fish_detector" / "weights" / "best.pt"
    print(f"\\nBest model: {best_model_path}")
"""))

# ── 9 ─ Training curves ────────────────────────────────────────────────────
cells.append(code("""
    import pandas as pd

    results_csv = WORK_DIR / "fish_detector" / "results.csv"
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    # ── Select metrics that exist in results.csv ───────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flat

    plot_pairs = [
        ("train/box_loss",  "val/box_loss",  "Box Loss"),
        ("train/cls_loss",  "val/cls_loss",  "Class Loss"),
        ("train/dfl_loss",  "val/dfl_loss",  "DFL Loss"),
        ("metrics/mAP50(B)","metrics/mAP50(B)", "mAP@50"),
        ("metrics/mAP50-95(B)","metrics/mAP50-95(B)", "mAP@50-95"),
        ("lr/pg0",           None,             "Learning Rate"),
    ]

    for ax, (train_col, val_col, title) in zip(axes, plot_pairs):
        if train_col in df.columns:
            ax.plot(df["epoch"], df[train_col], label="train", linewidth=2)
        if val_col and val_col != train_col and val_col in df.columns:
            ax.plot(df["epoch"], df[val_col], label="val", linewidth=2, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("YOLOv8l Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Print peak metrics
    if "metrics/mAP50(B)" in df.columns:
        best_row = df.loc[df["metrics/mAP50(B)"].idxmax()]
        print(f"Best epoch           : {int(best_row['epoch'])}")
        print(f"Best mAP@50          : {best_row['metrics/mAP50(B)']:.4f}")
        print(f"Best mAP@50-95       : {best_row['metrics/mAP50-95(B)']:.4f}")
"""))

# ── 10 ─ Validation metrics ────────────────────────────────────────────────
cells.append(code("""
    best_model = YOLO(str(WORK_DIR / "fish_detector" / "weights" / "best.pt"))
    val_metrics = best_model.val(
        data    = str(DS_DIR / "data.yaml"),
        split   = "test",
        imgsz   = TRAIN_IMGSZ,
        device  = 0,
        plots   = True,
        save_json = True,
    )

    print("\\n── Test-set Results ─────────────────────────────────────────────")
    print(f"  mAP@50     : {val_metrics.box.map50:.4f}")
    print(f"  mAP@50-95  : {val_metrics.box.map:.4f}")
    print(f"  Precision  : {val_metrics.box.mp:.4f}")
    print(f"  Recall     : {val_metrics.box.mr:.4f}")
    print("────────────────────────────────────────────────────────────────")

    # Display PR curve saved by ultralytics
    pr_img_path = WORK_DIR / "fish_detector" / "PR_curve.png"
    if pr_img_path.exists():
        pr_img = cv2.cvtColor(cv2.imread(str(pr_img_path)), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 7))
        plt.imshow(pr_img)
        plt.axis("off")
        plt.title("Precision-Recall Curve", fontsize=13)
        plt.tight_layout()
        plt.show()
"""))

# ── 11 ─ Inference visualisation ───────────────────────────────────────────
cells.append(code("""
    # Visualise predictions on test images (use best.pt)
    test_imgs = sorted((DS_DIR / "images" / "test").iterdir())
    test_imgs = random.sample(test_imgs, min(12, len(test_imgs)))

    CONF_VIS = 0.25   # visualisation confidence threshold
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for ax, img_path in zip(axes.flat, test_imgs):
        img_bgr = cv2.imread(str(img_path))
        h, w    = img_bgr.shape[:2]

        preds = best_model.predict(
            source=img_bgr, conf=CONF_VIS, imgsz=TRAIN_IMGSZ, verbose=False
        )[0]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        if preds.boxes is not None and len(preds.boxes) > 0:
            for box in preds.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_val = float(box.conf)
                rect = Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="lime", facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"fish {conf_val:.2f}",
                        color="lime", fontsize=7, fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1))
            ax.set_title(f"{len(preds.boxes)} fish", fontsize=9, color="green")
        else:
            ax.set_title("no detection", fontsize=9, color="red")

        ax.axis("off")

    plt.suptitle(
        f"Test Predictions (conf ≥ {CONF_VIS}) — best.pt",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
"""))

# ── 12 ─ Export + instructions ─────────────────────────────────────────────
cells.append(code("""
    # Copy best.pt to a clean output location for easy download
    final_out = WORK_DIR / "yolo_fish_detector_best.pt"
    shutil.copy2(
        WORK_DIR / "fish_detector" / "weights" / "best.pt",
        final_out
    )
    print(f"Final model saved: {final_out}")
    print()
    print("=" * 70)
    print("NEXT STEPS — Deploy the new detector")
    print("=" * 70)
    print()
    print("1. Download  /kaggle/working/yolo_fish_detector_best.pt")
    print()
    print("2. Replace in your backend:")
    print("     backend/models/yolo_detection_model.pt")
    print("   ← overwrite with  yolo_fish_detector_best.pt")
    print()
    print("3. No changes required in main.py — YOLO model path is already")
    print("   set to  YOLO_PATH = 'models/yolo_detection_model.pt'")
    print()
    print("4. Restart the backend and test with a fish image.")
    print()
    print("TIP: If some fish are still missed at inference, lower the YOLO")
    print("     confidence thresholds in main.py:")
    print("       Stage 1: conf=0.20 → 0.15")
    print("       Stage 3: conf=0.08 → 0.05")
    print("=" * 70)
"""))

# ── Assemble notebook ──────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.12",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [
                {
                    "sourceType": "datasetVersion",
                    "sourceId": 15420370,
                    "datasetId": 9864561,
                    "databundleVersionId": 16337934,
                }
            ],
            "dockerImageVersionId": 31329,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True,
        },
    },
    "cells": cells,
}

out = "yolo-fish-detector.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written {out}  ({len(cells)} cells)")
