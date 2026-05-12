import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os

# ================= CONFIG =================
YOLO_PATH = "models/yolo_detection_model.pt"
CLASSIFIER_PATH = "models/fish_classifier.keras"
IMG_SIZE = 224
CLASS_NAMES = ["Not Fresh", "Fresh"]

# ================= LOAD MODELS =================
print("Loading models...")
yolo_model = YOLO(YOLO_PATH)
classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
print("Models loaded successfully ✅")

# ================= DETECTION =================
def detect_best_fish(image):
    results = yolo_model(image)[0]

    if len(results.boxes) == 0:
        return None

    boxes = results.boxes

    # pick best box (confidence + size)
    best = max(
        boxes,
        key=lambda b: float(b.conf) * (float(b.xyxy[0][2]) - float(b.xyxy[0][0]))
    )

    x1, y1, x2, y2 = map(int, best.xyxy[0])
    return (x1, y1, x2, y2)


#==================BACKGROUND BLACK==================

def black_out_background(image, bbox):
    x1, y1, x2, y2 = bbox
    # Create a black image of the same size
    black_bg = np.zeros_like(image)
    # Copy only the fish into the black image
    black_bg[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return black_bg

# ================= BACKGROUND BLUR =================
def blur_background(image, bbox):
    x1, y1, x2, y2 = bbox

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    blurred = cv2.GaussianBlur(image, (21, 21), 0)

    result = np.where(mask[..., None] == 255, image, blurred)
    return result

# ================= PREPROCESS =================
def preprocess(crop):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ================= CLASSIFICATION =================
def classify(crop):
    img = preprocess(crop)
    preds = classifier.predict(img, verbose=0)[0]

    if len(preds) == 1:
        conf = preds[0]
        label = CLASS_NAMES[1] if conf > 0.5 else CLASS_NAMES[0]
    else:
        idx = np.argmax(preds)
        conf = preds[idx]
        label = CLASS_NAMES[idx]

    return label, float(conf), img

# ================= GRADCAM =================
def get_gradcam(model, img_array):
    last_conv = None

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    # with tf.GradientTape() as tape:
    #     conv_outputs, predictions = grad_model(img_array)
    #     class_idx = tf.argmax(predictions[0]).numpy()
    #     loss = predictions[:, class_idx]

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # 1. Grab the actual Tensor from the list
        # If grad_model returns a list, predictions is [Tensor]
        if isinstance(predictions, list):
            predictions = predictions[0] 
        class_idx = tf.argmax(predictions[0]).numpy()
        # 2. Now this slice will work because predictions is a Tensor
        loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap

def overlay_gradcam(crop, heatmap):
    heatmap = cv2.resize(heatmap, (crop.shape[1], crop.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(crop, 0.6, heatmap, 0.4, 0)

# ================= PIPELINE =================
def run_test(image_path):

    if not os.path.exists(image_path):
        print("Image not found ❌")
        return

    image = cv2.imread(image_path)

    bbox = detect_best_fish(image)

    if bbox is None:
        print("No fish detected ❌")
        return

    x1, y1, x2, y2 = bbox

    # blur background
    blurred = blur_background(image, bbox)

    # crop ROI
    crop = blurred[y1:y2, x1:x2]

    # classify
    label, conf, img_array = classify(crop)

    # gradcam
    heatmap = get_gradcam(classifier, img_array)
    cam = overlay_gradcam(crop, heatmap)

    # draw bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{label} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ================= DISPLAY =================
    cv2.imshow("Original + Detection", image)
    cv2.imshow("Cropped Fish", crop)
    cv2.imshow("GradCAM", cam)

    print(f"Prediction: {label} | Confidence: {conf:.3f}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ================= RUN =================
if __name__ == "__main__":
    test_image = "images/fresh/fresh2.jpeg"  # 🔥 change this
    run_test(test_image)