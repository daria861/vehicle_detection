import cv2
from ultralytics import YOLO
import os
import numpy as np
import json

def load_color_refs(json_path):
    """
    Load color references from a JSON file with BGR values.
    Expects: { "color_name": [B, G, R], ... }
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        refs = [{"name": name, "bgr": np.array(bgr)} for name, bgr in data.items()]
        return refs
    except FileNotFoundError:
        raise FileNotFoundError(f"Color reference file not found: {json_path}")

def estimate_color_bgr(image, bbox, color_refs, debug=False):
    """
    Estimate the dominant color in the bounding box using BGR distance.
    Returns the closest color name from the palette.
    """
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
    # Crop central region to avoid windows/tires
    crop_ratio = 0.6
    width, height = x2 - x1, y2 - y1
    x1_crop = x1 + int(width * (1 - crop_ratio) / 2)
    x2_crop = x2 - int(width * (1 - crop_ratio) / 2)
    y1_crop = y1 + int(height * (1 - crop_ratio) / 2)
    y2_crop = y2 - int(height * (1 - crop_ratio) / 2)
    roi = image[y1_crop:y2_crop, x1_crop:x2_crop]
    if roi.size == 0:
        return "Unknown"
    
    # Mask: filter by brightness in HSV
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    value_channel = roi_hsv[:, :, 2]
    mask = (value_channel > 10) & (value_channel < 230)
    if mask.sum() == 0:
        if debug:
            print(f"No valid pixels in ROI at {x1_crop},{y1_crop} to {x2_crop},{y2_crop}")
        return "Unknown"
    pixels = roi[mask]
    if pixels.size == 0:
        return "Unknown"
    
    # Find dominant color via histogram mode (64 bins per channel, for speed)
    hist, edges = np.histogramdd(pixels, bins=(64, 64, 64), range=((0,256), (0,256), (0,256)))
    idx = np.unravel_index(hist.argmax(), hist.shape)
    dominant_bgr = np.array([idx[0]*4, idx[1]*4, idx[2]*4])
    if debug:
        print(f"Dominant BGR: {dominant_bgr}, Histogram idx: {idx}")

    # Find closest palette color by Euclidean distance in BGR
    min_dist = float("inf")
    closest_color = "Unknown"
    for ref in color_refs:
        dist = np.linalg.norm(dominant_bgr - ref["bgr"])
        if dist < min_dist:
            min_dist = dist
            closest_color = ref["name"]
        if debug:
            print(f"Palette color: {ref['name']}, BGR: {ref['bgr']}, Distance: {dist:.2f}")
    if debug:
        print(f"Chosen color: {closest_color}, Distance: {min_dist:.2f}")
    return closest_color

def resize_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def read_image(source):
    image = cv2.imread(source)
    if image is None:
        raise FileNotFoundError(f"Image not found: {source}")
    return image

def run_detection(
    source="data/yellow.jpg",
    model_path="models/yolov8n.pt",
    conf=0.5,
    save_results=True,
    show_results=True,
    color_json="data/colors.json",
    debug=False
):
    """
    Run YOLO detection and estimate vehicle colors.
    Annotate each car/vehicle with detected color.
    """
    color_refs = load_color_refs(color_json)
    model = YOLO(model_path)
    image = read_image(source)
    results = model(source, conf=conf)
    vehicle_params = []
    car_count = 0

    # For overlaying results
    img_with_boxes = image.copy()

    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} objects")
        names = result.names

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = names[cls_id]
            conf_score = float(box.conf[0].item())
            bbox = box.xyxy[0].cpu().numpy()
            if label in ["car", "truck", "bus", "motorcycle"]:
                if label == "car":
                    car_count += 1
                color = estimate_color_bgr(image, bbox, color_refs, debug=debug)
                params = {
                    "class": label,
                    "confidence": conf_score,
                    "bbox": bbox.astype(int).tolist(),
                    "color": color,
                    "make_model": "N/A",
                    "license_plate": "N/A"
                }
                vehicle_params.append(params)

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if label == "car":
                    text = f"Car {car_count}: {color}"
                else:
                    text = f"{label}: {color}"
                text_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)
                cv2.putText(
                    img_with_boxes,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
    # Console output
    for i, det in enumerate(vehicle_params):
        print(f"Vehicle {i+1}:")
        print(f"  Class: {det['class']}")
        print(f"  Confidence: {det['confidence']:.2f}")
        print(f"  Bounding Box: {det['bbox']}")
        print(f"  Color: {det['color']}")
        print(f"  Make/Model: {det['make_model']}")
        print(f"  License Plate: {det['license_plate']}")
        print("-" * 30)

    # Show/save annotated image
    if show_results:
        try:
            resized_img = resize_frame(img_with_boxes, scale=1.0)
            cv2.imshow("Vehicle Detection", resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Cannot display image: {e}")
    if save_results:
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"detected_{os.path.basename(source)}")
        cv2.imwrite(out_path, img_with_boxes)
        print(f"Saved output to: {out_path}")

if __name__ == "__main__":
    run_detection(
        source="data/vehicle_types.jpg",
        model_path="models/yolov8n.pt",
        conf=0.5,
        save_results=True,
        show_results=True,
        color_json="data/colors.json",
        debug=True  # Set to True to see detailed color matching
    )