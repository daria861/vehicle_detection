import cv2
from ultralytics import YOLO
import os
from src.colors import load_color_refs, estimate_color_bgr
from src.classify import load_classifier, predict_make_model, preprocess_crop
import matplotlib.pyplot as plt

# EasyOCR for OCR functionality:
import easyocr

# Load classifier and class names once (at module level or before calling run_detection)
classifier_model = load_classifier("models/car_make_model_efficientnet.pth", device='cpu')

with open("models/names.csv", "r") as f:
    class_names = [line.strip() for line in f]

# Load YOLO model for license plate detection
lp_model = YOLO("models/license_plate_detector.pt")  # Path to license plate weights

# OCR reader
ocr_reader = easyocr.Reader(['en'])

def resize_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def read_image(source):
    image = cv2.imread(source)
    if image is None:
        raise FileNotFoundError(f"Image not found: {source}")
    return image

def detect_license_plate(crop):
    results = lp_model(crop)
    plate_text = "N/A"
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            lp_crop = crop[y1:y2, x1:x2]
            ocr_result = ocr_reader.readtext(lp_crop)
            if ocr_result:
                plate_text = ocr_result[0][1]  # Most confident result
    return plate_text

def run_detection(
    source="data/yellow.jpg",
    model_path="models/yolov8n.pt",
    conf=0.5,
    save_results=True,
    show_results=True,
    color_json="data/colors.json",
    debug=False
):
    color_refs = load_color_refs(color_json)
    model = YOLO(model_path)
    image = read_image(source)
    results = model(source, conf=conf)
    vehicle_params = []
    car_count = 0
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
                x1, y1, x2, y2 = map(int, bbox)
                vehicle_crop = image[y1:y2, x1:x2]
                make_model = predict_make_model(classifier_model, vehicle_crop, class_names)
                # License plate detection and OCR
                license_plate = detect_license_plate(vehicle_crop)
                params = {
                    "class": label,
                    "confidence": conf_score,
                    "bbox": bbox.astype(int).tolist(),
                    "color": color,
                    "make_model": make_model,
                    "license_plate": license_plate
                }
                vehicle_params.append(params)

                # Draw bounding box and label
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if label == "car":
                    text = f"Car {car_count}: {color} {make_model} {license_plate}"
                else:
                    text = f"{label}: {color} {make_model} {license_plate}"
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