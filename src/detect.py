import cv2
from ultralytics import YOLO
import os
import numpy as np
import pandas as pd


def load_color_refs(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["name", "label", "hex", "r", "g", "b"]
    df["bgr"] = df[["b", "g", "r"]].values.tolist()
    df["bgr"] = df["bgr"].apply(np.array)
    return df[["name", "hex", "bgr"]]

def estimate_color_vectorized(image, bbox, color_refs):
    x1, y1, x2, y2 = map(int, bbox)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return "Unknown"

    avg_color = crop.mean(axis=(0, 1))  # (B, G, R)
    
    ref_colors = np.stack(color_refs["bgr"].values)
    dists = np.linalg.norm(ref_colors - avg_color, axis=1)
    min_idx = np.argmin(dists)
    #return color_refs.iloc[min_idx]["hex"]
    return color_refs.iloc[min_idx]["hex"]

def resize_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def read_image(source):
    pass

def read_video(source):
    pass

def run_detection(
    source="data/porsche.jpg",
    model_path="models/yolov8n.pt",
    conf=0.5,
    save_results=True,
    show_results=True
):
    """
    Run YOLO detection on an image and estimate vehicle colors using CSV-based reference.
    """
    # Load color references
    color_refs = load_color_refs("data/colors.csv")

    # Load YOLO model
    model = YOLO(model_path)

    # Read image for color estimation
    image = cv2.imread(source)
    if image is None:
        raise FileNotFoundError(f"Image not found: {source}")

    # Run detection
    results = model(source, conf=conf)
    vehicle_params = []

    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} total boxes")
        names = result.names  # class index to label mapping

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = names[cls_id]
            conf_score = float(box.conf[0].item())
            bbox = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
            
            #if conf_score < 0.6:
                #continue

            # Only process vehicle classes
            if label in ["car", "truck", "bus", "motorcycle"]:
                color = estimate_color_vectorized(image, bbox, color_refs)
                params = {
                    "class": label,
                    "confidence": conf_score,
                    "bbox": bbox.astype(int).tolist(),
                    "color": color,
                    "make_model": "N/A",      # Placeholder
                    "license_plate": "N/A"    # Placeholder
                }
                vehicle_params.append(params)
                
                
                
    # Display results
    for i, det in enumerate(vehicle_params):
        print(f"Vehicle {i+1}:")
        print(f"  Class: {det['class']}")
        print(f"  Confidence: {det['confidence']:.2f}")
        print(f"  Bounding Box: {det['bbox']}")
        print(f"  Color: {det['color']}")
        print(f"  Make/Model: {det['make_model']}")
        print(f"  License Plate: {det['license_plate']}")
        print("-" * 30)

    # Show/save output
    for result in results:
        img_with_boxes = result.plot()
        if show_results:
            try:
                resized_img = resize_frame(img_with_boxes, scale=0.25)
                cv2.imshow("Vehicle Detection", resized_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("Unable to display image (headless environment?)")
        if save_results:
            out_dir = "results"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"detected_{os.path.basename(source)}")
            cv2.imwrite(out_path, img_with_boxes)
            print(f"Saved: {out_path}")
