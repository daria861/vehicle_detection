import cv2
from ultralytics import YOLO
import os
import numpy as np
import json

def load_color_refs(json_path):
    """Load color references from a JSON file and convert BGR to Lab."""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)  # {"color_name": [B, G, R]}
        refs = []
        for name, bgr in data.items():
            color_patch = np.uint8([[bgr]])
            lab = cv2.cvtColor(color_patch, cv2.COLOR_BGR2Lab)[0][0]
            refs.append({"name": name, "lab": lab})
        return refs
    except FileNotFoundError:
        raise FileNotFoundError(f"Color reference file not found: {json_path}")
    
def estimate_color_lab(image, bbox, color_refs):
    """
    Estimate the dominant color of the vehicle inside the bounding box
    using Lab color distance to reference colors.
    """
    # Extract bounding box coordinates and ensure they are within image bounds
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    
    # Crop ROI to central 60% of the bounding box for better body coverage
    width = x2 - x1
    height = y2 - y1
    crop_ratio = 0.6  # Use 60% to capture more body area
    x1_crop = x1 + int(width * (1 - crop_ratio) / 2)
    x2_crop = x2 - int(width * (1 - crop_ratio) / 2)
    y1_crop = y1 + int(height * (1 - crop_ratio) / 2)
    y2_crop = y2 - int(height * (1 - crop_ratio) / 2)
    
    # Extract cropped ROI
    roi = image[y1_crop:y2_crop, x1_crop:x2_crop]
    if roi.size == 0:
        return "Unknown"
    
    # Convert ROI to Lab color space
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
    
    # Filter out dark and bright pixels (likely tires and windows) using L channel
    l_channel = roi_lab[:, :, 0]  # L channel (lightness)
    min_lightness = 20  # Lower threshold for darker body shades
    max_lightness = 220  # Upper threshold for brighter body shades
    valid_pixels = (l_channel > min_lightness) & (l_channel < max_lightness)
    
    # Log number of valid pixels for debugging
    num_valid_pixels = np.sum(valid_pixels)
    if num_valid_pixels == 0:
        print(f"No valid pixels found in ROI at {x1_crop},{y1_crop} to {x2_crop},{y2_crop}")
        return "Unknown"
    else:
        print(f"Number of valid pixels: {num_valid_pixels}")
    
    # Extract valid Lab pixels
    valid_lab = roi_lab[valid_pixels]
    if valid_lab.size == 0:
        return "Unknown"
    
    # Use mode (most common color) in Lab space with reduced bins
    valid_lab_2d = valid_lab.reshape(-1, 3)
    hist, _ = np.histogramdd(valid_lab_2d, bins=(64, 64, 64), range=((0, 256), (0, 256), (0, 256)))
    dominant_lab = np.unravel_index(np.argmax(hist), (64, 64, 64))
    dominant_lab = np.array([dominant_lab[0] * 4, dominant_lab[1] * 4, dominant_lab[2] * 4])  # Scale back to 0-255
    
    # Find closest reference color using Euclidean distance in Lab space
    min_dist = float("inf")
    closest_color = "Unknown"
    for ref in color_refs:
        dist = np.linalg.norm(dominant_lab - ref["lab"])
        if dist < min_dist:
            min_dist = dist
            closest_color = ref["lab"]
    
    return closest_color


def resize_frame(frame, scale=0.75):
    """Resize the image frame to a smaller size for display."""
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def read_image(source):
    """Read an image from the specified path."""
    image = cv2.imread(source)
    if image is None:
        raise FileNotFoundError(f"Image not found: {source}")
    return image

def run_detection(
    source="data/porsche.jpg",
    model_path="models/yolov8n.pt",
    conf=0.5,
    save_results=True,
    show_results=True
):
    """
    Run YOLO detection on an image, estimate vehicle colors, and annotate car number and color on bounding boxes.
    """
    # Load color references
    color_refs = load_color_refs("data/colors.json")

    # Load YOLO model
    model = YOLO(model_path)

    # Read image
    image = read_image(source)
 
    # Run detection
    results = model(source, conf=conf)
    vehicle_params = []
    car_count = 0  # Counter for cars only

    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} objects")
        names = result.names  # Class index to label mapping
        img_with_boxes = image.copy()  # Copy image for custom annotations

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = names[cls_id]
            conf_score = float(box.conf[0].item())
            bbox = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)

            # Process only vehicle classes
            if label in ["car", "truck", "bus", "motorcycle"]:
                # Increment car_count only for "car" class
                if label == "car":
                    car_count += 1
                color = estimate_color_lab(image, bbox, color_refs)
                params = {
                    "class": label,
                    "confidence": conf_score,
                    "bbox": bbox.astype(int).tolist(),
                    "color": color,
                    "make_model": "N/A",  # Placeholder
                    "license_plate": "N/A"  # Placeholder
                }
                vehicle_params.append(params)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add text with car number (for cars only) and color
                if label == "car":
                    text = f"Car {car_count}: {color}"
                else:
                    text = f"{label}: {color}"
                text_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)  # Place above box, or below if near top
                cv2.putText(
                    img_with_boxes,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
    
    # Display detection results in console
    for i, det in enumerate(vehicle_params):
        print(f"Vehicle {i+1}:")
        print(f"  Class: {det['class']}")
        print(f"  Confidence: {det['confidence']:.2f}")
        print(f"  Bounding Box: {det['bbox']}")
        print(f"  Color: {det['color']}")
        print(f"  Make/Model: {det['make_model']}")
        print(f"  License Plate: {det['license_plate']}")
        print("-" * 30)

    # Show or save output image with custom annotations
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
        source="data/color_cars.jpg",
        model_path="models/yolov8n.pt",
        conf=0.5,
        save_results=True,
        show_results=True
    )