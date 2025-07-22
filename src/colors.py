import json
import cv2
import numpy as np

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
    s_channel = roi_hsv[:, :, 1]
    mask = (value_channel > 20) & (value_channel < 255) & (s_channel > 20) # Avoid too dark or too bright pixels
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
