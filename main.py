from src.detect import run_detection
 

if __name__ == "__main__":
    run_detection(
        source="data/white.jpg",
        model_path="models/yolov8n.pt",
        conf=0.5,
        save_results=True,
        show_results=True,
        color_json="data/colors.json",
        debug=True  # Set to True to see detailed color matching
    )