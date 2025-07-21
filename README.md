# Vehicle Detection Project

This is a template for a vehicle detection system using YOLO (v5/v8), OpenCV, and Python.

## Structure

- `data/` — Input images and videos for testing
- `models/` — Downloaded or trained YOLO model weights
- `src/` — Core detection and utility code
- `gui/` — GUI code (Tkinter or PyQt5)
- `main.py` — Run detection or launch GUI

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv vehicle-detection-env
   ```

2. Activate the environment:
   - Windows: `vehicle-detection-env\Scripts\activate`
   - Linux/Mac: `source vehicle-detection-env/bin/activate`

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your test images/videos in `data/`.

5. Run:
   ```bash
   python main.py
   ```