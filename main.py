import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import os

from src.detect import run_detection  # Import your detection function

class CarRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Car Recognition App")
        self.master.iconbitmap("src/car.ico")  # Set your icon file here
        self.master.geometry("800x600")
        self.canvas = tk.Canvas(master, width=600, height=480, bg="gray")
        self.canvas.pack()
        self.upload_btn = ttk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()
        self.detect_btn = ttk.Button(master, text="Run Detection", command=self.run_detection)
        self.detect_btn.pack()
        self.image_path = None
        self.detected_image_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((800, 600))
            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor='nw', image=self.img_tk)
            self.canvas.image = self.img_tk

    def run_detection(self):
        if not self.image_path:
            print("No image uploaded!")
            return
        # Run your detection pipeline
        run_detection(
            source=self.image_path,
            model_path="models/yolov8n.pt",
            conf=0.5,
            save_results=True,
            show_results=False,  # Don't use cv2.imshow in GUI mode
            color_json="data/colors.json",
            debug=False
        )
        # Load and display the result image (with boxes and labels)
        out_path = os.path.join("results", f"detected_{os.path.basename(self.image_path)}")
        if os.path.exists(out_path):
            img = Image.open(out_path)
            img.thumbnail((800, 600))
            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor='nw', image=self.img_tk)
            self.canvas.image = self.img_tk
        else:
            print("Detection results not found!")

if __name__ == "__main__":
    root = tk.Tk()
    app = CarRecognitionApp(root)
    root.mainloop()