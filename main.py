import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

from src.detect import run_detection  # Detection function

# Define fonts
FONT_NORMAL = ("Segoe UI", 13)
FONT_BOLD = ("Segoe UI", 13, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_HELP = ("Segoe UI", 11)

class DetailsLabel:
    def __init__(self, parent, title, font=FONT_NORMAL, fg="#fff", bg="#23272a", width=30):
        self.label = tk.Label(
            parent,
            text=f"{title}: -",
            font=font,
            fg=fg,
            bg=bg,
            width=width,
            anchor='w'
        )
        self.title = title
        self.label.pack(anchor='w', pady=7)
    
    def set(self, value):
        self.label.config(text=f"{self.title}: {value if value is not None else '-'}")
    
    def reset(self):
        self.set("-")

class CarRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Car Recognition App")
        self.master.geometry("1250x700")
        self.master.configure(bg="#23272a")
        try:
            self.master.iconbitmap("src/car.ico")
        except:
            pass

        # Main layout
        self.container = tk.Frame(master, bg="#23272a")
        self.container.pack(fill='both', expand=True, padx=25, pady=25)

        # Left - image canvas
        self.canvas = tk.Canvas(self.container, width=800, height=600, bg="#2c2f33", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=10, sticky="nsw", padx=(0,40))

        # Right - sidebar
        self.sidebar = tk.Frame(self.container, bg="#23272a", width=350)
        self.sidebar.grid(row=0, column=1, sticky="ne")
        self.sidebar.grid_propagate(False)
        
        # Button frame (top of sidebar)
        self.button_frame = tk.Frame(self.sidebar, bg="#23272a", width=350)
        self.button_frame.pack(fill='x', pady=(0, 15))

        # Buttons
        self.upload_btn = tk.Button(
            self.button_frame, text="Upload Image", font=FONT_BOLD,
            command=self.upload_image, bg="#7289da", fg="white", relief="flat", activebackground="#99aab5", width=30
        )
        self.upload_btn.pack(fill='x', pady=(0,15), ipadx=10, ipady=8)

        self.detect_btn = tk.Button(
            self.button_frame, text="Run Detection", font=FONT_BOLD,
            command=self.run_detection, bg="#43b581", fg="white", relief="flat", activebackground="#99aab5", width=30
        )
        self.detect_btn.pack(fill='x', pady=(0,15), ipadx=10, ipady=8)

        self.nav_frame = tk.Frame(self.sidebar, bg="#23272a", width=350)
        self.nav_frame.pack(fill='x', pady=(0,10))
        self.prev_btn = tk.Button(
            self.nav_frame, text="← Previous", font=FONT_NORMAL,
            command=self.show_prev_car, bg="#99aab5", fg="#23272a", relief="flat"
        )
        self.prev_btn.pack(side='left', padx=5, pady=5, expand=True, fill='x')
        self.next_btn = tk.Button(
            self.nav_frame, text="Next →", font=FONT_NORMAL,
            command=self.show_next_car, bg="#99aab5", fg="#23272a", relief="flat"
        )
        self.next_btn.pack(side='right', padx=5, pady=5, expand=True, fill='x')

        # Details title
        self.details_title = tk.Label(
            self.sidebar, text="Car Details", font=FONT_TITLE,
            fg="#fff", bg="#23272a"
        )
        self.details_title.pack(anchor='w', pady=(10,5))

        # Details labels using DetailsLabel class
        self.detail_color = DetailsLabel(self.sidebar, "Color", font=FONT_NORMAL)
        self.detail_model = DetailsLabel(self.sidebar, "Model", font=FONT_NORMAL)
        self.detail_plate = DetailsLabel(self.sidebar, "Plate", font=FONT_NORMAL)
        self.detail_count = DetailsLabel(self.sidebar, "Cars detected", font=FONT_NORMAL)
        self.detail_number = DetailsLabel(self.sidebar, "Showing car", font=FONT_NORMAL)

        # Help/instructions
        self.help_label = tk.Label(
            self.sidebar,
            text="• Upload an image of a car\n• Press 'Run Detection'\n• Use arrows to browse detected cars",
            font=FONT_HELP, fg="#b9bbbe", bg="#23272a", justify="left"
        )
        self.help_label.pack(anchor='w', pady=(30,0))

        # State
        self.image_path = None
        self.detected_image_path = None
        self.results = None
        self.current_car_index = 0

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((800, 600))
            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor='nw', image=self.img_tk)
            self.canvas.image = self.img_tk

            # Reset details
            self.results = None
            self.current_car_index = 0
            self.detail_color.reset()
            self.detail_model.reset()
            self.detail_plate.reset()
            self.detail_count.reset()
            self.detail_number.reset()

    def run_detection(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        # Run detection and store results
        results = run_detection(
            source=self.image_path,
            model_path="models/yolov8n.pt",
            conf=0.5,
            save_results=True,
            show_results=False,
            color_json="data/colors.json",
            debug=False
        )
        # results = {"vehicles": [{...}, {...}], "car_count": n}
        self.results = results
        self.current_car_index = 0

        # Show main image with all boxes
        out_path = os.path.join("results", f"detected_{os.path.basename(self.image_path)}")
        if os.path.exists(out_path):
            img = Image.open(out_path)
            img.thumbnail((800, 600))
            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor='nw', image=self.img_tk)
            self.canvas.image = self.img_tk
        else:
            messagebox.showerror("Detection Error", "Detection results not found!")

        self.update_car_details()

    def update_car_details(self):
        # Show details for the selected car
        if not self.results or "vehicles" not in self.results or not self.results["vehicles"]:
            self.detail_color.reset()
            self.detail_model.reset()
            self.detail_plate.reset()
            self.detail_count.reset()
            self.detail_number.reset()
            return

        n = self.results["car_count"]
        self.detail_count.set(str(n))

        car = self.results["vehicles"][self.current_car_index]
        self.detail_color.set(car.get('color', '-'))
        self.detail_model.set(car.get('model', '-'))
        self.detail_plate.set(car.get('plate', '-'))
        self.detail_number.set(f"{self.current_car_index + 1} of {n}")

    def show_next_car(self):
        if self.results and "vehicles" in self.results and len(self.results["vehicles"]) > 1:
            self.current_car_index = (self.current_car_index + 1) % len(self.results["vehicles"])
            self.update_car_details()

    def show_prev_car(self):
        if self.results and "vehicles" in self.results and len(self.results["vehicles"]) > 1:
            self.current_car_index = (self.current_car_index - 1) % len(self.results["vehicles"])
            self.update_car_details()

if __name__ == "__main__":
    root = tk.Tk()
    app = CarRecognitionApp(root)
    root.mainloop()