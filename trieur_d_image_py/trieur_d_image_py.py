import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import shutil


class ImageViewer(tk.Tk):
    def __init__(self, folder_path, output_folder):
        super().__init__()
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.image_files = [os.path.join(folder_path, f) for f in self.image_files]
        self.current_index = 0

        self.title("Image Viewer")
        self.geometry("800x600")

        self.image_label = tk.Label(self)
        self.image_label.pack(fill="both", expand=True)

        self.bind("<Right>", self.next_image)
        self.bind("<Left>", self.prev_image)
        self.bind("<s>", self.save_image)

        self.update_image()

    def update_image(self):
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]
        img = Image.open(image_path)
        img = img.resize((800, 600), Image.Resampling.LANCZOS)  # Update here
        img = ImageTk.PhotoImage(img)

        self.image_label.config(image=img)
        self.image_label.image = img  # keep a reference!

    def next_image(self, event):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_image()

    def prev_image(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()

    def save_image(self, event):
        current_image_path = self.image_files[self.current_index]
        shutil.copy(current_image_path, self.output_folder)
        print(f"Image saved to {self.output_folder}")


if __name__ == "__main__":
    folder_path = filedialog.askdirectory(title="Select Image Folder")
    output_folder = filedialog.askdirectory(title="Select Output Folder")
    if folder_path and output_folder:
        app = ImageViewer(folder_path, output_folder)
        app.mainloop()
