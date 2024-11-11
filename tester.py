import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Image Upload and Display")

# Function for Otsu Binarization
def otsu_binarization(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, otsu_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return otsu_img

# Function to open file dialog and upload an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        # Open the image using OpenCV
        img = cv2.imread(file_path)
        
        # Apply Otsu binarization
        otsu_img = otsu_binarization(img)
        
        # Convert the image to PIL format for display in tkinter
        otsu_img_pil = Image.fromarray(otsu_img)
        
        # Resize the image to fit in the label
        otsu_img_pil.thumbnail((400, 400))
        
        # Convert the image to a format that tkinter can display
        img_tk = ImageTk.PhotoImage(otsu_img_pil)
        
        # Display the image
        label.config(image=img_tk)
        label.image = img_tk  # Keep a reference to avoid garbage collection

# Create an upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

# Create a label to display the uploaded image
label = tk.Label(root)
label.pack(pady=10)

# Run the application
root.mainloop()
