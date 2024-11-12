import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Image Upload and Display")

# Function to apply brightness and contrast adjustments
def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        shadow = max(brightness, 0)
        highlight = 255 if brightness > 0 else 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

# Function to open file dialog, adjust brightness/contrast, and display the image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        # Read the image with OpenCV
        img_cv = cv2.imread(file_path)
        
        # Resize the image (optional)
        img_cv = cv2.resize(img_cv, (400, 400), interpolation=cv2.INTER_AREA)

        # Apply brightness and contrast adjustment
        img_cv = apply_brightness_contrast(img_cv, brightness=0, contrast=64)  # Adjust brightness and contrast as needed
        
        # Convert the color format from BGR to RGB for PIL compatibility
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL format
        img_pil = Image.fromarray(img_rgb)
        
        # Convert to a format Tkinter can display
        img_tk = ImageTk.PhotoImage(img_pil)
        
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
