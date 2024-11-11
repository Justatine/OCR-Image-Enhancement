import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Image Upload, Skew Correction, and Display")

# Function to detect and correct skew in an image
def correct_skew(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path, 0)  # Load in grayscale
    # Apply thresholding to make the image binary
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find the coordinates of non-zero pixels (text pixels)
    coords = np.column_stack(np.where(binary_img > 0))
    
    # Calculate the angle of skew using the Hough transform method
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle range (OpenCV returns angles in [-90, 0), we need [0, 90])
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Get the image dimensions (height, width)
    (h, w) = img.shape
    
    # Calculate the rotation matrix
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the image to correct skew
    rotated_image = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Convert the corrected image to RGB format
    rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2RGB)
    
    return rotated_image_rgb

# Function to open file dialog and upload an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        # Correct the skew of the uploaded image
        corrected_image = correct_skew(file_path)
        
        # Convert the corrected image to a format that tkinter can display
        img_pil = Image.fromarray(corrected_image)
        
        # Check if the image is horizontally oriented and rotate to vertical if necessary
        if img_pil.width > img_pil.height:
            img_pil = img_pil.rotate(90, expand=True)
        
        # Resize the image by adjusting the width while maintaining the aspect ratio
        aspect_ratio = img_pil.height / img_pil.width
        new_width = 1080
        new_height = int(aspect_ratio * new_width)
        img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # If the height exceeds 1080, we need to resize the height to 1080 while adjusting the width proportionally
        if new_height > 1080:
            aspect_ratio = img_pil.width / img_pil.height
            new_height = 1080
            new_width = int(aspect_ratio * new_height)
            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Add padding if the image is smaller than 1080x1080 (to prevent cropping)
        width, height = img_pil.size
        new_image = Image.new("RGB", (1080, 1080), (255, 255, 255))  # White background
        new_image.paste(img_pil, ((1080 - width) // 2, (1080 - height) // 2))
        
        # Convert the final image to a format suitable for tkinter display
        img_tk = ImageTk.PhotoImage(new_image)
        
        # Display the corrected and resized image
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
