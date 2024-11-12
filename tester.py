import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Image Upload and Display with Background Removal")

# Function to apply GrabCut (background removal)
def remove_background(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Convert the image to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create a mask for GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)

    # Create the background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define a rectangle to help GrabCut segment the foreground and background
    rect = (10, 10, img.shape[1]-10, img.shape[0]-10)

    # Apply GrabCut
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to create a binary mask where 0 is background, and 1 is foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    result = img * mask2[:, :, np.newaxis]

    # Convert the result to RGB format
    img_rgb_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return img_rgb_result

# Function to open file dialog and upload an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        # Apply background removal after the image is uploaded
        processed_image = remove_background(file_path)
        
        # Convert the processed image to PIL Image
        img_pil = Image.fromarray(processed_image)
        
        # Resize the image to fit in the label
        img_pil.thumbnail((400, 400))
        
        # Convert the image to a format that tkinter can display
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
