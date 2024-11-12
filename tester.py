import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Image Upload and Display")

# Function to remove lines from the image using morphological operations
def remove_lines_from_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to get a binary image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Define a kernel for morphological transformations
    kernel = np.ones((3, 3), np.uint8)
    
    # Perform morphological closing (to remove small holes in the foreground)
    morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Perform morphological opening (to remove lines and small noise)
    morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel)
    
    # Invert the image back to get the result
    final_result = cv2.bitwise_not(morph_img)
    
    # Return the processed image
    return final_result

# Function to open file dialog and upload an image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
    if file_path:
        # Open the image using OpenCV
        img = cv2.imread(file_path)
        
        # Apply the remove_lines_from_image function
        img_no_lines = remove_lines_from_image(img)
        
        # Convert the processed image to PIL format for display in tkinter
        img_no_lines_pil = Image.fromarray(cv2.cvtColor(img_no_lines, cv2.COLOR_BGR2RGB))
        
        # Resize the image to fit in the label
        img_no_lines_pil.thumbnail((400, 400))
        
        # Convert the image to a format that tkinter can display
        img_tk = ImageTk.PhotoImage(img_no_lines_pil)
        
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
