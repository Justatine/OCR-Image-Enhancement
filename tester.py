import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import pytesseract as tess
import cv2
import numpy as np
import os

# Configure Tesseract path
tess.pytesseract.tesseract_cmd = r"C:\Users\Jan Kenneth\Downloads\Tesseract-OCR-20241111T125308Z-001\Tesseract-OCR\tesseract.exe"

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

def setExtractedWord(word):
    global extracted_word
    extracted_word = word

def getExtractedWord():
    return extracted_word

def setNumberofRefWords(num):
    global refwords
    refwords = num

def getNumberofRefWords():
    return refwords

class ImageToTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image To Text Converter")
        self.root.geometry("600x600")
        
        # Button to select an image
        self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)
        
        # Label to display the selected image
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        
        # Text box to display extracted text
        self.text_display = tk.Text(root, height=5, wrap=tk.WORD)
        self.text_display.pack(pady=10)
        
        # Initialize variables for cropping functionality
        self.start_x = self.start_y = 0
        self.rect = None
        self.cropped_image = None

        # Create a textbox for input (reference words)
        self.text_box = tk.Text(root, wrap="word", width=50, height=10)  # Fixed this line
        self.text_box.pack(pady=10)

        # Create a button to compare words
        compare_button = tk.Button(root, text="Compare Words", command=self.compare_words)
        compare_button.pack(pady=5)

        # Label to display the result
        self.result_label = tk.Label(root, text="Incorrect Words: 0")
        self.result_label.pack(pady=10)

        self.accuracy_label = tk.Label(root, text="Accuracy: 0")
        self.accuracy_label.pack(pady=12)

    # Function to count incorrect words (Make this static)
    @staticmethod
    def count_incorrect_words(extracted_words, reference_words):
        # Ensure both arrays are of the same length for comparison
        min_length = min(len(extracted_words), len(reference_words))
        incorrect_count = 0
        
        # Compare each word in the two arrays
        for i in range(min_length):
            if extracted_words[i] != reference_words[i]:
                incorrect_count += 1
        
        # If the arrays have different lengths, count all extra words in the longer array as incorrect
        incorrect_count += abs(len(extracted_words) - len(reference_words))
        
        return incorrect_count

    # Function to handle the word count and comparison
    def compare_words(self):
        # Extract the text input from the textbox (treated as reference)
        reference_text = self.text_box.get("1.0", tk.END).strip()  # Get reference text
        reference_words = reference_text.split()  # Split the reference text into words
        
        # print(f"Number of words: {len(reference_words)}")
        setNumberofRefWords(len(reference_words))

        # Initialize an empty list to store the words
        ex_words = []

        # Get the extracted text
        extracted_words = getExtractedWord()

        # Check if there's a valid extracted string (not empty)
        if extracted_words:
            # Clean the extracted words by removing any newline characters
            cleaned_words = extracted_words.replace('\n', ' ')  # Replace newlines with spaces

            # Split the cleaned text into individual words based on spaces
            words_list = cleaned_words.split()

            # Add each word to the ex_words list
            ex_words.extend(words_list)

            # Output the result
            print(ex_words)

        # Count incorrect words by comparing the extracted and reference words
        incorrect_count = self.count_incorrect_words(ex_words, reference_words)
        total =  getNumberofRefWords();
        accuracyValue1 = (incorrect_count / total) * 100;
        accuracyValue2 = 100 - accuracyValue1
        # Display the result
        self.result_label.config(text=f"Incorrect Words: {incorrect_count}")  # Show number of incorrect words
        self.accuracy_label.config(text=f"Accuracy: {accuracyValue2:.2f}%")  # Show accuracy with 2 decimal places

    def select_image(self):
        # Open file dialog to select an image file
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if not file_path:
            return

        # Load the image using PIL
        self.original_image = Image.open(file_path)
        self.image_thumbnail = self.original_image.copy()
        self.image_thumbnail.thumbnail((400, 400))  # Resize for display
        
        # Display the selected image
        img_tk = ImageTk.PhotoImage(self.image_thumbnail)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Save reference
        
        # Open a new window for region selection
        self.open_crop_window()

    def open_crop_window(self):
        # Create a new window for cropping
        self.crop_window = tk.Toplevel(self.root)
        self.crop_window.title("Select Area to Convert to Text")
        
        # Canvas to display the image and select crop area
        self.canvas = tk.Canvas(self.crop_window, width=self.image_thumbnail.width, height=self.image_thumbnail.height)
        self.canvas.pack()

        # Display image on the canvas
        self.image_tk = ImageTk.PhotoImage(self.image_thumbnail)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        
        # Bind mouse events for cropping
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        # Button to read the selected area
        read_button = tk.Button(self.crop_window, text="Read", command=self.read_selected_area)
        read_button.pack(pady=10)
    
    def on_button_press(self, event):
        # Record the start point of the rectangle
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)

    def on_mouse_drag(self, event):
        # Update the rectangle as the mouse is dragged
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red")

    def on_button_release(self, event):
        # Record the end point of the rectangle and create the crop box
        self.end_x, self.end_y = event.x, event.y
        self.crop_box = (
            int(self.start_x * (self.original_image.width / self.image_thumbnail.width)),
            int(self.start_y * (self.original_image.height / self.image_thumbnail.height)),
            int(self.end_x * (self.original_image.width / self.image_thumbnail.width)),
            int(self.end_y * (self.original_image.height / self.image_thumbnail.height)),
        )

    def remove_background(self, image_path):
        # Read the image using OpenCV
        
        img = cv2.imread(image_path)
        
           # Convert the image to grayscale for contour detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

          # Apply binary thresholding to the grayscale image
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

          # Find contours in the thresholded image
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

          # Create a mask for the image, initializing it to zero (black)
        mask = np.zeros(img.shape, dtype=np.uint8)

          # Fill the contours with white color (255)
        cv2.fillPoly(mask, cnts, (255, 255, 255))

          # Invert the mask to get the background (black)
        mask = 255 - mask

          # Use bitwise OR to combine the mask with the original image
        result = cv2.bitwise_or(img, mask)

          # Optionally, convert the result to RGB format (if needed for PIL or display)
        img_rgb_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return img_rgb_result

    
    def read_selected_area(self):
        # Crop the selected area and run OCR with preprocessing
        try:
            # Crop the selected area from the original image
            cropped_image = self.original_image.crop(self.crop_box)
            
            # Convert PIL image to OpenCV format
            cropped_cv_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
            
            contrast = apply_brightness_contrast(cropped_cv_image, brightness=0, contrast=64)
            
            # Apply Otsu's binarization
            gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
            _, binarized_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,8))
            morph_img = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Save binarized image
            os.makedirs("otsu_binarization", exist_ok=True)
            binarized_path = os.path.join("otsu_binarization", "binarized_image.png")
            cv2.imwrite(binarized_path, morph_img)

            # Skew correction
            coords = np.column_stack(np.where(morph_img > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = binarized_image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            corrected_image = cv2.warpAffine(binarized_image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Save skew-corrected image
            os.makedirs("skew_correction", exist_ok=True)
            corrected_path = os.path.join("skew_correction", "corrected_image.png")
            cv2.imwrite(corrected_path, corrected_image)

            # Apply contour preprocessing
            contour_image = self.remove_background(corrected_path)
            
            # Save the contoured image
            os.makedirs("contour", exist_ok=True)
            contour_path = os.path.join("contour", "contoured_image.png")
            cv2.imwrite(contour_path, cv2.cvtColor(contour_image, cv2.COLOR_RGB2BGR))

            # Run OCR on the final processed image
            extracted_text = tess.image_to_string(contour_image)
            
            setExtractedWord(extracted_text)

            # Display extracted text in the main window
            self.text_display.delete("1.0", tk.END)  # Clear previous text
            self.text_display.insert(tk.END, extracted_text)
            
            # Close the crop window
            self.crop_window.destroy()
            
            messagebox.showinfo("Success", f"Preprocessed images saved at:\n- Binarized: {binarized_path}\n- Skew-Corrected: {corrected_path}\n- Contoured: {contour_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not process selected area:\n{e}")

# Run the app
root = tk.Tk()
app = ImageToTextApp(root)
root.mainloop()