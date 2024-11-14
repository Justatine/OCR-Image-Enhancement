import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import pytesseract as tess
import cv2
import numpy as np
import os
from imutils.perspective import four_point_transform
import imutils

# Configure Tesseract path
tess.pytesseract.tesseract_cmd = r"C:\Users\Jan Kenneth\Downloads\Tesseract-OCR-20241111T125308Z-001\Tesseract-OCR\tesseract.exe"

def apply_brightness_contrast(img, brightness=0, contrast=0):
    if brightness != 0:
        shadow = max(brightness, 0)
        highlight = 255 if brightness > 0 else 255 + brightness
        alpha_b = (highlight - shadow) / 255
        buf = cv2.addWeighted(img, alpha_b, img, 0, shadow)
    else:
        buf = img.copy()

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

def save_img(image, output_folder, output_filename):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Check if the file exists and append a number if it does
    base_filename, file_extension = os.path.splitext(output_filename)
    output_image_path = os.path.join(output_folder, output_filename)
    
    # Add a number suffix to the filename if the file exists
    counter = 1
    while os.path.exists(output_image_path):
        # Construct a new filename with a counter
        output_image_path = os.path.join(output_folder, f"{base_filename}_{counter}{file_extension}")
        counter += 1
    
    # Save the image to the output path
    cv2.imwrite(output_image_path, image)
    print(f"Image saved to: {output_image_path}")

class ImageToTextApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image To Text Converter")
        self.root.geometry("600x600")
        
        self.initUI()

    def initUI(self):
        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)
        
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        self.text_display = tk.Text(self.root, height=5, wrap=tk.WORD)
        self.text_display.pack(pady=10)
        
        self.text_box = tk.Text(self.root, wrap="word", width=50, height=10)
        self.text_box.pack(pady=10)
        
        compare_button = tk.Button(self.root, text="Compare Words", command=self.compare_words)
        compare_button.pack(pady=5)
        
        self.result_label = tk.Label(self.root, text="Incorrect Words: 0")
        self.result_label.pack(pady=10)
        
        self.accuracy_label = tk.Label(self.root, text="Accuracy: 0")
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
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if not file_path:
            return

        self.original_image = Image.open(file_path)
        self.image_thumbnail = self.original_image.copy()
        self.image_thumbnail.thumbnail((400, 400))
        
        img_tk = ImageTk.PhotoImage(self.image_thumbnail)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
        
        self.open_crop_window()

    def open_crop_window(self):
        self.crop_window = tk.Toplevel(self.root)
        self.crop_window.title("Select Area to Convert to Text")
        
        self.canvas = tk.Canvas(self.crop_window, width=self.image_thumbnail.width, height=self.image_thumbnail.height)
        self.canvas.pack()
        
        self.image_tk = ImageTk.PhotoImage(self.image_thumbnail)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        read_button = tk.Button(self.crop_window, text="Read", command=self.read_selected_area)
        read_button.pack(pady=10)
    
    def on_button_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if hasattr(self, 'rect') and self.rect:
            self.canvas.delete(self.rect)

    def on_mouse_drag(self, event):
        if hasattr(self, 'rect') and self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red")

    def on_button_release(self, event):
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
        try:
            cropped_image = self.original_image.crop(self.crop_box)
            cropped_cv_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

            # CONTRAST
            # ==================================================
            contrast_image = apply_brightness_contrast(cropped_cv_image, contrast=64)
            save_img(contrast_image, 'contrast_adjustment', 'contrast_image.png')

            # OTSU
            image = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2GRAY)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            bins_num = 256
            hist, bin_edges = np.histogram(image, bins=bins_num)
            hist = np.divide(hist.ravel(), hist.max())
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            mean1 = np.cumsum(hist * bin_mids) / weight1
            mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
            inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            index_of_max_val = np.argmax(inter_class_variance)
            threshold = bin_mids[:-1][index_of_max_val]
            print("Otsu's algorithm implementation thresholding result: ", threshold)
            _, binarized_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            save_img(binarized_image, 'otsu_binarization', 'otsu_binarization.png')

            # CONTOUR
            # ==================================================
            corrected_image = self.apply_contour(binarized_image)
            save_img(corrected_image, 'contour', 'contour.png')

            extracted_text = tess.image_to_string(corrected_image)
            self.text_display.delete("1.0", tk.END)
            self.text_display.insert(tk.END, extracted_text)

            self.crop_window.destroy()
            messagebox.showinfo("Success", "Text extraction completed.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not process selected area:\n{e}")

    def apply_contour(self, image):
        cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return four_point_transform(image, approx.reshape(4, 2))
        return image

    def compare_words(self):
        reference_text = self.text_box.get("1.0", tk.END).strip()
        reference_words = reference_text.split()
        
        extracted_text = self.text_display.get("1.0", tk.END).strip()
        extracted_words = extracted_text.split()
        
        incorrect_count = sum(1 for r, e in zip(reference_words, extracted_words) if r != e)
        incorrect_count += abs(len(reference_words) - len(extracted_words))
        
        accuracy = max(0, (len(reference_words) - incorrect_count) / len(reference_words) * 100)
        
        self.result_label.config(text=f"Incorrect Words: {incorrect_count}")
        self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")
# Run the app
root = tk.Tk()
app = ImageToTextApp(root)
root.mainloop()