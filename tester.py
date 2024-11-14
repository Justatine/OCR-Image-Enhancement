import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Label, Button, Tk, Text, END
import os
import warnings
import logging

# Suppress specific warnings from EasyOCR and PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("torch").setLevel(logging.ERROR)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize GUI
root = Tk()
root.title("Image to Text Extraction using OCR")

def show_image_with_matplotlib(img, title, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    if len(img.shape) == 2: 
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def contour_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape[:2]
    blur_ksize = (5, 5) if min(height, width) < 1000 else (7, 7)
    blurred = cv2.GaussianBlur(gray_image, blur_ksize, 0)
    median_intensity = np.median(blurred)
    lower_threshold = int(max(0, 0.66 * median_intensity))
    upper_threshold = int(min(255, 1.33 * median_intensity))
    edged = cv2.Canny(blurred, lower_threshold, upper_threshold)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    document_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        tolerance = 0.02 * perimeter if min(height, width) < 1000 else 0.015 * perimeter
        approx = cv2.approxPolyDP(contour, tolerance, True)
        if len(approx) == 4 and cv2.contourArea(approx) > (height * width * 0.05):
            document_contour = approx
            break
    if document_contour is not None:
        pts = document_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (top_left, top_right, bottom_right, bottom_left) = rect
        width_a = np.linalg.norm(bottom_right - bottom_left)
        width_b = np.linalg.norm(top_right - top_left)
        max_width = max(int(width_a), int(width_b))
        height_a = np.linalg.norm(top_right - bottom_right)
        height_b = np.linalg.norm(top_left - bottom_left)
        max_height = max(int(height_a), int(height_b))
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        cropped_image = cv2.warpPerspective(image, M, (max_width, max_height))
        return cropped_image
    else:
        return image

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel /= kernel.sum()
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image, kernel

def otsu_binarization(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0, 256))
    histogram = histogram / histogram.sum()
    cumulative_sum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))
    total_mean = cumulative_mean[-1]
    mask = (cumulative_sum * (1 - cumulative_sum)) > 0
    between_class_variance = np.zeros(256)
    between_class_variance[mask] = (total_mean * cumulative_sum[mask] - cumulative_mean[mask]) ** 2 / (cumulative_sum[mask] * (1 - cumulative_sum[mask]))
    optimal_threshold = np.argmax(between_class_variance)
    print(f"Optimal threshold: {optimal_threshold}")
    binary_image = (gray_image >= optimal_threshold) * 255
    return binary_image.astype(np.uint8)

def load_image():
    global extracted_text
    if not os.path.exists('Results'):
        os.makedirs('Results')
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        cropped_image = contour_detection(img)
        show_image_with_matplotlib(cropped_image, "Cropped Image with Document Contour Detected")
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        blurred_image, kernel = gaussian_blur(gray_image)
        binary_image = otsu_binarization(blurred_image)
        combined_path = os.path.join('Results', 'gaussian_otsu_binarization.png')
        cv2.imwrite(combined_path, binary_image)
        show_image_with_matplotlib(binary_image, "Preprocessed Image (Gaussian Blur + Otsu Binarization)")

        # Pass the preprocessed image to EasyOCR for OCR
        result = reader.readtext(binary_image, detail=0, paragraph=True)
        extracted_text = "\n".join(result)
        extracted_text_box.delete(1.0, END)
        extracted_text_box.insert(END, extracted_text)

def calculate_word_error_rate(extracted_text, reference_text):
    # Split both texts into words
    extracted_words = extracted_text.split()
    reference_words = reference_text.split()

    # Count the number of incorrect words
    incorrect_words = sum(1 for ew, rw in zip(extracted_words, reference_words) if ew != rw)
    incorrect_words += abs(len(extracted_words) - len(reference_words))

    # Total words in the reference text
    total_words = max(len(reference_words), 1)  # Avoid division by zero

    # WER Calculation
    wer = (incorrect_words / total_words) * 100
    accuracy = 100 - wer  # Accuracy based on WER
    return accuracy

def calculate_character_error_rate(extracted_text, reference_text):
    # Calculate the number of incorrect characters
    incorrect_chars = sum(1 for ec, rc in zip(extracted_text, reference_text) if ec != rc)
    incorrect_chars += abs(len(extracted_text) - len(reference_text))

    # Total characters in the reference text
    total_chars = max(len(reference_text), 1)  # Avoid division by zero

    # CER Calculation
    cer = (incorrect_chars / total_chars) * 100
    accuracy = 100 - cer  # Accuracy based on CER
    return accuracy

def calculate_accuracy():
    extracted_text = extracted_text_box.get(1.0, END).strip()
    reference_text = reference_text_box.get(1.0, END).strip()
    
    # Calculate WER-based accuracy
    wer_accuracy = calculate_word_error_rate(extracted_text, reference_text)
    accuracy_label.config(text=f"OCR Accuracy (WER): {wer_accuracy:.2f}%")

    # Calculate CER-based accuracy
    cer_accuracy = calculate_character_error_rate(extracted_text, reference_text)
    cer_label.config(text=f"OCR Accuracy (CER): {cer_accuracy:.2f}%")

root.geometry("800x600")

# GUI Elements
load_button = Button(root, text="Load Image", command=load_image)
load_button.pack()
extracted_text_label = Label(root, text="Extracted Text:")
extracted_text_label.pack()
extracted_text_box = Text(root, height=10, width=50)
extracted_text_box.pack()
reference_text_label = Label(root, text="Reference Text:")
reference_text_label.pack()
reference_text_box = Text(root, height=10, width=50)
reference_text_box.pack()
accuracy_button = Button(root, text="Calculate Accuracy", command=calculate_accuracy)
accuracy_button.pack()
accuracy_label = Label(root, text="OCR Accuracy (WER): N/A")
accuracy_label.pack()
cer_label = Label(root, text="OCR Accuracy (CER): N/A")
cer_label.pack()

# Run the GUI
root.mainloop()