import numpy as np
import cv2
from tkinter import filedialog, Label, Button, Tk, Text, END
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from pytesseract import pytesseract, Output
import os

pytesseract.tesseract_cmd = r"C:\Users\Jan Kenneth\Downloads\Tesseract-OCR-20241111T125308Z-001\Tesseract-OCR\tesseract.exe"

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

def contour_detection(gray_image):
    # Process grayscale image for contour detection
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    document_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
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
        width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
        height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        cropped_image = cv2.warpPerspective(gray_image, M, (max_width, max_height))
        return cropped_image
    else:
        return gray_image

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel /= kernel.sum()
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

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


def line_and_word_segmentation(image):
    blurred_image = gaussian_blur(image)
    binary_image = otsu_binarization(blurred_image)
    
    line_box_image = image.copy()
    word_box_image = image.copy()
    
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 5))
    line_img = cv2.dilate(binary_image, kernel_line, iterations=1)

    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        line_boxes.append((x, y, w, h))
        cv2.rectangle(line_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in line_boxes:
        line_crop = binary_image[y:y+h, x:x+w]
        kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        word_img = cv2.dilate(line_crop, kernel_word, iterations=1)

        word_contours, _ = cv2.findContours(word_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for wc in word_contours:
            wx, wy, ww, wh = cv2.boundingRect(wc)
            cv2.rectangle(word_box_image, (x + wx, y + wy), (x + wx + ww, y + wy + wh), (255, 0, 0), 2)
    return word_box_image

def load_image():
    global extracted_text
    if not os.path.exists('Results'):
        os.makedirs('Results')

    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)

        # Show original image
        show_image_with_matplotlib(img, "Original Image")

        # Allow user to select a region of interest (ROI) for cropping
        r = cv2.selectROI("Select Region to Crop", img, fromCenter=False, showCrosshair=True)
        if r != (0, 0, 0, 0):
            # Get coordinates of the selected ROI without resizing or zooming in
            x, y, w, h = map(int, r)

            # Create a mask to focus only on the selected region
            mask = np.zeros_like(img)
            mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]
            img = mask  # Update `img` with the masked version to keep original size

            cv2.destroyWindow("Select Region to Crop")
            show_image_with_matplotlib(img, "Cropped Area with Original Size")

        # Convert to grayscale for further processing
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cropped_image = contour_detection(gray_image)

        preprocessed_image = gaussian_blur(cropped_image)
        binary_image = otsu_binarization(preprocessed_image)

        show_image_with_matplotlib(binary_image, "Preprocessed Image (Gaussian Blur + Otsu Binarization)")

        segmented_image = line_and_word_segmentation(binary_image)
        segmented_path = os.path.join('Results', 'segmented_image.png')
        cv2.imwrite(segmented_path, segmented_image)
        show_image_with_matplotlib(segmented_image, "Final Segmentation with Bounding Boxes")

        # Perform OCR with layout preservation
        extracted_text = pytesseract.image_to_string(binary_image, config="--psm 6")

        # Display the extracted text
        extracted_text_box.delete(1.0, END)
        extracted_text_box.insert(END, extracted_text)
        
        # Calculate and display average confidence score (optional)
        ocr_data = pytesseract.image_to_data(binary_image, output_type=Output.DICT)
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) != -1]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0
        confidence_label.config(text=f"Average Confidence Score: {average_confidence:.2f}%")

root.geometry("800x600")

load_button = Button(root, text="Load Image", command=load_image)
load_button.pack()

extracted_text_label = Label(root, text="Extracted Text:")
extracted_text_label.pack()
extracted_text_box = Text(root, height=10, width=50)
extracted_text_box.pack()

confidence_label = Label(root, text="Average Confidence Score: N/A")
confidence_label.pack()

root.mainloop()