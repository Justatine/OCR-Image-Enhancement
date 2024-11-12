import tkinter as tk

# Function to count incorrect words
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
def compare_words():
    # Extract the text input from the textbox (treated as reference)
    reference_text = text_box.get("1.0", tk.END).strip()  # Get reference text
    reference_words = reference_text.split()  # Split the reference text into words
    
    # Example extracted words (could be input elsewhere in your app)
    extracted_words = ["The", "quick", "brown", "fox", "jump", "over", "the", "lazy", "dog"]
    
    # Count incorrect words by comparing the extracted and reference words
    incorrect_count = count_incorrect_words(extracted_words, reference_words)
    
    # Display the result
    result_label.config(text=f"Incorrect Words: {incorrect_count}")  # Show number of incorrect words

# Create the main window
root = tk.Tk()
root.title("Word Comparison")

# Create a textbox for input (reference words)
text_box = tk.Text(root, wrap="word", width=50, height=10)
text_box.pack(pady=10)

# Create a button to compare words
compare_button = tk.Button(root, text="Compare Words", command=compare_words)
compare_button.pack(pady=5)

# Label to display the result
result_label = tk.Label(root, text="Incorrect Words: 0")
result_label.pack(pady=10)

# Run the application
root.mainloop()
