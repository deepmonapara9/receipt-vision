import cv2
import pytesseract
import ollama
import json

# This function reads an image from the specified path, converts it to grayscale,
# applies thresholding, and returns the processed image to improve OCR results.
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.jpg', gray)

    # Apply thresholding
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('thresholded_image.jpg', threshold)

    return threshold

#  This function extracts text from the image using Tesseract OCR.
def extract_text(image_path):
    return pytesseract.image_to_string(image_path)

# This function uses the Mistral model to extract structured receipt data from the extracted text.
def ai_extract(text_content):
    prompt = f"""
    You are a receipt parser AI. Below is the text extracted from an image of a store receipt.
    Convert it into a JSON object with the following structure:
    
    {{
        "total": "Total amount in paisa (₹1 = 100 paisa)",
        "business": "Store name",
        "items": [
            {{
                "title": "Item name",
                "quantity": "Quantity purchased",
                "price": "Price in paisa (₹1 = 100 paisa)"
            }}
        ],
        "transaction_timestamp": "YYYY-MM-DD HH:MM:SS"
    }}

    Only return a valid JSON object. Do not include extra text. Here is the extracted text:

    {text_content}
    """

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        json_output = json.loads(response["message"]["content"])
        return json_output
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from the model"}

# Main function
if __name__ == '__main__':
    image_path = "receipt.jpg"

    # Preprocess and extract text
    preprocessed_image = preprocess_image(image_path)
    text_content = extract_text(image_path)

    # Get structured receipt data
    json_data = ai_extract(text_content)

    # Save to a JSON file
    with open('receipt.json', 'w') as f:
        json.dump(json_data, f, indent=4)

    print("Receipt data saved to receipt.json")
