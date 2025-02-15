# Receipt Vision 🧾🔍 

Receipt Vision is a **local AI-powered receipt parser** that extracts structured data from store receipts using **OCR (Tesseract)** and the **Mistral model** running on Ollama.  

## ✨ Features  
- Extracts text from receipts using **Tesseract OCR**  
- Uses **Mistral (via Ollama)** to parse text into **structured JSON**  
- Converts amounts to **Indian paisa (₹1 = 100 paisa)**  
- **Works offline** (No internet required after model download)  
- Supports **multiple currencies** (₹, $, €, £) with conversion settings  

---  

## ⚙️ Installation  

### 1️⃣ Clone the Repository  
```bash  
git clone https://github.com/yourusername/receipt-vision.git  
cd receipt-vision  
```  

To install dependencies from the file:  
```bash  
pip install -r requirements.txt  
```  

### 2️⃣ Prerequisites  
Ensure you have the following installed on your system:  
- **Python 3.10+**  
- **Tesseract OCR** ([Install Guide](https://github.com/tesseract-ocr/tesseract))  
- **Ollama** ([Install Guide](https://ollama.com))  
- Required Python libraries:  
  ```bash  
  pip install opencv-python pytesseract ollama  
  ```  

### 3️⃣ Download Mistral Model  
Pull the **Mistral model** using:  
```bash  
ollama pull mistral  
```  
---

## 🚀 Usage  

### 1️⃣ Place your receipt image  
Put the receipt image (e.g., `receipt.jpg`) in the project folder.  

### 2️⃣ Run the script  
```bash  
python main.py  
```  

### 3️⃣ View extracted data  
The extracted JSON is saved in **receipt.json**:  
```json  
{  
    "total": 15000,  
    "business": "Supermart",  
    "items": [  
        {  
            "title": "Milk",  
            "quantity": 2,  
            "price": 6000  
        },  
        {  
            "title": "Bread",  
            "quantity": 1,  
            "price": 3000  
        }  
    ],  
    "transaction_timestamp": "2025-02-13 14:30:00",  
    "currency": "INR"  
}  
```  

---  

## 🌍 Multi-Currency Support  
Receipt Vision now supports multiple currencies. The currency type is detected from the receipt text and included in the JSON output. Future updates will add automatic currency conversion.  

Supported currencies:  
- **Indian Rupees (₹) - INR**  
- **US Dollars ($) - USD**  
- **Euros (€) - EUR**  
- **British Pounds (£) - GBP**  

---  

## 🛠️ Troubleshooting  
### **1️⃣ OCR is inaccurate?**  
- Ensure good **lighting & image quality**  
- Try increasing contrast before processing  

### **2️⃣ Model not responding?**  
- Restart Ollama:  
  ```bash  
  ollama serve  
  ```  

### **3️⃣ Invalid JSON output?**  
- Ensure the **Mistral model** is installed correctly  
- Check if **Ollama is running**  

---  


## 🔥 Future Improvements  
- Add **Mixtral** for better accuracy  
- Improve **receipt pre-processing**  
- Support **multiple languages**  
- **Automatic currency conversion**  

---  

## 👨‍💻 Author  
Developed by **Deep** 🚀  

