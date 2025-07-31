import requests
import base64
from fastapi import FastAPI, UploadFile, File , Depends

def encode_image_to_base64(file):
    return base64.b64encode(file).decode('utf-8')

def ask_llava(base64_image):
    prompt = """Act as an OCR assistant. Analyze the provided image. Extract structured data in the format with make price  conversion to Saudi Riyal:
{
  "store_name": "...",
  "date": "...",
  "total_amount": "...",
  "items": [{"name": "...", "price": "...", "quantity": "..."}]
}"""

    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)

    # show the raw output of the OCR process
    print("[INFO] raw output:")
    print("==================")
    print(response.json()["response"])
    print("\n")    
    return response.json()["response"]

def ask_llama(base64_image):
    prompt = """Act as an OCR assistant. Analyze the provided image. Extract structured data in the format with make price  conversion to Saudi Riyal:
{
  "store_name": "...",
  "date": "...",
  "total_amount": "...",
  "items": [{"name": "...", "price": "...", "quantity": "..."}]
}"""

    payload = {
        "model": "llama3.2-vision",#"llava",
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)

    # show the raw output of the OCR process
    print("[INFO] raw output:")
    print("==================")
    print(response.json()["response"])
    print("\n")    
    return response.json()["response"]

file_path = "C:/Users/Nora-Basalamah/Downloads/WhatsApp Image 2025-07-31 at 10.42.27 AM.jpeg"
with open(file_path, "rb") as f:
    image_bytes = f.read()
base64_image = encode_image_to_base64(image_bytes)
try:
    llama_response = ask_llama(base64_image)
    print(llama_response)
except Exception as e:
    print("error")