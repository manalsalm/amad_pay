import requests
import base64
from fastapi import FastAPI, UploadFile, File , Depends
import json
import os

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
    prompt = """Extract only the data from the image in the following exact JSON format. Do not include any explanation or extra text. Return only valid JSON and all as string:
{
  "store_name": "...",
  "date": "...",
  "total_amount": "...",
  "items": [
    {
      "name": "...",
      "price": "...",
      "quantity": "..."
    }
  ]
}"""

    payload = {
        "model": "llama3.2-vision",#"llava",
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)

    # show the raw output of the OCR process
    # print("[INFO] raw output:")
    # print("==================")
    # print(response.json()["response"])
    # print("\n")    
    print(response.json()["response"])
    new_data = json.loads(response.json()["response"])
    json_file = os.path.join(os.path.dirname(__file__), "..", "purchases.json")
    json_file = os.path.abspath(json_file)
    
    # Load existing data (if any)
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Append new data
    existing_data.append(new_data)

    # Write back
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    return response.json()["response"]