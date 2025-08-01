from fastapi import FastAPI, UploadFile, File , Depends
from fastapi.responses import JSONResponse, StreamingResponse

import os
import io
import cv2

from imutils.perspective import four_point_transform
import numpy as np
import pandas as pd
import json


from app.ocr.ocr_parser import tensseract_receipt_parser 
from app.llama.llama_receipt_parser import ask_llava, ask_llama, encode_image_to_base64
from app.forecast import arima_forecast, lstm_forecast, Prophet, sarimax_forecast
from app.forecast import check_saving_target_yearly, check_saving_target, forecast_all_categories, forecast_category
from app.offers.offers import get_matching_offers, get_offers
from app.ather.ather_bot import ask_llama3
import requests
import base64

from app.forecasting.forecast_prophet import prophet_forecast_category, prophet_forecast_all_categories, prophet_check_saving_target, prophet_check_saving_target_yearly
from app.Module.ForcastCategories import ForecastCategories

from app.supermartket.suppermartket_offers import get_tamimi_supermarket_offer
app = FastAPI()
history = ""

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/OCR/")
def read_items():
    return {"OCR" : "Test"}

@app.get("/Prophet_Forcast")
def read_items():
    return {"Forcast" : "Prophet"}

@app.get("/Arima_Forcast")
def read_items():
    return {"Forcast" : "Arima"}

@app.get("/supermartket-offer/tamimi")
async def get_tamimi_offer():
    return get_tamimi_supermarket_offer()
    
@app.post("/process-and-return-image")
async def process_and_return_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Apply edge detection
    edges = cv2.Canny(img, 100, 200)
    
    # Encode the processed image
    _, encoded_img = cv2.imencode('.png', edges)
    
    # Return the image directly
    return StreamingResponse(
        io.BytesIO(encoded_img.tobytes()),
        media_type="image/png"
    )

@app.post("/ocr/parse-receipt-tenssoract")
async def parse_receipt(file: UploadFile = File(...)):
    # Check if the uploaded file is an image
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"message": "File is not an image"}
        )
    
    # Read image file
    contents = await file.read()
    
    text = tensseract_receipt_parser(contents)
    return {
        "content": text
    }

@app.post("/ocr/parse-receipt-llava")
async def parse_receipt(file: UploadFile = File(...)):
    image_bytes = await file.read()
    base64_image = encode_image_to_base64(image_bytes)
    try:
        llava_response = ask_llava(base64_image)
        return JSONResponse(content={"result": llava_response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/ocr/parse-receipt-llama-vision")
async def parse_receipt(file: UploadFile = File(...)):
    
    """ parse-receipt-llama-vision"

    Returns:
        [type]: [description]
    """    
    image_bytes = await file.read()
    base64_image = encode_image_to_base64(image_bytes)
    try:
        ask_llama(base64_image)
        # return JSONResponse(content={"result": llama_response})
    except Exception as e:
        print(e)
        # return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/forecast/prophet_forecast")
def read_items(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    result = prophet_forecast_all_categories(df)
    result = {'results': [obj.__dict__ for obj in result]}
    print(result)
    return result

@app.post("/offers/for_you_offers")
def get_general_offers():
    offers_folder = os.path.join(os.path.dirname(__file__), "..", "offers", "tamimi-product.json")
    offers_folder = os.path.abspath(offers_folder)
    purchases_folder = os.path.join(os.path.dirname(__file__), "purchases.json")
    purchases_folder = os.path.abspath(purchases_folder)
    with open(purchases_folder) as f:
        purchase = json.load(f)
    with open(offers_folder) as f:
        offers = json.load(f) 
        matched_offers = get_matching_offers(purchase, offers)
        return matched_offers

@app.post("/offers/general_offers")
def get_general_offers():
    folder = os.path.join(os.path.dirname(__file__), "..", "offers", "tamimi-product.json")
    folder = os.path.abspath(folder)
    with open(folder) as f:
        offers = json.load(f) 
    matched_offers = get_offers(offers)
    return matched_offers

@app.post("/chatbot/ather")
def ask_ather(query: str):
    global history
    (ather, history) = ask_llama3(query, history)
    print(ather)
    return ather
