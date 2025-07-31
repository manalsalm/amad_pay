# Amad Pay

This repository hosts a script and a Docker-compose setup for Amad Pay.

## Receipt OCR

Optical Character Recognition (OCR) on receipt images.
- using tenssact.
- ollam (llava, llama3.2-vision)

## Ather Chatbot

Ather is a chatbot that helps predict the outcomes of financial decisions.
- using pandas.
- ollam (llama3)

## Offers

Offer recommendation feature that suggests personalized deals based on the user's most frequently purchased products, as well as available general promotions.

## Prerequisites

- Python 3.x
- Docker
- Docker-compose

## Installation

### Clone the repository

```bash
git clone https://github.com/bhimrazy/amad_pay.git
cd receipt-ocr
```

### Set up Python environment (if not using Docker)
- Install [tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- Install Docker.
- Run Ollama in Docker
```bash
# Pull the Ollama Docker image
docker pull ollama/ollama

# Run the container 
## GPU running on port 11434 (default)
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
## CPU running on port 11435
docker run -d -v ollamacpu:/root/.ollamacpu -p 11435:11434 --name ollamacpu ollama/ollama

# On each container pull the llama3.2-vision image. Run the command
docker exec ollama ollama pull llama3.2-vision

# On each container pull the llava image. Run the command
docker exec ollama ollama pull llava

# On each container pull the llama3 image. Run the command
docker exec ollama ollama pull llama3
```
  

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # For Windows, use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the script locally

#### `fastapi`

The `app/main.py` script performs OCR on an input image of a receipt.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```


### Using Docker-compose

The repository includes a Docker-compose setup for running the OCR engine as a service.

```bash
docker-compose up
```

Once the service is up and running, you can perform OCR on receipt images by sending a POST request to `http://localhost:8000/ocr/` with the image file.

## API Endpoint

The OCR functionality can be accessed via a FastAPI endpoint:

- **POST** `/parse-receipt`: Upload a receipt image file to perform OCR. The response will contain the extracted text from the receipt.

Example usage with cURL:

```bash
curl -X POST -F "file=@C:\Users\manal_qckxaa\Desktop\receipt-ocr\images\receipt1.jpg" http://localhost:8000/parse-receipt
```


## License

This project is licensed under the terms of the MIT license.

## References
- [Automatically OCR’ing Receipts and Scans](https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/)


