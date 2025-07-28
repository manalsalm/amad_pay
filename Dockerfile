# pull the official docker image
FROM python:3.11.1-slim

# Install Tesseract and its dependencies
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=secret
ENV POSTGRES_DB=mydatabase
ENV DATABASE_URL=postgresql://postgres:secret@db:5432/mydatabase

# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy project
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]