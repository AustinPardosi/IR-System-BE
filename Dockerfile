FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY .env .env

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 