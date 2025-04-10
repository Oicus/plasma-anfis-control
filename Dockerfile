FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src
CMD ["python", "src/anfis_model.py"]
