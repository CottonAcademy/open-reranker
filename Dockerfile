FROM python:3.12-slim
WORKDIR /app
COPY huggingface.co /app/huggingface.co

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bge_ranker_v2_m3.py .

EXPOSE 8000
CMD ["python", "bge_ranker_v2_m3.py"]

