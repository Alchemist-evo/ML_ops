FROM python:3.11-slim

WORKDIR /app

# Install EXACTLY what training used -- this is the whole point
COPY model_export/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_export/ /app/model_export/
COPY score_batch.py .

ENTRYPOINT ["python", "score_batch.py"]
