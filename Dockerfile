FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 DB_PATH=/app/data/checkpoints.sqlite

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Checkpoints live here; mount a volume to keep plans across container restarts.
VOLUME ["/app/data"]
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
