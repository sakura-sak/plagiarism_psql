FROM python:3.11-slim

WORKDIR /app

# Minimal system deps for scientific/python ML wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333
ENV QDRANT_COLLECTION=university_docs

EXPOSE 8765

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8765"]
