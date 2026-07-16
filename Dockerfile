FROM python:3.13-slim

WORKDIR /app

# System deps needed to build some Python packages (e.g. regex, mlflow deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK corpora at build time
RUN python -m nltk.downloader stopwords wordnet

# Copy application code and required artifacts
COPY backend/app.py .
COPY tfidf_vectorizer.pkl .

# DAGSHUB_PAT must be passed at runtime, e.g.:
# docker run -e DAGSHUB_PAT=xxxx -p 8000:8000 <image>

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]