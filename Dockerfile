FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads exports cache

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE $PORT

# Use non-root user for security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser:appuser /app
USER appuser

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]