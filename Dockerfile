# Use slim Python base
FROM python:3.12-slim

# Prevent buffered stdout/stderr (helpful on Railway)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list first for better layer caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy project files
COPY main.py ./

# Default runtime env vars (override on Railway)
# Alpaca
ENV APCA_API_KEY_ID="" \
    APCA_API_SECRET_KEY="" \
    APCA_API_BASE_URL="https://paper-api.alpaca.markets" \
    APCA_DATA_BASE_URL="https://data.alpaca.markets"
# Google Sheets
ENV GOOGLE_SHEET_NAME="Trading Log" \
    GOOGLE_CREDS_JSON=""  # paste service account JSON here or set as Railway variable
# Kraken (optional for now – establishes connectivity)
ENV KRAKEN_API_KEY="" \
    KRAKEN_API_SECRET=""
# Behavior tuning
ENV REFRESH_MINUTES=30 \
    BATCH_SIZE=50 \
    SLEEP_BETWEEN_BATCHES=5

CMD ["python", "main.py"]
