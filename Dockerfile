FROM python:3.12-slim

# curl is only used by the compose healthcheck; build-essential covers any
# dependency in requirements.txt that doesn't ship a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime state (db/, logs/, secrets/, data/, sections/) is mounted from the
# host via docker-compose — these mkdirs just guarantee the paths exist even
# on a first run before any volume has been populated.
RUN mkdir -p db logs secrets data sections app/static/catalogue

EXPOSE 8501

CMD ["python", "run.py"]
