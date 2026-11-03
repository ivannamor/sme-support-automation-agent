FROM python:3.12-slim

# curl is only used by the compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY . .

# Runtime state (db/, logs/, secrets/, data/, sections/, app/static/) is mounted
# from the host via docker-compose — these mkdirs just guarantee the paths exist
# even on a first run before any volume has been populated. The app runs as a
# non-root user who can write only there: the code stays owned by root.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p db logs secrets data sections app/static/catalogue \
    && chown -R appuser:appuser db logs secrets data sections app/static
USER appuser

# Show output in `docker logs` right away. The code in an image never changes,
# so Streamlit doesn't need to watch it (these override .streamlit/config.toml).
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8501

CMD ["python", "run.py"]
