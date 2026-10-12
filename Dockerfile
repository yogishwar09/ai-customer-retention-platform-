# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    RETENTIONIQ_ARTIFACTS_DIR=/data/artifacts RETENTIONIQ_DB_PATH=/data/artifacts/retentioniq.db \
    RETENTIONIQ_REPORTS_DIR=/data/reports
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps -e .
COPY configs ./configs
COPY data ./data
COPY api ./api
COPY ui ./ui
COPY app_pages ./app_pages
COPY styles ./styles
COPY scripts ./scripts
COPY assets ./assets
COPY .streamlit ./.streamlit
COPY app.py ./
RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown -R app /data
USER app
# Train + seed on first start if the volume is empty, then run the requested service.
COPY --chown=app docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD \
    curl -fs http://localhost:8501/_stcore/health || curl -fs http://localhost:8000/health || exit 1
