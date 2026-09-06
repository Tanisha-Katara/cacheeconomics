FROM python:3.12.14-slim-trixie@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime

ARG VCS_REF="unknown"
LABEL org.opencontainers.image.title="cacheeconomics control plane" \
      org.opencontainers.image.description="Prompt-free hosted API and analysis worker" \
      org.opencontainers.image.source="https://github.com/Tanisha-Katara/cacheeconomics" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/harness \
    OTEL_SERVICE_NAME=cacheeconomics-api \
    OTEL_TRACES_EXPORTER=none \
    OTEL_METRICS_EXPORTER=none \
    OTEL_LOGS_EXPORTER=none \
    OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SANITIZE_FIELDS=authorization,cookie,set-cookie,x-api-key

WORKDIR /app

COPY services/control_plane/requirements-runtime.lock /app/requirements-runtime.lock
COPY services/control_plane/requirements-observability.lock /app/requirements-observability.lock
RUN pip install --no-cache-dir --requirement /app/requirements-observability.lock

COPY services/control_plane/src /app/src
COPY harness/cacheeconomics /app/harness/cacheeconomics
COPY services/control_plane/migrations /app/migrations
COPY services/control_plane/alembic.ini /app/alembic.ini

RUN addgroup --system app && adduser --system --ingroup app app \
    && chown -R app:app /app
USER app

FROM runtime AS migrate
CMD ["alembic", "-c", "/app/alembic.ini", "upgrade", "head"]

FROM runtime AS worker
ENV OTEL_SERVICE_NAME=cacheeconomics-worker
HEALTHCHECK NONE
CMD ["opentelemetry-instrument", "python", "-m", "cacheeconomics_control_plane.worker"]

FROM runtime AS api
ENV OTEL_SERVICE_NAME=cacheeconomics-api
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

CMD ["opentelemetry-instrument", "uvicorn", "cacheeconomics_control_plane.app:app", "--host", "0.0.0.0", "--port", "8000"]
