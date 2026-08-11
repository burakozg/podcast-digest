# Pinned by digest so a rebuild is reproducible (§10.2 supply chain).
# Refresh with:
#   docker buildx imagetools inspect python:3.12-slim | head
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS base

# ffmpeg is required by faster-whisper to decode podcast audio. tini gives the
# container a real init so SIGTERM reaches uvicorn and shutdown is graceful.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Model cache lives on a writable volume, not the read-only root filesystem.
    HF_HOME=/data/work/models \
    XDG_CACHE_HOME=/data/work/cache

WORKDIR /app

# --- dependency layer -------------------------------------------------------
# Installed from a hash-pinned requirements export so the image cannot drift
# even if an upstream release is yanked or re-published (§10.2).
COPY requirements.lock.txt ./
RUN pip install --require-hashes --no-deps -r requirements.lock.txt

# --- application layer ------------------------------------------------------
COPY pyproject.toml README.md ./
COPY podcast_agent ./podcast_agent
RUN pip install --no-deps .

# Non-root, and it must not own the code it runs (§10.2).
RUN useradd --system --uid 10001 --create-home --home-dir /home/podagent podagent \
    && mkdir -p /data/digests /data/work/audio /data/work/models /data/work/cache \
    && chown -R podagent:podagent /data
USER 10001

EXPOSE 8080

# Uses the unauthenticated /healthz, which fails 503 when CouchDB is unreachable
# or the scheduler has stopped (§9).
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=8).status==200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["podcast-agent"]
