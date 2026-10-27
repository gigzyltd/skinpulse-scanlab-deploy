# Stateless Scan Lab API. Build from this directory:
#   docker build -f services/scanlab/Dockerfile -t scanlab-api services/scanlab
# The image does not download a landmark model and does not enable paid providers.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/services \
    PORT=10000 \
    FORWARDED_ALLOW_IPS=

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY cv/requirements.txt /tmp/scanlab-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/scanlab-requirements.txt \
    && rm /tmp/scanlab-requirements.txt

COPY __init__.py /app/services/scanlab/__init__.py
COPY cv /app/services/scanlab/cv
COPY providers /app/services/scanlab/providers
COPY score /app/services/scanlab/score
COPY server /app/services/scanlab/server

RUN rm -rf /app/services/scanlab/cv/tests /app/services/scanlab/server/tests \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin scanlab \
    && chown -R scanlab:scanlab /app

USER scanlab
EXPOSE 10000
CMD ["python", "-m", "scanlab.server"]
