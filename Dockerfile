# Reproduction image (U8). Does not invent τc: mount a measured
# results/cold_start.json (and optionally the rest of results/) at runtime.
#
#   docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t pact-repro .
#   docker run --rm -v "$PWD/results:/app/results" pact-repro
#
# Then `make reproduce` writes tables, overhead.csv, manifest.json, and figures.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

COPY pact ./pact
COPY configs ./configs
COPY scripts ./scripts
COPY make_figures.py Makefile ./
RUN pip install --no-cache-dir -e . --no-deps

ARG GIT_SHA=
ENV PACT_GIT_SHA=$GIT_SHA

CMD ["make", "reproduce"]
