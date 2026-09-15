FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY pact /app/pact
COPY configs /app/configs
RUN pip install --no-cache-dir .
# Fractional CPU is reserved in compose (cpus: 0.25) so controller cost
# counts against the same host budget as the workload.
CMD ["python", "-m", "pact.loop"]
