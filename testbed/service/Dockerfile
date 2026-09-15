FROM python:3.11-slim
WORKDIR /app
COPY app.py /app/app.py
ENV WORK_MS=20
ENV CONCURRENCY=1
EXPOSE 8080
CMD ["python", "-u", "/app/app.py"]
