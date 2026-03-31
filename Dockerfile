FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --upgrade pip && pip install -r requirements-deploy.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
