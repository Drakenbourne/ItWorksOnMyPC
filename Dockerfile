# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUTF8=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD sh -c "streamlit run source/ai-interview-bot/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"

