FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --only-binary=:all: -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["sh", "-c", "python -m streamlit run dashboard/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
