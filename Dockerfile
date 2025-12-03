# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей для matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libfreetype6-dev \
    libpng-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements-pro.txt .
RUN pip install --no-cache-dir -r requirements-pro.txt

# Копирование кода
COPY . .

# Создание пользователя для безопасности
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Запуск приложения
CMD ["python", "bot_mvp_phase3.py"]
