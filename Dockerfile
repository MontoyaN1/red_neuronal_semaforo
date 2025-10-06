FROM python:3.10-slim

# Instalar SUMO y dependencias
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]