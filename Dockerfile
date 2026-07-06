FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Harness contract (Track 2): reads /input/tasks.json, writes /output/results.json.
ENTRYPOINT ["python", "agent.py"]
