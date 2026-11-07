FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY weights ./weights
COPY samples ./samples
COPY app.py .

# run as a normal user (also needed on Hugging Face Spaces)
RUN useradd -m appuser
USER appuser

EXPOSE 7860
CMD ["python", "app.py"]
