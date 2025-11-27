FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including espeak-ng and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git \
    ffmpeg \
    espeak-ng \
    espeak-data \
    libespeak-ng-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Kokoro from GitHub
RUN pip install --no-cache-dir git+https://github.com/hexgrad/kokoro.git

# Install additional audio processing libraries
RUN pip install --no-cache-dir \
    soundfile \
    gTTS \
    PyPDF2 \
    torch \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the Python script
COPY pdf_to_audiobook.py .

# Create directories for input/output
RUN mkdir -p input output

# Set the entrypoint
CMD ["python", "pdf_to_audiobook.py"]