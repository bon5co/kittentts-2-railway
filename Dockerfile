FROM python:3.12-slim-bookworm

# ffmpeg encodes mp3/opus/flac for the OpenAI-style response_format values.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# kitten_tts_cpp pins torch==2.13.0 (its decoder is TorchScript). The CPU wheel keeps the
# image ~2 GB smaller than the CUDA build pip would pick by default.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
       kitten_tts_cpp==0.2.0 \
       fastapi==0.118.0 \
       "uvicorn[standard]==0.37.0" \
       python-multipart==0.0.20

COPY server.py NOTICE /app/

# The model weights are NOT in this image. They are licensed under the Stellon Labs
# Community License and are fetched from Hugging Face on first boot into the volume.
ENV MAX_INPUT_CHARS=300 \
    KITTEN_REVISION=1f00fee525c7b2e30bbe9ca9ba67529734fce9c9 \
    KITTEN_CACHE_DIR=/data/cache \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

CMD ["python", "/app/server.py"]
