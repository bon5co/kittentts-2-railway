"""OpenAI-compatible speech API for Kitten-TTS 2 on CPU (kitten_tts_cpp, llama.cpp fork).

POST /v1/audio/speech   JSON {model, input, voice, response_format, stream}
GET  /v1/audio/voices   built-in voices (also GET /v1/voices)
GET  /v1/models
GET  /health            200 always; body.status is loading | ready | error
GET  /                  attribution page ("Powered by KittenML", licence)

Bearer auth (API_KEY) is enforced on /v1/*. Weights download on first boot into
KITTEN_CACHE_DIR (the volume), in a background thread so /health answers immediately.
"""

import asyncio
import io
import os
import subprocess
import threading
import traceback
import wave
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

API_KEY = os.environ.get("API_KEY", "")
REPO = os.environ.get("KITTEN_REPO", "KittenML/kitten-tts-2")
REVISION = os.environ.get("KITTEN_REVISION") or None
DECODER = os.environ.get("KITTEN_DECODER", "student_w4")
CACHE_DIR = os.environ.get("KITTEN_CACHE_DIR", "/data/cache")
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "Bruno")
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "1500"))
MODEL_ID = "kitten-tts-2"
SAMPLE_RATE = 24000

FFMPEG_FORMATS = {
    "mp3": ("mp3", "audio/mpeg"),
    "opus": ("ogg", "audio/ogg"),
    "aac": ("adts", "audio/aac"),
    "flac": ("flac", "audio/flac"),
}


def cgroup_cpus() -> int:
    """Threads follow the container's CPU quota; the runtime would otherwise read the host's cores."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


THREADS = int(os.environ.get("KITTEN_THREADS") or min(cgroup_cpus(), 16))

state = {"status": "loading", "error": None, "voices": []}
model = None


def load_model():
    global model
    try:
        from kitten_tts_cpp import KittenTTS

        os.makedirs(CACHE_DIR, exist_ok=True)
        kwargs = {"decoder": DECODER, "threads": THREADS, "cache_dir": CACHE_DIR}
        if REVISION:
            kwargs["revision"] = REVISION
        model = KittenTTS(REPO, **kwargs)
        state["voices"] = sorted(model.available_voices)
        state["status"] = "ready"
        print(f"ready: {REPO}@{REVISION} decoder={DECODER} threads={THREADS} voices={len(state['voices'])}", flush=True)
    except Exception as exc:  # download failure, unsupported CPU, bad revision
        traceback.print_exc()
        state["status"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"[:300]


app = FastAPI(title="Kitten-TTS 2 API", description="OpenAI-compatible TTS, CPU. Powered by KittenML.")
gen_lock = asyncio.Lock()  # native inference is serialized by the library anyway


@app.on_event("startup")
def start_loader():
    threading.Thread(target=load_model, daemon=True).start()


@app.middleware("http")
async def bearer_auth(request: Request, call_next):
    if API_KEY and request.url.path.startswith("/v1/"):
        header = request.headers.get("authorization", "")
        if header.removeprefix("Bearer ").strip() != API_KEY:
            return JSONResponse({"error": {"message": "invalid api key"}}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1)
    model: str = MODEL_ID
    voice: str = DEFAULT_VOICE
    response_format: Literal["wav", "pcm", "mp3", "opus", "aac", "flac"] = "mp3"
    stream: bool = False  # chunk-level streaming, wav and pcm only


def pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def wav_bytes(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def encode(pcm: bytes, fmt: str) -> tuple[bytes, str]:
    if fmt == "wav":
        return wav_bytes(pcm), "audio/wav"
    if fmt == "pcm":
        return pcm, "audio/pcm"
    ffmpeg_fmt, media_type = FFMPEG_FORMATS[fmt]
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", ffmpeg_fmt, "pipe:1"],
        input=wav_bytes(pcm), capture_output=True)
    if proc.returncode != 0:
        raise HTTPException(500, f"encoding to {fmt} failed: {proc.stderr.decode()[:200]}")
    return proc.stdout, media_type


def require_ready():
    if state["status"] == "error":
        raise HTTPException(500, f"model failed to load: {state['error']}")
    if state["status"] != "ready":
        raise HTTPException(503, "model is loading (first boot downloads ~1 GB); poll GET /health until status is ready",
                            headers={"Retry-After": "15"})


@app.get("/health")
def health():
    return {"status": state["status"], "model": MODEL_ID, "error": state["error"], "threads": THREADS}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "KittenML"}]}


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
def voices():
    require_ready()
    return {"voices": state["voices"]}


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    require_ready()
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"input is {len(req.input)} chars; max {MAX_INPUT_CHARS} per request (CPU inference is slow, split long text)")
    if req.voice not in state["voices"]:
        raise HTTPException(400, f"unknown voice {req.voice!r}; GET /v1/audio/voices lists them")
    loop = asyncio.get_running_loop()

    if req.stream:
        if req.response_format not in ("wav", "pcm"):
            raise HTTPException(400, "stream=true supports response_format wav or pcm")

        async def chunks():
            async with gen_lock:
                it = model.generate_stream(req.input, voice=req.voice)
                if req.response_format == "wav":
                    # open-ended WAV header: players read until the connection closes
                    yield wav_bytes(b"")[:40] + b"\xff\xff\xff\xff"
                while True:
                    chunk = await loop.run_in_executor(None, next, it, None)
                    if chunk is None:
                        break
                    yield pcm16(chunk)

        media = "audio/wav" if req.response_format == "wav" else "audio/pcm"
        return StreamingResponse(chunks(), media_type=media)

    async with gen_lock:
        try:
            samples = await loop.run_in_executor(None, lambda: model.generate(req.input, voice=req.voice))
        except Exception as exc:
            raise HTTPException(400, f"generation failed: {exc}") from exc
    audio, media_type = encode(pcm16(samples), req.response_format)
    return Response(content=audio, media_type=media_type)


@app.get("/", response_class=HTMLResponse)
def index():
    return ("<!doctype html><title>Kitten-TTS 2 API</title><h1>Kitten-TTS 2 API</h1>"
            "<p>OpenAI-compatible text-to-speech, CPU. <b>Powered by KittenML</b>.</p>"
            "<p>POST /v1/audio/speech with a Bearer token. GET /health for status.</p>"
            "<p>Model weights: Stellon Labs Community License "
            "(<a href='https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md'>text</a>); "
            "commercial use requires registration with Stellon Labs.</p>")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
