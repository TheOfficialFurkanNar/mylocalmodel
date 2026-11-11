"""
TTS (Text-to-Speech) tool using kokoro-onnx (no espeak-ng required).

Model files are downloaded from GitHub releases on first use and cached
in tools/tts_cache/.  No CUDA is needed; ONNX Runtime runs on CPU.

Voices (from kokoro-onnx v1.0):
  American English  — af_heart, af_sky, af_bella, af_sarah, af_nicole,
                      am_adam, am_michael
  British English   — bf_emma, bf_isabella, bm_george, bm_lewis
"""

from __future__ import annotations

import asyncio
import io
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional
from urllib.request import urlretrieve

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

try:
    from config import TTSConfig
except ImportError:
    from mylocalmodel.config import TTSConfig

# ---------------------------------------------------------------------------
# Model file URLs (GitHub releases — no espeak-ng, pure ONNX)
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/kokoro-v1.0.onnx"
)
_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/voices-v1.0.bin"
)

_CACHE_DIR = Path(__file__).resolve().parent / "tts_cache"
_MODEL_PATH = _CACHE_DIR / "kokoro-v1.0.onnx"
_VOICES_PATH = _CACHE_DIR / "voices-v1.0.bin"

SAMPLE_RATE = 24_000  # Kokoro outputs 24 kHz PCM

# ---------------------------------------------------------------------------
# Voice catalogue
# ---------------------------------------------------------------------------

# Maps voice_id -> display metadata.
# lang "en-us" = American English, "en-gb" = British English.
VOICES: dict[str, dict] = {
    # American English — female
    "af_heart":    {"name": "Heart",    "gender": "F", "accent": "US", "lang": "en-us"},
    "af_sky":      {"name": "Sky",      "gender": "F", "accent": "US", "lang": "en-us"},
    "af_bella":    {"name": "Bella",    "gender": "F", "accent": "US", "lang": "en-us"},
    "af_sarah":    {"name": "Sarah",    "gender": "F", "accent": "US", "lang": "en-us"},
    "af_nicole":   {"name": "Nicole",   "gender": "F", "accent": "US", "lang": "en-us"},
    # American English — male
    "am_adam":     {"name": "Adam",     "gender": "M", "accent": "US", "lang": "en-us"},
    "am_michael":  {"name": "Michael",  "gender": "M", "accent": "US", "lang": "en-us"},
    # British English — female
    "bf_emma":     {"name": "Emma",     "gender": "F", "accent": "UK", "lang": "en-gb"},
    "bf_isabella": {"name": "Isabella", "gender": "F", "accent": "UK", "lang": "en-gb"},
    # British English — male
    "bm_george":   {"name": "George",   "gender": "M", "accent": "UK", "lang": "en-gb"},
    "bm_lewis":    {"name": "Lewis",    "gender": "M", "accent": "UK", "lang": "en-gb"},
}


def list_voices() -> list[dict]:
    """Return the voice catalogue sorted by accent, gender, name."""
    return sorted(
        [{"id": vid, **meta} for vid, meta in VOICES.items()],
        key=lambda v: (v["accent"], v["gender"], v["name"]),
    )

# ---------------------------------------------------------------------------
# Model file management
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path, label: str) -> None:
    """Download url to dest, printing a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[TTS] Downloading {label}…")

    def _reporthook(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r[TTS]   {label}: {pct}%", end="", flush=True)

    urlretrieve(url, dest, reporthook=_reporthook)
    print(f"\r[TTS]   {label}: done ({dest.stat().st_size // 1024} KB)")


def ensure_model_files() -> tuple[Path, Path]:
    """
    Download model + voices files to tts_cache/ if not already present.
    Returns (model_path, voices_path).
    """
    if not _MODEL_PATH.exists():
        _download_with_progress(_MODEL_URL, _MODEL_PATH, "kokoro-v1.0.onnx")
    if not _VOICES_PATH.exists():
        _download_with_progress(_VOICES_URL, _VOICES_PATH, "voices-v1.0.bin")
    return _MODEL_PATH, _VOICES_PATH

# ---------------------------------------------------------------------------
# KokoroTTS class
# ---------------------------------------------------------------------------

class KokoroTTS:
    """
    Wrapper around kokoro-onnx that:
    - Downloads model files on first use to tools/tts_cache/
    - Runs entirely on CPU (no CUDA required, no espeak-ng required)
    - Streams audio as an async generator of (samples, sample_rate) chunks
    """

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self.config = config or TTSConfig()
        self._kokoro: Optional[Kokoro] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _get_kokoro(self) -> Kokoro:
        with self._lock:
            if self._kokoro is None:
                model_path, voices_path = ensure_model_files()
                print("[TTS] Loading Kokoro ONNX model…")
                self._kokoro = Kokoro(str(model_path), str(voices_path))
                print("[TTS] Kokoro ready.")
            return self._kokoro

    # ------------------------------------------------------------------
    # Streaming synthesis (async generator)
    # ------------------------------------------------------------------

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> AsyncGenerator[tuple[np.ndarray, int], None]:
        """
        Async generator that yields (samples: np.ndarray, sample_rate: int)
        per sentence chunk.  Runs ONNX inference in a thread pool so the
        FastAPI event loop is never blocked.
        """
        voice = voice or self.config.default_voice
        speed = speed if speed is not None else self.config.speed
        lang = VOICES.get(voice, {}).get("lang", "en-us")

        kokoro = await asyncio.to_thread(self._get_kokoro)
        stream = kokoro.create_stream(text, voice=voice, speed=speed, lang=lang)
        async for samples, sample_rate in stream:
            if samples is not None and len(samples) > 0:
                yield samples, sample_rate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
        """Encode a float32 numpy array as an in-memory 16-bit WAV blob."""
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

_tts_instance: Optional[KokoroTTS] = None
_tts_lock = threading.Lock()


def get_tts() -> KokoroTTS:
    """Return the process-wide KokoroTTS singleton."""
    global _tts_instance
    with _tts_lock:
        if _tts_instance is None:
            _tts_instance = KokoroTTS()
    return _tts_instance


# ---------------------------------------------------------------------------
# CLI smoke-test:  python tools/tts.py "Hello world" af_heart
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_text = sys.argv[1] if len(sys.argv) > 1 else "Hello! Kokoro ONNX is running locally."
    test_voice = sys.argv[2] if len(sys.argv) > 2 else "af_heart"
    out_path = Path("tts_test_output.wav")

    print(f"=== Kokoro ONNX TTS Test ===")
    print(f"Text : {test_text!r}")
    print(f"Voice: {test_voice}")

    async def _run() -> None:
        tts = get_tts()
        all_chunks: list[np.ndarray] = []
        async for samples, sr in tts.synthesize_stream(test_text, voice=test_voice):
            all_chunks.append(samples)
        if all_chunks:
            combined = np.concatenate(all_chunks)
            wav = tts.chunk_to_wav_bytes(combined)
            out_path.write_bytes(wav)
            print(f"Output: {out_path} ({len(combined) / SAMPLE_RATE:.2f}s, {len(all_chunks)} chunk(s))")
        else:
            print("No audio generated.")

    asyncio.run(_run())
