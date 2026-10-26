"""Korean speech recognition server specialized for voice-phishing calls.

Model configuration (voice-phishing domain):

    Whisper-large-v3 + LoRA adapter trained on real voice-phishing calls
    (pseudo-label self-training after telephone-channel pre-adaptation), merged in fp16
    + Trie keyword boosting at decoding time (keywords/keywords_vp_top100.txt,
      lambda=1.0, lambda_start=0)
    decode: language=ko, task=transcribe, num_beams=5, max_new_tokens=440

API
    GET  /health                -> {"status": "ok", ...}
    POST /v1/transcribe         body = audio file bytes (WAV/FLAC/...)
                                -> {"text": str, "silent": bool,
                                    "duration_sec": float, "elapsed_sec": float}

Run
    python asr_server.py --base-model <whisper-large-v3> --adapter <adapter_dir> \
        --keywords keywords/keywords_vp_top100.txt --port 8000
"""
from __future__ import annotations

import argparse
import io
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from keyword_boost import KeywordBoostProcessor

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0  # Whisper input window
# Chunks quieter than this (about -50 dBFS) are not decoded: Whisper tends to
# hallucinate short phrases on silence.
SILENCE_RMS = 0.003


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio(data: bytes) -> np.ndarray:
    """Audio bytes -> 16 kHz mono float32."""
    x, sr = sf.read(io.BytesIO(data))
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
    return x.astype(np.float32)


def adapter_name(adapter: str | None) -> str | None:
    """Readable adapter name; a PEFT checkpoint is often saved as <name>/adapter_final."""
    if not adapter:
        return None
    p = Path(adapter).resolve()
    return p.parent.name if p.name == "adapter_final" else p.name


class Recognizer:
    def __init__(self, base_model: str, adapter: str | None, keywords: str | None,
                 device: str, num_beams: int = 5, max_new_tokens: int = 440,
                 boost_lambda: float = 1.0, boost_lambda_start: float = 0.0):
        self.device = device
        # The reference configuration loads and merges in fp16. fp16 matmuls are not
        # usable on CPU, so CPU runs in fp32 (results may differ slightly).
        self.dtype = torch.float32 if device == "cpu" else torch.float16
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model, dtype=torch.float16, low_cpu_mem_usage=True)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(device=device, dtype=self.dtype).eval()
        self.processor = WhisperProcessor.from_pretrained(base_model)
        self.gen = dict(language="ko", task="transcribe",
                        num_beams=num_beams, max_new_tokens=max_new_tokens)
        if keywords:
            self.gen["logits_processor"] = [KeywordBoostProcessor.from_file(
                self.processor.tokenizer, keywords,
                lam=boost_lambda, lam_start=boost_lambda_start)]
        self.info = {
            "base_model": Path(base_model).name,
            "adapter": adapter_name(adapter),
            "keywords": Path(keywords).name if keywords else None,
            "device": device, "dtype": str(self.dtype).replace("torch.", ""),
            "num_beams": num_beams,
        }
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray) -> str:
        feat = self.processor(audio, sampling_rate=SAMPLE_RATE,
                              return_tensors="pt").input_features
        feat = feat.to(device=self.device, dtype=self.dtype)
        with self._lock, torch.no_grad():
            ids = self.model.generate(feat, **self.gen)
        return self.processor.decode(ids[0], skip_special_tokens=True).strip()


def create_app(recognizer: Recognizer) -> FastAPI:
    app = FastAPI(title="MMAI Korean ASR")

    @app.get("/health")
    def health():
        return {"status": "ok", **recognizer.info}

    @app.post("/v1/transcribe")
    async def transcribe(request: Request):
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty body")
        try:
            audio = load_audio(body)
        except Exception as e:  # noqa: BLE001 - report decode errors to the client
            raise HTTPException(400, f"cannot decode audio: {e}")
        duration = len(audio) / SAMPLE_RATE
        if duration > MAX_SECONDS:
            raise HTTPException(413, f"audio longer than {MAX_SECONDS:.0f} s")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0
        if rms < SILENCE_RMS:
            return {"text": "", "silent": True, "duration_sec": duration, "elapsed_sec": 0.0}
        t0 = time.time()
        # generate() blocks; run it in the threadpool so /health stays responsive.
        from starlette.concurrency import run_in_threadpool
        text = await run_in_threadpool(recognizer.transcribe, audio)
        return {"text": text, "silent": False, "duration_sec": duration,
                "elapsed_sec": round(time.time() - t0, 3)}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True, help="voice-phishing LoRA adapter directory")
    ap.add_argument("--keywords", required=True, help="keyword list, one word per line")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = pick_device(args.device)
    t0 = time.time()
    recognizer = Recognizer(args.base_model, args.adapter, args.keywords, device)
    print(f"model loaded in {time.time() - t0:.0f}s: {recognizer.info}", flush=True)
    uvicorn.run(create_app(recognizer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
