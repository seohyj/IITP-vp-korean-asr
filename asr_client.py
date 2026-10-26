"""Call-recording client for asr_server.py.

Splits a call recording into short chunks the way an on-device call recorder delivers them,
sends each chunk to the ASR server in order, and assembles the call transcript.

  --split fixed  (default) fixed-length chunks, as recorded on the device (5 s)
  --split pause  cut at pauses so that words are not split

Optionally compares the transcript with a reference text (character error rate, ignoring
spaces and punctuation).

Example:
    python asr_client.py call.m4a --url http://127.0.0.1:8000 --out call_transcript.json
"""
import argparse
import io
import json
import re
import time
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf

SEND_SR = 16000


def split_fixed(x: np.ndarray, sr: int, sec: float) -> list:
    n = int(sec * sr)
    return [(i, x[i:i + n]) for i in range(0, len(x), n)]


def split_pause(x: np.ndarray, sr: int, sec: float, top_db: float) -> list:
    """Groups voiced intervals into chunks of at most `sec` seconds."""
    max_len = int(sec * sr)
    chunks, start, end = [], None, None
    for s, e in librosa.effects.split(x, top_db=top_db):
        if start is None:
            start, end = s, e
        elif e - start <= max_len:
            end = e
        else:
            chunks.append((start, end))
            start, end = s, e
    if start is not None:
        chunks.append((start, end))
    out = []
    for s, e in chunks:
        out += [(s + o, x[s + o:min(s + o + max_len, e)]) for o in range(0, e - s, max_len)]
    return out


def to_wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def cer(ref: str, hyp: str) -> float:
    """Character error rate ignoring spaces and punctuation."""
    ref, hyp = re.sub(r"[^\w]", "", ref), re.sub(r"[^\w]", "", hyp)
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(len(ref), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="call recording (wav/flac/m4a/mp3...; m4a/mp3 need ffmpeg)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--split", choices=["fixed", "pause"], default="fixed")
    ap.add_argument("--chunk-sec", type=float, default=5.0)
    ap.add_argument("--top-db", type=float, default=35.0, help="silence threshold for --split pause")
    ap.add_argument("--ref", type=Path, help="reference transcript (text file) for CER")
    ap.add_argument("--out", type=Path, help="write chunk results and transcript as JSON")
    args = ap.parse_args()

    health = requests.get(f"{args.url}/health", timeout=10).json()
    print(f"server: {health}")

    x, _ = librosa.load(args.audio, sr=SEND_SR, mono=True)
    chunks = (split_fixed(x, SEND_SR, args.chunk_sec) if args.split == "fixed"
              else split_pause(x, SEND_SR, args.chunk_sec, args.top_db))

    results, t0 = [], time.time()
    for n, (start, chunk) in enumerate(chunks, 1):
        r = requests.post(f"{args.url}/v1/transcribe", data=to_wav_bytes(chunk, SEND_SR),
                          headers={"Content-Type": "audio/wav"}, timeout=600)
        r.raise_for_status()
        res = r.json()
        results.append({"chunk": n, "start_sec": round(start / SEND_SR, 2),
                        "duration_sec": round(len(chunk) / SEND_SR, 2), **res})
        mark = "(silence)" if res["silent"] else res["text"]
        print(f"[{n:02d}] {start / SEND_SR:6.1f}s  {mark}")

    transcript = " ".join(r["text"] for r in results if r["text"])
    print(f"\ntranscript ({len(chunks)} chunks, {time.time() - t0:.1f} s):\n{transcript}")

    report = {"audio": str(args.audio), "server": health, "chunks": results, "transcript": transcript}
    if args.ref:
        ref = args.ref.read_text(encoding="utf-8").strip()
        report["cer"] = round(cer(ref, transcript), 4)
        print(f"CER (spaces/punctuation ignored): {report['cer']:.2%}")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
