#!/usr/bin/env python3
"""Transcribe a lecture with word-level timestamps (mlx-whisper, Apple Silicon).

Usage: transcribe.py <video_or_audio> --out transcript.json [--model REPO]
Writes transcript.json (segments + words) and transcript.txt (readable, with ids).
"""
import argparse, json, os, subprocess, sys, tempfile, time

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def extract_audio(src: str, dst: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", dst],
        check=True,
    )


def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--language", default="en")
    a = ap.parse_args()

    import mlx_whisper  # slow import; keep local

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio.wav")
        print(f"[transcribe] extracting audio from {a.src}")
        extract_audio(a.src, wav)
        t0 = time.time()
        print(f"[transcribe] running {a.model} (first run downloads the model)")
        res = mlx_whisper.transcribe(
            wav,
            path_or_hf_repo=a.model,
            language=a.language,
            word_timestamps=True,
            condition_on_previous_text=False,
            verbose=False,
        )
        print(f"[transcribe] done in {time.time() - t0:.0f}s")

    segs = []
    for i, s in enumerate(res["segments"]):
        words = [
            {"w": w["word"].strip(), "s": round(float(w["start"]), 3), "e": round(float(w["end"]), 3)}
            for w in s.get("words", [])
            if w["word"].strip()
        ]
        if not words:
            continue
        segs.append({
            "id": len(segs),
            "start": round(float(s["start"]), 3),
            "end": round(float(s["end"]), 3),
            "text": s["text"].strip(),
            "words": words,
        })

    out = {"source": os.path.abspath(a.src), "model": a.model, "language": res.get("language"), "segments": segs}
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w") as f:
        for s in segs:
            f.write(f"[{s['id']:4d}] {fmt(s['start'])} - {fmt(s['end'])}  {s['text']}\n")
    print(f"[transcribe] {len(segs)} segments -> {a.out}, {txt}")


if __name__ == "__main__":
    main()
