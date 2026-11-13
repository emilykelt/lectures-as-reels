#!/usr/bin/env python3
"""Work out where the speaker and the slides are.

Returns a dict: {"speaker": {"file", "crop": [x,y,w,h] or None}, "slides": {...}, "mode": ...}
Modes: two-files | sbs (side-by-side in one file) | pip (inset speaker) | single (no slides found)
"""
import json, subprocess, sys
import numpy as np


def probe(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                       capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    num, den = (v["r_frame_rate"].split("/") + ["1"])[:2]
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": float(num) / float(den),
            "dur": float(info["format"]["duration"]), "has_audio": a is not None}


def sample_frames(path: str, n: int = 16, scale_w: int = 320):
    import cv2
    p = probe(path)
    frames = []
    for i in range(n):
        t = p["dur"] * (i + 0.5) / n
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
                            "-vf", f"scale={scale_w}:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                           capture_output=True, check=True)
        arr = np.frombuffer(r.stdout, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img.astype(np.float32))
    return np.stack(frames), p


def temporal_activity(frames: np.ndarray) -> np.ndarray:
    """Per-pixel mean abs difference between consecutive sampled frames (grey)."""
    g = frames.mean(axis=3)
    return np.abs(np.diff(g, axis=0)).mean(axis=0)


def detect(speaker_file: str, slides_file: str | None, layout: str = "auto") -> dict:
    if slides_file:
        ps, pl = probe(speaker_file), probe(slides_file)
        return {"mode": "two-files",
                "speaker": {"file": speaker_file, "crop": None, **ps},
                "slides": {"file": slides_file, "crop": None, **pl}}

    frames, p = sample_frames(speaker_file)
    H, W = frames.shape[1:3]
    sx, sy = p["w"] / W, p["h"] / H
    act = temporal_activity(frames)

    if layout == "sbs" or (layout == "auto" and p["w"] / p["h"] > 2.4):
        left, right = act[:, : W // 2].mean(), act[:, W // 2:].mean()
        spk_left = left > right
        half = p["w"] // 2
        spk = [0, 0, half, p["h"]] if spk_left else [half, 0, half, p["h"]]
        sld = [half, 0, half, p["h"]] if spk_left else [0, 0, half, p["h"]]
        return {"mode": "sbs", "speaker": {"file": speaker_file, "crop": spk, **p},
                "slides": {"file": speaker_file, "crop": sld, **p}}

    if layout in ("auto", "pip"):
        # Look for a compact high-activity rectangle (camera inset) inside a mostly static frame.
        thr = max(act.mean() * 3, np.percentile(act, 90))
        mask = (act > thr).astype(np.uint8)
        import cv2
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        best = None
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            frac = (w * h) / (W * H)
            if 0.03 < frac < 0.45 and w > 20 and h > 20 and area / (w * h) > 0.35:
                if best is None or w * h > best[2] * best[3]:
                    best = (x, y, w, h)
        if best is not None:
            x, y, w, h = best
            spk = [int(x * sx), int(y * sy), int(w * sx), int(h * sy)]
            return {"mode": "pip", "speaker": {"file": speaker_file, "crop": spk, **p},
                    "slides": {"file": speaker_file, "crop": None, "mask": spk, **p}}
        if layout == "pip":
            print("[layout] could not find an inset; falling back to single", file=sys.stderr)

    # single stream: show it as the 'slides' panel and crop a speaker-ish region from it too
    return {"mode": "single", "speaker": {"file": speaker_file, "crop": None, **p},
            "slides": {"file": speaker_file, "crop": None, **p}}


if __name__ == "__main__":
    print(json.dumps(detect(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None), indent=2))
