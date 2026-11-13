#!/usr/bin/env python3
"""Lecture -> Instagram reels, end to end.

  python make_reels.py LECTURE.mp4 [--slides SLIDES.mp4] [--out out/NAME] [--reels auto|N]
                        [--layout auto|sbs|pip|single] [--speaker-crop x,y,w,h] [--slides-crop x,y,w,h]
                        [--tighten 0.8] [--speed 1.0] [--reselect] [--only N]

Steps (each cached in the out dir): transcribe -> detect layout -> pick clips (Claude) -> render.
Edit out/NAME/reels.json by hand and re-run to re-render with your own choices.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(args):
    print("+", " ".join(os.path.basename(x) if i == 0 else x for i, x in enumerate(args)), flush=True)
    subprocess.run(args, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--slides", default=None, help="separate slides/screen recording, if you have one")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reels", default="auto")
    ap.add_argument("--layout", default="auto", choices=["auto", "sbs", "pip", "single"])
    ap.add_argument("--speaker-crop", default=None)
    ap.add_argument("--slides-crop", default=None)
    ap.add_argument("--tighten", type=float, default=0.8)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--target", type=float, default=60)
    ap.add_argument("--whisper-model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--claude-model", default=None)
    ap.add_argument("--reselect", action="store_true", help="ignore cached reels.json")
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args()

    video = os.path.abspath(a.video)
    name = os.path.splitext(os.path.basename(video))[0]
    out = os.path.abspath(a.out or os.path.join(HERE, "out", name))
    os.makedirs(out, exist_ok=True)
    transcript = os.path.join(out, "transcript.json")
    layout = os.path.join(out, "layout.json")
    reels = os.path.join(out, "reels.json")

    # 1. transcript
    if not os.path.exists(transcript):
        run([PY, os.path.join(HERE, "reels", "transcribe.py"), video, "--out", transcript, "--model", a.whisper_model])
    else:
        print(f"[cache] {transcript}")

    # 2. layout
    if not os.path.exists(layout):
        sys.path.insert(0, os.path.join(HERE, "reels"))
        from layout import detect
        lay = detect(video, os.path.abspath(a.slides) if a.slides else None, a.layout)
        if a.speaker_crop:
            lay["speaker"]["crop"] = [int(x) for x in a.speaker_crop.split(",")]
        if a.slides_crop:
            lay["slides"]["crop"] = [int(x) for x in a.slides_crop.split(",")]
        json.dump(lay, open(layout, "w"), indent=2)
        print(f"[layout] mode={lay['mode']} speaker.crop={lay['speaker']['crop']} slides.crop={lay['slides']['crop']}")
    else:
        print(f"[cache] {layout}")

    # 3. clip selection
    if a.reselect or not os.path.exists(reels):
        cmd = [PY, os.path.join(HERE, "reels", "select_clips.py"), transcript, "--out", reels,
               "--reels", a.reels, "--target", str(a.target)]
        if a.claude_model:
            cmd += ["--model", a.claude_model]
        run(cmd)
    else:
        print(f"[cache] {reels}")

    # 4. render
    cmd = [PY, os.path.join(HERE, "reels", "render.py"), "--layout", layout, "--transcript", transcript,
           "--reels", reels, "--out", out, "--tighten", str(a.tighten), "--speed", str(a.speed)]
    if a.only:
        cmd += ["--only", str(a.only)]
    run(cmd)
    print(f"\nDone. Reels are in: {out}")


if __name__ == "__main__":
    main()
