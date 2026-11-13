# Lectures as Reels

Turns a lecture recording into 45–90 s Instagram-style reels: slides on top, lecturer below,
podcast-style word-highlight captions, hook title, progress bar. Jump-cuts out the admin and
rambling. Claude picks the clips; whisper does the transcript; ffmpeg does the editing.

## Setup (Apple Silicon Mac)

    brew install ffmpeg uv
    uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt

Clip selection uses the `claude` CLI (Claude Code), so `claude` must be installed and logged in.

## 1. Get the lecture video out of Panopto

Install the **Get cookies.txt LOCALLY** browser extension. Open the lecture page, logged in,
click the extension, click Export (saves to `~/Downloads`). Then:

    ./download.sh "https://YOUR-PANOPTO-HOST/Panopto/Pages/Viewer.aspx?id=..."

This grabs the camera stream (with audio) and the slides stream as separate files into `lectures/`.
Only download recordings you are entitled to view, and keep the reels for your own study.

### Alternative

On the Panopto page click the Download button (if the lecturer enabled it). You get one mp4,
usually side-by-side or picture-in-picture. The pipeline detects both automatically.

## 2. Make the reels

    .venv/bin/python make_reels.py lectures/LECTURE.mp4 --slides lectures/SLIDES.mp4

or with a single combined file:

    .venv/bin/python make_reels.py lectures/LECTURE.mp4

Output: `out/LECTURE/reel_01_*.mp4`, `reel_02_*.mp4`, ... (1080x1920, H.264, ready to upload).

A 50-minute lecture takes roughly: transcript 3–4 min, clip selection 1 min, rendering about
1 min per reel.

## Useful flags

| flag | what it does |
|---|---|
| `--reels 6` | force a number of reels (default: auto, ~1 per 8 lecture-minutes) |
| `--target 60` | target seconds per reel |
| `--tighten 0.8` | cut any pause longer than 0.8 s (0 = off) |
| `--speed 1.15` | speed everything up |
| `--layout sbs|pip|single` | override layout detection |
| `--speaker-crop x,y,w,h` | where the lecturer is in the frame (pixel rect) |
| `--reselect` | ask Claude for a fresh set of clips |
| `--only 3` | re-render just reel 3 |

## Tweak by hand

`out/LECTURE/reels.json` lists each reel's title and clips as transcript segment-id ranges.
`out/LECTURE/transcript.txt` shows every segment with its id. Edit reels.json, re-run the same
command, and only rendering happens again.

## How it works

1. `reels/transcribe.py` — mlx-whisper (large-v3-turbo) with word timestamps.
2. `reels/layout.py` — finds speaker/slides regions (two files, side-by-side, or PiP inset).
3. `reels/select_clips.py` — sends the transcript to `claude -p`, gets back reels as segment ranges.
4. `reels/render.py` — per cut: face-centred crop of the speaker, slides panel, stack; concat;
   burn ASS captions (current word in yellow) and title; loudness-normalise.
