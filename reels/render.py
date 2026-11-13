#!/usr/bin/env python3
"""Render 9:16 reels: slides on top, speaker below, word-highlight captions, hook title.

Usage: render.py --layout layout.json --transcript transcript.json --reels reels.json --out outdir
"""
import argparse, json, os, re, shutil, subprocess, sys
import numpy as np

W, H = 1080, 1920
TOP = 150                 # title band height
SLIDE_H = 608             # 16:9 slide panel (1080x608)
SPK_Y = TOP + SLIDE_H     # speaker panel starts here
SPK_H = H - SPK_Y         # 1162
BG = "0x0e0e10"
FPS = 30
FONT = "Avenir Next Heavy"   # upright heavy weight; "Arial Black" also works
PAD_BEFORE, PAD_AFTER = 0.12, 0.30


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit("ffmpeg failed:\n" + " ".join(cmd) + "\n" + r.stderr[-3000:])
    return r


# ---------- cut list -------------------------------------------------------

def build_cuts(reel, segs, tighten_gap, dur):
    """Return list of (start, end) in source seconds, plus the words inside them."""
    by_id = {s["id"]: s for s in segs}
    cuts = []
    for c in reel["clips"]:
        words = [w for i in range(c["from_id"], c["to_id"] + 1) for w in by_id[i]["words"]]
        if not words:
            continue
        if tighten_gap:
            run = [words[0]]
            for prev, w in zip(words, words[1:]):
                if w["s"] - prev["e"] > tighten_gap:
                    cuts.append((run[0]["s"] - PAD_BEFORE, run[-1]["e"] + PAD_AFTER))
                    run = [w]
                else:
                    run.append(w)
            cuts.append((run[0]["s"] - PAD_BEFORE, run[-1]["e"] + PAD_AFTER))
        else:
            cuts.append((words[0]["s"] - PAD_BEFORE, words[-1]["e"] + PAD_AFTER))
    # clamp + merge overlaps/tiny gaps
    cuts = [(max(0.0, s), min(dur, e)) for s, e in cuts if e > s]
    merged = []
    for s, e in sorted(cuts):
        if merged and s <= merged[-1][1] + 0.25:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if e - s > 0.4]


# ---------- speaker locating -----------------------------------------------
# The lecture camera is static and wide, so the lecturer is (a) the thing that moves and
# (b) the biggest upper-body detection. We zoom a window around that position.

_ub = None
ZOOM_MIN_H, ZOOM_MAX_H = 650, 1080   # crop window height bounds (source px)
GRADE = "eq=contrast=1.25:brightness=-0.03:saturation=1.35:gamma=0.97,unsharp=5:5:0.7"


def _frames(path, crop, t0, t1, n, width):
    import cv2
    vf = (f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else "") + f"scale={width}:-2"
    out = []
    for i in range(n):
        t = t0 + (t1 - t0) * (i + 0.5) / n
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-vf", vf,
                            "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
        img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    return out


def locate_speaker(path, crop, t0, t1, n=8):
    """Return (cx, cy, body_w) normalised 0..1 inside the speaker crop, or None."""
    global _ub
    import cv2
    if _ub is None:
        _ub = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
    frames = _frames(path, crop, t0, t1, n, 960)
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    # upper-body boxes across frames; keep the largest per frame, take the median
    boxes = []
    for img in frames[:: max(1, len(frames) // 4)]:
        g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        det = _ub.detectMultiScale(g, 1.05, 3, minSize=(int(w * 0.08), int(w * 0.08)))
        if len(det):
            boxes.append(max(det, key=lambda b: b[2] * b[3]))
    if boxes:
        b = np.median(np.array(boxes), axis=0)
        return ((b[0] + b[2] / 2) / w, (b[1] + b[3] * 0.3) / h, b[2] / w)
    # motion centroid fallback
    if len(frames) >= 2:
        g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
        act = np.abs(np.diff(g, axis=0)).mean(axis=0)
        act = cv2.GaussianBlur(act, (0, 0), 6)
        mask = act > max(act.mean() * 4, np.percentile(act, 98))
        if mask.sum() > 50:
            ys, xs = np.nonzero(mask)
            return (float(np.median(xs)) / w, float(np.median(ys)) / h, 0.18)
    return None


def speaker_chain(lay, loc):
    """Filter chain: zoomed window around the lecturer -> 1080xSPK_H panel, colour graded."""
    crop = lay["speaker"]["crop"]
    sw, sh_ = (crop[2], crop[3]) if crop else (lay["speaker"]["w"], lay["speaker"]["h"])
    chain = f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else ""
    cx, cy, bw = loc if loc else (0.5, 0.45, 0.2)
    aspect = W / SPK_H
    win_w = int(bw * sw * 2.2)                       # body ~45% of the panel width
    win_h = int(min(sh_, min(ZOOM_MAX_H, max(ZOOM_MIN_H, win_w / aspect))))
    win_w = int(win_h * aspect)
    if win_w > sw:
        win_w, win_h = sw, int(sw / aspect)
    x = int(max(0, min(sw - win_w, cx * sw - win_w / 2)))
    y = int(max(0, min(sh_ - win_h, cy * sh_ - win_h / 2)))
    chain += f"crop={win_w}:{win_h}:{x}:{y},scale={W}:{SPK_H}:flags=lanczos,{GRADE},setsar=1"
    return chain


def slides_chain(lay):
    crop = lay["slides"]["crop"]
    chain = f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else ""
    mask = lay["slides"].get("mask")
    if mask:  # cover the camera inset on the slides panel
        chain += f"drawbox=x={mask[0]}:y={mask[1]}:w={mask[2]}:h={mask[3]}:color={BG}:t=fill,"
    chain += f"scale={W}:{SLIDE_H}:force_original_aspect_ratio=decrease,pad={W}:{SLIDE_H}:-1:-1:color={BG},setsar=1"
    return chain


def render_cut(lay, s, e, loc, out, audio_file):
    spk, sld = lay["speaker"]["file"], lay["slides"]["file"]
    inputs = ["-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", spk]
    sld_idx = 0
    if sld != spk:
        inputs += ["-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", sld]
        sld_idx = 1
    aud_idx = 0 if audio_file == spk else (sld_idx if audio_file == sld else None)
    if aud_idx is None:
        inputs += ["-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", audio_file]
        aud_idx = 2 if sld != spk else 1
    fc = (f"[0:v]{speaker_chain(lay, loc)},fps={FPS}[spk];"
          f"[{sld_idx}:v]{slides_chain(lay)},fps={FPS}[sld];"
          f"color=c={BG}:s={W}x{H}:r={FPS}:d={e - s:.3f}[bg];"
          f"[bg][sld]overlay=0:{TOP}:shortest=1[t1];"
          f"[t1][spk]overlay=0:{SPK_Y}:shortest=1[v];"
          f"[{aud_idx}:a]afade=t=in:d=0.04,afade=t=out:st={max(0, e - s - 0.06):.3f}:d=0.06,aresample=48000[a]")
    sh(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
        "-c:v", "h264_videotoolbox", "-b:v", "14M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000", "-ac", "2", "-movflags", "+faststart", out])


# ---------- captions -------------------------------------------------------

def ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def esc(s):
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def chunk_words(words, max_words=4, max_chars=24, gap=0.7):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(x["w"] for x in cur + [w])) > max_chars
                    or w["t0"] - cur[-1]["t1"] > gap or re.search(r"[.!?]$", cur[-1]["w"])):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def write_ass(path, words, title, part, nparts, total):
    hdr = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{FONT},72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,5,2,2,40,40,380,1
Style: Title,{FONT},58,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,8,40,40,34,1
Style: Part,{FONT},30,&H00C8C8C8,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,2,0,1,1,0,8,40,40,106,1
Style: Bar,{FONT},20,&H00FFE000,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    ev.append(f"Dialogue: 0,{ass_time(0)},{ass_time(total)},Title,,0,0,0,,{esc(title.upper())}")
    ev.append(f"Dialogue: 0,{ass_time(0)},{ass_time(total)},Part,,0,0,0,,PART {part} OF {nparts}")
    # progress bar along the very top edge
    steps = max(1, int(total * 4))
    for i in range(steps):
        t0, t1 = total * i / steps, total * (i + 1) / steps
        wpx = int(W * (i + 1) / steps)
        ev.append(f"Dialogue: 1,{ass_time(t0)},{ass_time(t1)},Bar,,0,0,0,,{{\\p1\\pos(0,0)\\bord0\\shad0}}m 0 0 l {wpx} 0 {wpx} 10 0 10{{\\p0}}")
    for ch in chunk_words(words):
        for i, w in enumerate(ch):
            t0 = w["t0"]
            t1 = ch[i + 1]["t0"] if i + 1 < len(ch) else ch[-1]["t1"] + 0.08
            parts = []
            for j, x in enumerate(ch):
                txt = esc(x["w"])
                parts.append(f"{{\\c&H00E5FF&\\fscx108\\fscy108}}{txt}{{\\r}}" if j == i else txt)
            ev.append(f"Dialogue: 2,{ass_time(t0)},{ass_time(t1)},Cap,,0,0,0,,{' '.join(parts)}")
    with open(path, "w") as f:
        f.write(hdr + "\n".join(ev) + "\n")


# ---------- main -----------------------------------------------------------

def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--reels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tighten", type=float, default=0.8, help="cut pauses longer than this (s); 0=off")
    ap.add_argument("--only", type=int, default=None, help="render only this reel number (1-based)")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    lay = json.load(open(a.layout))
    segs = json.load(open(a.transcript))["segments"]
    reels = json.load(open(a.reels))
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "work")
    os.makedirs(work, exist_ok=True)
    audio_file = lay["speaker"]["file"] if lay["speaker"]["has_audio"] else lay["slides"]["file"]
    dur = min(lay["speaker"]["dur"], lay["slides"]["dur"])
    n = len(reels["reels"])
    outputs = []
    default_loc = locate_speaker(lay["speaker"]["file"], lay["speaker"]["crop"], 0, dur, n=12)
    print(f"[render] typical speaker position: {default_loc}")

    for idx, reel in enumerate(reels["reels"], 1):
        if a.only and idx != a.only:
            continue
        cuts = build_cuts(reel, segs, a.tighten, dur)
        print(f"[render] reel {idx}/{n} '{reel['title']}': {len(cuts)} cuts, {sum(e - s for s, e in cuts):.1f}s")
        parts, t_acc, words = [], 0.0, []
        for k, (s, e) in enumerate(cuts):
            loc = locate_speaker(lay["speaker"]["file"], lay["speaker"]["crop"], s, e) or default_loc
            part = os.path.join(work, f"r{idx:02d}_c{k:03d}.mp4")
            render_cut(lay, s, e, loc, part, audio_file)
            parts.append(part)
            for seg in segs:
                for w in seg["words"]:
                    if w["s"] >= s - 0.05 and w["e"] <= e + 0.05:
                        words.append({"w": w["w"], "t0": t_acc + max(0, w["s"] - s), "t1": t_acc + min(e - s, w["e"] - s)})
            t_acc += e - s
        lst = os.path.join(work, f"r{idx:02d}.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
        raw = os.path.join(work, f"r{idx:02d}_raw.mp4")
        sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", raw])

        if a.speed != 1.0:
            for w in words:
                w["t0"] /= a.speed
                w["t1"] /= a.speed
            t_acc /= a.speed
        ass = os.path.join(work, f"r{idx:02d}.ass")
        write_ass(ass, words, reel["title"], idx, n, t_acc)
        final = os.path.join(a.out, f"reel_{idx:02d}_{slug(reel['title'])}.mp4")
        vf = f"ass={ass}"
        af = "loudnorm=I=-16:TP=-1.5:LRA=11"
        if a.speed != 1.0:
            vf = f"setpts=PTS/{a.speed},{vf}"
            af = f"atempo={a.speed},{af}"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-vf", vf, "-af", af,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-r", str(FPS), "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", final])
        outputs.append(final)
        print(f"[render]   -> {final} ({t_acc:.0f}s)")
    return outputs


if __name__ == "__main__":
    main()
