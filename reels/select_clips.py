#!/usr/bin/env python3
"""Pick the reels: ask Claude (via the `claude` CLI) which transcript segments make
self-contained, engaging reels. Writes reels.json.

Usage: select_clips.py transcript.json --out reels.json [--reels auto|N] [--target 60]
"""
import argparse, json, os, re, subprocess, sys

PROMPT = """You are an expert short-form video editor who turns university lectures into
Instagram Reels for a student who cannot sit through a whole lecture but learns well from
punchy, podcast-style clips.

Below is a lecture transcript split into numbered segments with timestamps.

Your job: choose {n_reels} reels. Each reel must:
- Teach ONE important idea from the lecture so it is self-contained and makes sense on its own.
- Be {min_s}-{max_s} seconds long in total (sum of its clips). Aim for about {target_s}s.
- Be made of 2-8 clips. A clip is a contiguous range of segment ids (from_id..to_id inclusive).
  Clips are played back-to-back as jump cuts, so cut out the boring bits: admin, "as I said
  before", lost-my-place rambling, repeated sentences, long examples that go nowhere, anything
  that is not needed to understand the idea. Keep the lecturer's best explanations, analogies,
  definitions, "the key point is...", and anything exam-relevant.
- Start on a strong, clear sentence (a hook, question, or claim), never mid-thought.
- End on a complete thought.
- Clips within a reel must be in chronological order and must not overlap other reels.
- Prioritise the ideas a student most needs to know for the exam; keep the whole lecture's
  arc covered across the set of reels where possible, in lecture order.

Return ONLY a JSON object (no markdown fences, no commentary) with this shape:
{{
  "lecture_title": "short title of the lecture",
  "reels": [
    {{
      "title": "Hook title shown on screen, max 7 words, punchy, no clickbait lies",
      "idea": "one sentence: what the viewer will understand after watching",
      "clips": [{{"from_id": 12, "to_id": 15}}, {{"from_id": 19, "to_id": 22}}]
    }}
  ]
}}

TRANSCRIPT (id, start-end seconds, text):
{transcript}
"""


def run_claude(prompt: str, model: str | None) -> str:
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"claude CLI failed: {r.stderr[:2000]}")
    try:
        return json.loads(r.stdout)["result"]
    except Exception:
        return r.stdout


def parse_json(s: str) -> dict:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError("no JSON object in model output:\n" + s[:1000])
    return json.loads(m.group(0))


def validate(reels: dict, segs: list, min_s: float, max_s: float) -> dict:
    by_id = {s["id"]: s for s in segs}
    used = set()
    ok = []
    for r in reels.get("reels", []):
        clips, total = [], 0.0
        good = True
        for c in r.get("clips", []):
            a, b = int(c["from_id"]), int(c["to_id"])
            if a > b or a not in by_id or b not in by_id:
                print(f"[select] reel '{r.get('title')}': bad clip {c}, skipping reel", file=sys.stderr)
                good = False
                break
            ids = set(range(a, b + 1))
            if ids & used:
                print(f"[select] reel '{r.get('title')}': overlaps another reel at ids {sorted(ids & used)[:5]}, trimming", file=sys.stderr)
                ids -= used
                if not ids:
                    continue
                a, b = min(ids), max(ids)
            used |= ids
            clips.append({"from_id": a, "to_id": b})
            total += by_id[b]["end"] - by_id[a]["start"]
        if not good or not clips:
            continue
        if total < min_s * 0.6 or total > max_s * 1.6:
            print(f"[select] reel '{r.get('title')}' is {total:.0f}s, outside range, keeping anyway", file=sys.stderr)
        r["clips"] = clips
        r["est_seconds"] = round(total, 1)
        ok.append(r)
    reels["reels"] = ok
    return reels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reels", default="auto", help="'auto' or a number")
    ap.add_argument("--target", type=float, default=60.0, help="target seconds per reel")
    ap.add_argument("--min", type=float, default=40.0)
    ap.add_argument("--max", type=float, default=90.0)
    ap.add_argument("--model", default=None, help="claude model id (default: CLI default)")
    a = ap.parse_args()

    tr = json.load(open(a.transcript))
    segs = tr["segments"]
    lines = [f"{s['id']}\t{s['start']:.1f}-{s['end']:.1f}\t{s['text']}" for s in segs]
    total_min = (segs[-1]["end"] - segs[0]["start"]) / 60 if segs else 0
    if a.reels == "auto":
        n = max(2, min(10, round(total_min / 8)))  # roughly one reel per 8 lecture-minutes
        n_reels = f"between {n - 1} and {n + 2}"
    else:
        n_reels = f"exactly {int(a.reels)}"
    prompt = PROMPT.format(n_reels=n_reels, min_s=int(a.min), max_s=int(a.max), target_s=int(a.target),
                           transcript="\n".join(lines))
    print(f"[select] asking Claude for {n_reels} reels from {len(segs)} segments ({total_min:.0f} min)")
    out = run_claude(prompt, a.model)
    reels = validate(parse_json(out), segs, a.min, a.max)
    with open(a.out, "w") as f:
        json.dump(reels, f, indent=2, ensure_ascii=False)
    print(f"[select] {len(reels['reels'])} reels -> {a.out}")
    for i, r in enumerate(reels["reels"], 1):
        print(f"  {i}. {r['title']}  (~{r['est_seconds']}s, {len(r['clips'])} clips)")


if __name__ == "__main__":
    main()
