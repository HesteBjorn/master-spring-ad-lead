#!/usr/bin/env python3
"""Live monitor for a multi-route video sweep (one model pass over one town).

Runs alongside the leaderboard. The leaderboard writes the checkpoint JSON after
every route and the agent renders that route's frames into
``<clip_root>/route_<NNNN>/`` (NNNN = setup order = record order). This monitor
polls the checkpoint and, as each route completes:

  * route deviation (route_dev) -> delete the route folder immediately.
  * anything else (success / isolated infraction / timeout / combo) -> stitch the
    frames into an mp4 and append a line to <out_jsonl> for the overview builder.

Exits once <done_file> exists and every record has been processed.

Usage (the orchestrator wires this up):
    python rl_finetuning/eval_viz/monitor_routes.py \
        --checkpoint <dir>/checkpoint_endpoint.json --clip-root <dir>/clip_viz \
        --model tfv6 --town 12 --out-jsonl <dir>/kept.jsonl \
        --done-file <dir>/.run_done --fps 20
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

from rl_finetuning.eval_viz.video_classify import DELETE, classify


def _load_records(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return []
    recs = data.get("_checkpoint", {}).get("records")
    return recs if isinstance(recs, list) else []


def _frames(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(
        1 for f in os.listdir(folder) if f.startswith("frame_") and f.endswith(".jpg")
    )


def _stitch(folder: str, out_mp4: str, fps: int) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(folder, "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        out_mp4,
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and os.path.exists(out_mp4)


def process_record(args, idx: int, record: dict) -> dict | None:
    """Classify record idx, act on its route folder, return a kept entry or None."""
    info = classify(record)
    folder = os.path.join(args.clip_root, f"route_{idx:04d}")
    tag = f"r{idx:04d}_route{info['route_id']}"

    if info["category"] == DELETE:
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
        print(f"[monitor] {tag} {info['status']} -> DELETE (route_dev)", flush=True)
        return None

    n = _frames(folder)
    video = None
    if n > 0:
        out_mp4 = os.path.join(
            folder,
            f"t{args.town}_{args.model}_{info['category']}_route{info['route_id']}.mp4",
        )
        if _stitch(folder, out_mp4, args.fps):
            video = out_mp4
    else:
        print(f"[monitor] {tag} has no frames (folder={folder})", flush=True)

    entry = {
        "model": args.model,
        "index": idx,
        "folder": folder,
        "video": video,
        "frames": n,
        **info,
    }
    with open(args.out_jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(
        f"[monitor] {tag} -> KEEP {info['category']} "
        f"(len={info['route_length']:.0f}m dur={info['duration_game']:.0f}s frames={n})",
        flush=True,
    )
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clip-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--town", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--done-file", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--poll", type=float, default=4.0)
    args = ap.parse_args()

    # Fresh pass: start the kept log empty.
    open(args.out_jsonl, "w", encoding="utf-8").close()
    processed = 0
    print(
        f"[monitor] watching {args.checkpoint} (model={args.model} town={args.town})",
        flush=True,
    )

    while True:
        records = _load_records(args.checkpoint)
        while processed < len(records):
            try:
                process_record(args, processed, records[processed])
            except Exception as exc:  # noqa: BLE001 — never let the monitor die mid-sweep
                print(f"[monitor] error on record {processed}: {exc}", flush=True)
            processed += 1
        done = os.path.exists(args.done_file)
        if done and processed >= len(_load_records(args.checkpoint)):
            break
        time.sleep(args.poll)

    print(
        f"[monitor] done: processed {processed} route(s) for {args.model} town {args.town}",
        flush=True,
    )


if __name__ == "__main__":
    main()
