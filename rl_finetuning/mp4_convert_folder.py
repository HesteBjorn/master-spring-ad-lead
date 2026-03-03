#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

FILENAME_RE = re.compile(
    r"^u(?P<update>\d+)_gs(?P<gs>\d+)_step(?P<step>\d+)_env(?P<env>\d+)\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Frame:
    path: Path
    env: int
    update: int
    gs: int
    step: int


@dataclass(frozen=True)
class Recording:
    env: int
    update: int
    frames: list[Frame]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def gs_start(self) -> int:
        return min(f.gs for f in self.frames)

    @property
    def gs_end(self) -> int:
        return max(f.gs for f in self.frames)


def round_to_k(gs: int) -> int:
    return int((gs + 500) // 1000)


def discover_frames(input_dir: Path) -> list[Frame]:
    frames: list[Frame] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        m = FILENAME_RE.match(path.name)
        if m is None:
            continue
        frames.append(
            Frame(
                path=path,
                env=int(m.group("env")),
                update=int(m.group("update")),
                gs=int(m.group("gs")),
                step=int(m.group("step")),
            )
        )
    return frames


def build_recordings(frames: list[Frame]) -> dict[int, list[Recording]]:
    by_env_update: dict[int, dict[int, list[Frame]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for frame in frames:
        by_env_update[frame.env][frame.update].append(frame)

    recordings_by_env: dict[int, list[Recording]] = {}
    for env, by_update in by_env_update.items():
        recordings: list[Recording] = []
        for update in sorted(by_update.keys()):
            rec_frames = sorted(
                by_update[update], key=lambda f: (f.step, f.gs, f.path.name)
            )
            recordings.append(Recording(env=env, update=update, frames=rec_frames))
        recordings_by_env[env] = recordings
    return recordings_by_env


def batch_recordings(
    recordings: list[Recording], target_frames: int
) -> list[list[Recording]]:
    batches: list[list[Recording]] = []
    current: list[Recording] = []
    current_len = 0

    for rec in recordings:
        rec_len = rec.frame_count
        if not current:
            current = [rec]
            current_len = rec_len
            continue

        if current_len + rec_len > target_frames:
            batches.append(current)
            current = [rec]
            current_len = rec_len
            continue

        current.append(rec)
        current_len += rec_len

    if current:
        batches.append(current)
    return batches


def make_output_stem(run_dir: Path) -> str:
    run_name = run_dir.name
    m = re.match(r"^run_(\d{8}_\d{6})$", run_name)
    if m:
        return f"debug_run_{m.group(1)}"
    return f"debug_{run_name}"


def create_video_from_frames(
    frames: list[Frame], output_path: Path, fps: int, crf: int, preset: str
) -> None:
    with tempfile.TemporaryDirectory(prefix="m4p_seq_") as tmp_dir:
        seq_dir = Path(tmp_dir)
        for idx, frame in enumerate(frames):
            seq_name = f"frame_{idx:06d}.jpg"
            os.symlink(frame.path.resolve(), seq_dir / seq_name)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            str(seq_dir / "frame_%06d.jpg"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=pc:out_range=tv,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert debug_viz frames into env-separated MP4 clips with GS ranges."
    )
    parser.add_argument("debug_viz_folder", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--target-seconds", type=int, default=60)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="medium")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.debug_viz_folder.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Error: folder does not exist: {input_dir}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("Error: ffmpeg is not available in PATH.")
    if args.fps <= 0:
        raise SystemExit("Error: --fps must be > 0")
    if args.target_seconds <= 0:
        raise SystemExit("Error: --target-seconds must be > 0")

    frames = discover_frames(input_dir)
    if not frames:
        raise SystemExit(f"Error: no matching debug_viz images found in: {input_dir}")

    run_dir = input_dir.parent
    output_stem = make_output_stem(run_dir)
    target_frames = args.fps * args.target_seconds

    recordings_by_env = build_recordings(frames)
    written_outputs: list[Path] = []
    used_frame_paths: list[Path] = []
    name_counts: dict[str, int] = defaultdict(int)

    for env in sorted(recordings_by_env.keys()):
        env_recordings = recordings_by_env[env]
        batches = batch_recordings(env_recordings, target_frames=target_frames)

        for batch in batches:
            batch_frames = [f for rec in batch for f in rec.frames]
            gs_start = min(rec.gs_start for rec in batch)
            gs_end = max(rec.gs_end for rec in batch)

            start_k = round_to_k(gs_start)
            end_k = round_to_k(gs_end)
            base_name = f"{output_stem}_gs{start_k}kto{end_k}k_env{env}"
            name_counts[base_name] += 1
            if name_counts[base_name] == 1:
                output_name = f"{base_name}.mp4"
            else:
                output_name = f"{output_stem}_gs{start_k}kto{end_k}k_part{name_counts[base_name]}_env{env}.mp4"
            output_path = run_dir / output_name

            create_video_from_frames(
                batch_frames,
                output_path=output_path,
                fps=args.fps,
                crf=args.crf,
                preset=args.preset,
            )
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise SystemExit(f"Error: output video missing or empty: {output_path}")

            # Delete consumed frames after successful write.
            for frame in batch_frames:
                try:
                    frame.path.unlink()
                except FileNotFoundError:
                    pass
            used_frame_paths.extend(frame.path for frame in batch_frames)
            written_outputs.append(output_path)
            print(f"Wrote: {output_path}")

    # Remove input folder if empty after deleting used frames.
    try:
        next(input_dir.iterdir())
    except StopIteration:
        input_dir.rmdir()
        print(f"Deleted source folder: {input_dir}")
    else:
        print(f"Kept non-empty source folder: {input_dir}")

    print(
        "Frames: "
        f"{len(used_frame_paths)} | FPS: {args.fps} | Target seconds: {args.target_seconds} | Videos: {len(written_outputs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
