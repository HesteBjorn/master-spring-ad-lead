#!/usr/bin/env python3
"""Build the 'potential videos' overview from a sweep's kept.jsonl files.

Reads <sweep_root>/town<TOWN>/{tfv6,td3}/kept.jsonl (written live by
monitor_routes.py) and produces:

  * Two TD3-vs-TFv6 side-by-side comparison sets, with stitched mp4s:
      A. TD3 succeeds where TFv6 fails (collision / timeout / ... ; route_dev is
         already deleted so never appears here).
      B. Both succeed but TFv6 takes much longer to finish the route.
  * A catalogue of every kept standalone clip, grouped by category.

All of it is written to the overview markdown between generated markers, so any
manual notes outside those markers are preserved.

Usage:
    python rl_finetuning/eval_viz/build_overview.py \
        --sweep-root outputs/eval_viz/sweep \
        --overview outputs/eval_viz/videos_to_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from datetime import datetime

_BEGIN = "<!-- BEGIN GENERATED: sweep video catalogue -->"
_END = "<!-- END GENERATED: sweep video catalogue -->"


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _by_route(entries: list[dict]) -> dict[str, dict]:
    return {e["route_id"]: e for e in entries}


def stitch_sidebyside(left: dict, right: dict, out_mp4: str, fps: int) -> bool:
    """hstack two route folders' frames (TFv6 left, TD3 right), frozen-pad to equal
    length, with titles. Frame-0 aligned preview; refine offsets with
    stitch_sidebyside.sh."""
    lf = os.path.join(left["folder"], "frame_%06d.jpg")
    rf = os.path.join(right["folder"], "frame_%06d.jpg")
    if left.get("frames", 0) == 0 or right.get("frames", 0) == 0:
        return False
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    def title(t):
        if os.path.exists(font):
            return (
                f"drawtext=fontfile={font}:text='{t}':x=(w-tw)/2:y=12:fontsize=40:"
                f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8,"
            )
        return ""

    fc = (
        f"[0:v]{title('TFv6')}tpad=stop_mode=clone:stop_duration=30[l];"
        f"[1:v]{title('TD3 fine-tuned')}tpad=stop_mode=clone:stop_duration=30[r];"
        f"[l][r]hstack=inputs=2[v]"
    )
    # Trim to the longer of the two so the freeze-pad doesn't add dead air.
    dur = max(left.get("frames", 0), right.get("frames", 0)) / float(fps)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        lf,
        "-framerate",
        str(fps),
        "-i",
        rf,
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-t",
        f"{dur:.3f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-r",
        str(fps),
        out_mp4,
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and os.path.exists(out_mp4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", default="outputs/eval_viz/sweep")
    ap.add_argument("--overview", default="outputs/eval_viz/videos_to_report.md")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument(
        "--slow-gap-s",
        type=float,
        default=4.0,
        help="min TFv6-minus-TD3 game-seconds to count as 'TFv6 slower'.",
    )
    ap.add_argument(
        "--slow-ratio",
        type=float,
        default=1.25,
        help="min TFv6/TD3 duration ratio to count as 'TFv6 slower'.",
    )
    args = ap.parse_args()

    comp_dir = os.path.join(args.sweep_root, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    towns = sorted(
        {
            os.path.basename(os.path.dirname(os.path.dirname(p)))
            for p in glob.glob(
                os.path.join(args.sweep_root, "town*", "*", "kept.jsonl")
            )
        }
    )

    cmp_a: list[dict] = []  # TD3 success, TFv6 fail
    cmp_b: list[dict] = []  # both success, TFv6 slower
    standalone: dict[str, list[dict]] = {}

    for town in towns:
        tfv6 = _by_route(
            _read_jsonl(os.path.join(args.sweep_root, town, "tfv6", "kept.jsonl"))
        )
        td3 = _by_route(
            _read_jsonl(os.path.join(args.sweep_root, town, "td3", "kept.jsonl"))
        )

        # Catalogue every kept standalone clip (both models).
        for entries in (tfv6.values(), td3.values()):
            for e in entries:
                standalone.setdefault(e["category"], []).append({"town": town, **e})

        # Comparisons need the same route in both passes.
        for rid, t in td3.items():
            f = tfv6.get(rid)
            if f is None:
                continue
            if t["is_success"] and not f["is_success"]:
                out = os.path.join(
                    comp_dir, f"{town}_route{rid}_A_td3win_{f['category']}.mp4"
                )
                ok = stitch_sidebyside(f, t, out, args.fps)
                cmp_a.append(
                    {
                        "town": town,
                        "route": rid,
                        "tfv6": f["category"],
                        "td3_dur": t["duration_game"],
                        "tfv6_dur": f["duration_game"],
                        "video": out if ok else None,
                    }
                )
            elif t["is_success"] and f["is_success"]:
                gap = f["duration_game"] - t["duration_game"]
                ratio = f["duration_game"] / max(t["duration_game"], 1e-6)
                if gap >= args.slow_gap_s and ratio >= args.slow_ratio:
                    out = os.path.join(comp_dir, f"{town}_route{rid}_B_tfv6slower.mp4")
                    ok = stitch_sidebyside(f, t, out, args.fps)
                    cmp_b.append(
                        {
                            "town": town,
                            "route": rid,
                            "gap": gap,
                            "ratio": ratio,
                            "td3_dur": t["duration_game"],
                            "tfv6_dur": f["duration_game"],
                            "video": out if ok else None,
                        }
                    )

    cmp_b.sort(key=lambda x: -x["gap"])

    # ---- render markdown ----
    L = []
    L.append(_BEGIN)
    L.append(f"\n## Demo video catalogue (generated {datetime.now():%Y-%m-%d %H:%M})\n")
    L.append(
        f"Sweep root: `{args.sweep_root}` — towns: {', '.join(towns) or 'none yet'}\n"
    )

    L.append("\n### A. TD3 succeeds where TFv6 fails (side-by-side)\n")
    if cmp_a:
        L.append(
            "| town | route | TFv6 outcome | TD3 dur (s) | TFv6 dur (s) | side-by-side |"
        )
        L.append("|---|---|---|---|---|---|")
        for c in cmp_a:
            v = f"`{c['video']}`" if c["video"] else "_(missing frames)_"
            L.append(
                f"| {c['town']} | {c['route']} | {c['tfv6']} | "
                f"{c['td3_dur']:.0f} | {c['tfv6_dur']:.0f} | {v} |"
            )
    else:
        L.append("_none found yet_")

    L.append("\n### B. Both succeed but TFv6 is much slower (side-by-side)\n")
    if cmp_b:
        L.append(
            "| town | route | TD3 dur (s) | TFv6 dur (s) | gap (s) | ratio | side-by-side |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for c in cmp_b:
            v = f"`{c['video']}`" if c["video"] else "_(missing frames)_"
            L.append(
                f"| {c['town']} | {c['route']} | {c['td3_dur']:.0f} | {c['tfv6_dur']:.0f} | "
                f"{c['gap']:.0f} | {c['ratio']:.2f} | {v} |"
            )
    else:
        L.append("_none found yet_")

    L.append("\n### Standalone clips by category\n")
    for cat in sorted(standalone):
        rows = sorted(
            standalone[cat],
            key=lambda e: (
                e["town"],
                e["model"],
                int(e["route_id"]) if e["route_id"].isdigit() else 0,
            ),
        )
        L.append(f"\n**{cat}** ({len(rows)})\n")
        L.append("| town | model | route | len (m) | dur (s) | video |")
        L.append("|---|---|---|---|---|---|")
        for e in rows:
            v = f"`{e['video']}`" if e.get("video") else "_(no video)_"
            L.append(
                f"| {e['town']} | {e['model']} | {e['route_id']} | "
                f"{e['route_length']:.0f} | {e['duration_game']:.0f} | {v} |"
            )
    L.append("\n" + _END)
    generated = "\n".join(L)

    # Preserve manual notes outside the generated markers.
    prior = ""
    if os.path.exists(args.overview):
        with open(args.overview, encoding="utf-8") as fh:
            prior = fh.read()
    if _BEGIN in prior and _END in prior:
        head, rest = prior.split(_BEGIN, 1)
        _, tail = rest.split(_END, 1)
        new = head.rstrip() + "\n\n" + generated + "\n" + tail.lstrip()
    else:
        new = (prior.rstrip() + "\n\n" if prior.strip() else "") + generated + "\n"
    os.makedirs(os.path.dirname(args.overview) or ".", exist_ok=True)
    with open(args.overview, "w", encoding="utf-8") as fh:
        fh.write(new)

    print(
        f"[overview] wrote {args.overview}: {len(cmp_a)} A-comparisons, "
        f"{len(cmp_b)} B-comparisons, "
        f"{sum(len(v) for v in standalone.values())} standalone clips across towns {towns}"
    )


if __name__ == "__main__":
    main()
