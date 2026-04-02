from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2

from test_match_icons import (
    DetectionState,
    build_template_set,
    draw_match,
    format_video_time,
    get_video_fps,
    load_template,
    run_match,
    update_detection_state,
)
from tracking_service.config import WATCHBOT_ROOT, ensure_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a long video or YouTube stream for match-start icons and save each match as a fixed clip. "
            "A clip begins when the icon detector confirms a match start and ends a fixed number of seconds later."
        )
    )
    parser.add_argument("--template-a", required=True, help="Path to the first icon image.")
    parser.add_argument("--template-b", required=True, help="Path to the second icon image.")
    parser.add_argument(
        "--video",
        required=True,
        help="Video path, YouTube URL, direct stream URL, or camera index.",
    )
    parser.add_argument(
        "--output-dir",
        default=str((WATCHBOT_ROOT / "clips").resolve()),
        help="Directory where clips will be written. Default: backend/data/watchbot/clips",
    )
    parser.add_argument(
        "--match-prefix",
        default="match",
        help="Filename prefix for saved clips. Default: match",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help="Optional explicit starting match number. If omitted, the next number is inferred from the output directory.",
    )
    parser.add_argument(
        "--index-digits",
        type=int,
        default=3,
        help="Zero-padding width for match numbers. Default: 3",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=160.0,
        help="Clip duration in seconds after a match start. Default: 160 (2m40s)",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Optional region of interest for icon detection.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Score threshold used to stay active once a match has started. Default: 0.55",
    )
    parser.add_argument(
        "--start-threshold",
        type=float,
        default=0.68,
        help="Higher score threshold required to start a match. Default: 0.68",
    )
    parser.add_argument(
        "--start-frames",
        type=int,
        default=3,
        help="Analyzed frames with either icon visible before declaring a match start. Default: 3",
    )
    parser.add_argument(
        "--stop-frames",
        type=int,
        default=20,
        help="Analyzed frames with neither icon visible before declaring a match end. Default: 20",
    )
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=6.0,
        help="How many frames per second to analyze for icon detection. Default: 6.0",
    )
    parser.add_argument(
        "--search-scale",
        type=float,
        default=1.0,
        help="Scale factor used for template matching. Lower is faster but less exact. Default: 1.0",
    )
    parser.add_argument(
        "--seek-seconds",
        type=float,
        help="Optional initial seek offset in seconds. If omitted, `t=` from a YouTube URL is used when present.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        help="Optional limit for how many clips to save before exiting.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a debug preview window while scanning.",
    )
    return parser.parse_args()


def normalize_video_arg(video_arg: str) -> str:
    return (
        video_arg.strip()
        .replace("\\?", "?")
        .replace("\\&", "&")
        .replace("\\=", "=")
    )


def parse_time_to_seconds(value: str) -> Optional[float]:
    text = value.strip().lower()
    if not text:
        return None
    if text.isdigit():
        return float(text)
    if text.endswith("s") and text[:-1].isdigit():
        return float(text[:-1])

    total = 0.0
    matches = re.findall(r"(\d+)([hms])", text)
    if matches:
        for amount, unit in matches:
            factor = {"h": 3600, "m": 60, "s": 1}[unit]
            total += int(amount) * factor
        return total
    return None


def extract_seek_seconds(video_arg: str) -> Optional[float]:
    normalized = normalize_video_arg(video_arg)
    parsed = urlparse(normalized)
    query = parse_qs(parsed.query)
    for key in ("t", "start"):
        values = query.get(key)
        if not values:
            continue
        seconds = parse_time_to_seconds(values[0])
        if seconds is not None:
            return seconds
    return None


def _youtube_format_sort_key(fmt: dict) -> tuple[int, int, int, int, int, float, int, float]:
    height = int(fmt.get("height") or 0)
    fps = int(fmt.get("fps") or 0)
    has_audio = 1 if str(fmt.get("acodec") or "").lower() not in {"", "none"} else 0
    non_hls = 1 if str(fmt.get("protocol") or "").lower() not in {"m3u8", "m3u8_native"} else 0
    mp4_container = 1 if str(fmt.get("ext") or "").lower() == "mp4" else 0
    codec_score = 1 if "avc" in str(fmt.get("vcodec") or "").lower() else 0
    tbr = float(fmt.get("tbr") or 0.0)
    return (height >= 720, non_hls, mp4_container, codec_score, height, fps, has_audio, tbr)


def _select_youtube_stream(info: dict) -> tuple[Optional[str], str]:
    formats = info.get("formats") or []
    candidates = [
        fmt
        for fmt in formats
        if fmt.get("url") and str(fmt.get("vcodec") or "").lower() not in {"", "none"}
    ]
    if candidates:
        selected = max(candidates, key=_youtube_format_sort_key)
        return selected.get("url"), str(info.get("title") or "YouTube stream")

    stream_url = info.get("url")
    if stream_url:
        return stream_url, str(info.get("title") or "YouTube stream")

    return None, str(info.get("title") or "YouTube stream")


def resolve_youtube_with_python(url: str) -> tuple[str, str]:
    try:
        import yt_dlp  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Python yt-dlp import failed: {exc}") from exc

    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise RuntimeError(f"Python yt-dlp extraction failed: {exc}") from exc

    stream_url, display_name = _select_youtube_stream(info)
    if not stream_url:
        raise RuntimeError("Python yt-dlp resolved the URL but did not return a playable stream URL.")
    return stream_url, display_name


def resolve_youtube_with_cli(url: str) -> tuple[str, str]:
    command = ["yt-dlp", "--dump-single-json", "--no-playlist", url]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp CLI is not installed or not on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"yt-dlp CLI failed: {stderr or exc}") from exc

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"yt-dlp CLI returned invalid JSON: {exc}") from exc

    stream_url, display_name = _select_youtube_stream(info)
    if not stream_url:
        raise RuntimeError("yt-dlp CLI resolved the URL but did not return a playable stream URL.")
    return stream_url, display_name


def resolve_video_source(video_arg: str) -> tuple[str | int, str]:
    video_arg = normalize_video_arg(video_arg)

    if video_arg.isdigit():
        return int(video_arg), video_arg

    if video_arg.startswith("http"):
        errors: list[str] = []
        try:
            return resolve_youtube_with_python(video_arg)
        except Exception as exc:
            errors.append(str(exc))
        try:
            return resolve_youtube_with_cli(video_arg)
        except Exception as exc:
            errors.append(str(exc))
        raise RuntimeError("Could not resolve the YouTube source. " + " | ".join(errors))

    path = Path(video_arg)
    if path.exists():
        return str(path), path.name

    return video_arg, video_arg


def next_match_index(output_dir: Path, match_prefix: str, digits: int) -> int:
    pattern = re.compile(rf"^{re.escape(match_prefix)}_(\d{{{max(1, digits)},}})\.[^.]+$")
    highest = 0
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def create_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output file: {path}")
    return writer


def main() -> int:
    args = parse_args()
    ensure_data_dirs()

    template_a = build_template_set(load_template(args.template_a), args.search_scale)
    template_b = build_template_set(load_template(args.template_b), args.search_scale)
    roi = tuple(args.roi) if args.roi else None

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source, source_name = resolve_video_source(args.video)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {args.video}", file=sys.stderr)
        return 1

    seek_seconds = args.seek_seconds if args.seek_seconds is not None else extract_seek_seconds(args.video)
    if seek_seconds and seek_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_seconds * 1000.0)

    source_fps = get_video_fps(cap)
    analysis_stride = 1
    if args.analysis_fps > 0:
        analysis_stride = max(1, int(round(source_fps / args.analysis_fps)))
    clip_frame_count = max(1, int(round(args.clip_seconds * source_fps)))

    state = DetectionState()
    frame_index = 0
    clip_index = (
        args.start_index
        if args.start_index is not None
        else next_match_index(output_dir, args.match_prefix, args.index_digits)
    )
    clip_writer: Optional[cv2.VideoWriter] = None
    clip_path: Optional[Path] = None
    clip_end_frame: Optional[int] = None
    clips_written = 0
    saw_start_event = False
    partial_clip_saved = False

    print(f"Resolved source: {source_name}")
    print(f"Saving clips to: {output_dir}")
    if seek_seconds and seek_seconds > 0:
        print(f"Initial seek: {format_video_time(seek_seconds)}")
    print(
        f"Detector settings: start_threshold={args.start_threshold:.2f}, end_threshold={args.threshold:.2f}, "
        f"start_frames={args.start_frames}, stop_frames={args.stop_frames}, analysis_stride={analysis_stride}, "
        f"search_scale={args.search_scale:.2f}"
    )
    print(
        f"Clip settings: duration={args.clip_seconds:.1f}s ({clip_frame_count} frames at {source_fps:.2f} fps), "
        f"prefix={args.match_prefix}, next_index={clip_index}"
    )

    if args.show:
        cv2.namedWindow("match-clipper", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            analyzed_this_frame = frame_index % analysis_stride == 0
            match_result = None
            roi_offset = (0, 0)
            current_best_score = 0.0
            wrote_current_frame = False

            if analyzed_this_frame:
                match_result, _, roi_offset = run_match(frame, template_a, template_b, roi, args.threshold)
                current_best_score = max(match_result.score_a, match_result.score_b)
                events = update_detection_state(
                    state,
                    current_best_score,
                    args.start_frames,
                    args.stop_frames,
                    args.start_threshold,
                    args.threshold,
                )

                for event in events:
                    video_time = format_video_time(frame_index / source_fps)
                    stamp = time.strftime("%H:%M:%S")
                    print(f"[{stamp}] {event} at {video_time} (frame {frame_index}, score={current_best_score:.3f})")

                    if event == "MATCH STARTED":
                        saw_start_event = True
                        if clip_writer is None:
                            height, width = frame.shape[:2]
                            clip_path = output_dir / f"{args.match_prefix}_{clip_index:0{args.index_digits}d}.mp4"
                            clip_writer = create_writer(clip_path, source_fps, (width, height))
                            clip_end_frame = frame_index + clip_frame_count - 1
                            print(
                                f"Started writing {clip_path.name} from frame {frame_index}; "
                                f"scheduled end at frame {clip_end_frame}."
                            )
                            clip_writer.write(frame)
                            wrote_current_frame = True

            if clip_writer is not None:
                if not wrote_current_frame:
                    clip_writer.write(frame)

                if clip_end_frame is not None and frame_index >= clip_end_frame:
                    clip_writer.release()
                    assert clip_path is not None
                    print(f"Saved {clip_path.name}")
                    clip_writer = None
                    clip_path = None
                    clip_end_frame = None
                    clip_index += 1
                    clips_written += 1
                    if args.max_clips is not None and clips_written >= args.max_clips:
                        break

            if args.show and match_result is not None:
                debug_frame = draw_match(
                    frame,
                    roi_offset,
                    match_result.score_a,
                    match_result.loc_a,
                    match_result.size_a,
                    match_result.score_b,
                    match_result.loc_b,
                    match_result.size_b,
                    args.start_threshold if not state.active else args.threshold,
                    state,
                )
                video_time = format_video_time(frame_index / source_fps)
                cv2.putText(
                    debug_frame,
                    f"frame={frame_index} time={video_time}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("match-clipper", debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_index += 1

    finally:
        cap.release()
        if clip_writer is not None:
            clip_writer.release()
            partial_clip_saved = clip_path is not None
        cv2.destroyAllWindows()

    if not saw_start_event:
        print("No match start was detected.")
    elif partial_clip_saved and clip_path is not None:
        print(f"Source ended before the full clip duration; saved partial clip {clip_path.name}.")

    print(f"Finished after writing {clips_written} clip(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
