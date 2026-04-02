from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from mss import mss
except Exception:
    mss = None


YELLOW_KERNEL = np.ones((5, 5), dtype=np.uint8)
EXPAND_KERNEL = np.ones((9, 9), dtype=np.uint8)
BLACK_KERNEL = np.ones((3, 3), dtype=np.uint8)


@dataclass
class DetectionState:
    active: bool = False
    present_streak: int = 0
    missing_streak: int = 0
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


@dataclass
class MatchResult:
    score_a: float
    loc_a: tuple[int, int]
    size_a: tuple[int, int]
    score_b: float
    loc_b: tuple[int, int]
    size_b: tuple[int, int]
    visible: bool = False


@dataclass
class TemplateSet:
    full: np.ndarray
    search: np.ndarray
    scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple match start/stop tester using two small on-screen template images. "
            "By default it searches the entire frame or screen, and the match is "
            "considered active when either template is consistently visible."
        )
    )
    parser.add_argument("--template-a", required=True, help="Path to the first icon image.")
    parser.add_argument("--template-b", required=True, help="Path to the second icon image.")
    parser.add_argument(
        "--video",
        help="Video path, stream URL, or camera index. Omit this and add --screen to capture the desktop.",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Capture the desktop instead of opening a video source. Requires `mss`.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Optional region of interest. If omitted, the program searches the whole frame or screen.",
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
        help="Frames with either icon visible before declaring a match start. Default: 3",
    )
    parser.add_argument(
        "--stop-frames",
        type=int,
        default=20,
        help="Frames with neither icon visible before declaring a match end. Default: 20",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Capture rate for screen mode. Default: 8.0",
    )
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=6.0,
        help="How many video frames per second to actually analyze from video files. Default: 6.0",
    )
    parser.add_argument(
        "--idle-scan-seconds",
        type=float,
        default=2.0,
        help="When using a seekable video file and the match is idle, skip ahead by this many seconds between probes. Default: 2.0",
    )
    parser.add_argument(
        "--idle-samples-per-step",
        type=int,
        default=4,
        help="How many frames to sample inside each idle fast-forward step. Higher is safer, lower is faster. Default: 4",
    )
    parser.add_argument(
        "--search-scale",
        type=float,
        default=0.5,
        help="Scale factor used for template matching on video. Lower is faster but less exact. Default: 0.5",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a debug window with match scores and state.",
    )
    return parser.parse_args()


def load_template(path_str: str) -> np.ndarray:
    path = Path(path_str)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read template image: {path}")
    return image


def build_shape_signature(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, (15, 60, 80), (45, 255, 255))
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 95))

    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, YELLOW_KERNEL)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, YELLOW_KERNEL)

    expanded_yellow = cv2.dilate(yellow_mask, EXPAND_KERNEL, iterations=1)
    black_on_yellow = cv2.bitwise_and(black_mask, expanded_yellow)
    black_on_yellow = cv2.morphologyEx(black_on_yellow, cv2.MORPH_OPEN, BLACK_KERNEL)

    signature = np.zeros(image.shape[:2], dtype=np.uint8)
    signature[yellow_mask > 0] = 96
    signature[black_on_yellow > 0] = 255
    return signature


def build_template_set(template: np.ndarray, search_scale: float) -> TemplateSet:
    signature = build_shape_signature(template)
    scale = min(1.0, max(0.1, float(search_scale)))
    if scale == 1.0:
        return TemplateSet(full=signature, search=signature, scale=1.0)

    width = max(1, int(round(signature.shape[1] * scale)))
    height = max(1, int(round(signature.shape[0] * scale)))
    resized = cv2.resize(signature, (width, height), interpolation=cv2.INTER_AREA)
    return TemplateSet(full=signature, search=resized, scale=scale)


def open_video_source(video_arg: str) -> cv2.VideoCapture:
    source: str | int
    source = int(video_arg) if video_arg.isdigit() else video_arg
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_arg}")
    return cap


def capture_screen_frame(screen_region: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    if mss is None:
        raise RuntimeError("Screen capture requires the `mss` package. Install it with `pip install mss`.")

    with mss() as screen_capture:
        monitor = screen_capture.monitors[1]
        if screen_region is None:
            region = {
                "left": monitor["left"],
                "top": monitor["top"],
                "width": monitor["width"],
                "height": monitor["height"],
            }
        else:
            x, y, w, h = screen_region
            region = {"left": x, "top": y, "width": w, "height": h}
        shot = np.array(screen_capture.grab(region))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def crop_roi(frame: np.ndarray, roi: Optional[tuple[int, int, int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return frame, (0, 0)

    x, y, w, h = roi
    frame_h, frame_w = frame.shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return frame[y:y + h, x:x + w], (x, y)


def match_template_gray(gray: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int], tuple[int, int]]:
    if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
        return 0.0, (0, 0), (template.shape[1], template.shape[0])

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, max_loc = cv2.minMaxLoc(result)
    return float(max_score), max_loc, (template.shape[1], template.shape[0])


def run_match(
    frame: np.ndarray,
    template_a: TemplateSet,
    template_b: TemplateSet,
    roi: Optional[tuple[int, int, int, int]],
    threshold: float,
) -> tuple[MatchResult, np.ndarray, tuple[int, int]]:
    roi_frame, roi_offset = crop_roi(frame, roi)
    search_scale = min(template_a.scale, template_b.scale)
    if search_scale < 1.0:
        scaled_width = max(1, int(round(roi_frame.shape[1] * search_scale)))
        scaled_height = max(1, int(round(roi_frame.shape[0] * search_scale)))
        scaled_frame = cv2.resize(roi_frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        search_image = build_shape_signature(scaled_frame)
    else:
        search_image = build_shape_signature(roi_frame)

    score_a, loc_a_scaled, size_a_scaled = match_template_gray(search_image, template_a.search)
    score_b, loc_b_scaled, size_b_scaled = match_template_gray(search_image, template_b.search)

    inv_scale = 1.0 / search_scale
    loc_a = (int(round(loc_a_scaled[0] * inv_scale)), int(round(loc_a_scaled[1] * inv_scale)))
    loc_b = (int(round(loc_b_scaled[0] * inv_scale)), int(round(loc_b_scaled[1] * inv_scale)))
    size_a = (int(round(size_a_scaled[0] * inv_scale)), int(round(size_a_scaled[1] * inv_scale)))
    size_b = (int(round(size_b_scaled[0] * inv_scale)), int(round(size_b_scaled[1] * inv_scale)))

    result = MatchResult(
        score_a=score_a,
        loc_a=loc_a,
        size_a=size_a,
        score_b=score_b,
        loc_b=loc_b,
        size_b=size_b,
    )
    result.visible = score_a >= threshold or score_b >= threshold
    return result, roi_frame, roi_offset


def update_detection_state(
    state: DetectionState,
    current_score: float,
    start_frames: int,
    stop_frames: int,
    start_threshold: float,
    end_threshold: float,
) -> list[str]:
    events: list[str] = []
    should_start = current_score >= start_threshold
    should_stay_active = current_score >= end_threshold

    if not state.active and should_start:
        state.present_streak += 1
        state.missing_streak = 0
    elif state.active and should_stay_active:
        state.present_streak += 1
        state.missing_streak = 0
    else:
        state.present_streak = 0
        if state.active:
            state.missing_streak += 1

    if not state.active and state.present_streak >= start_frames:
        state.active = True
        state.started_at = time.time()
        state.ended_at = None
        state.missing_streak = 0
        events.append("MATCH STARTED")

    if state.active and not should_stay_active and state.missing_streak >= stop_frames:
        state.active = False
        state.ended_at = time.time()
        state.present_streak = 0
        events.append("MATCH ENDED")

    return events


def is_seekable_video(video_arg: Optional[str]) -> bool:
    if not video_arg:
        return False
    if video_arg.isdigit():
        return False
    if "://" in video_arg:
        return False
    return Path(video_arg).exists()


def get_frame_count(cap: cv2.VideoCapture) -> int:
    value = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return max(0, value)


def get_video_fps(cap: cv2.VideoCapture) -> float:
    value = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if value <= 0:
        return 30.0
    return value


def format_video_time(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> tuple[bool, Optional[np.ndarray]]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        return False, None
    return True, frame


def evaluate_frame_at(
    cap: cv2.VideoCapture,
    frame_index: int,
    template_a: TemplateSet,
    template_b: TemplateSet,
    roi: Optional[tuple[int, int, int, int]],
    threshold: float,
    cache: dict[int, MatchResult],
) -> Optional[MatchResult]:
    if frame_index in cache:
        return cache[frame_index]

    ok, frame = read_frame_at(cap, frame_index)
    if not ok or frame is None:
        return None

    result, _, _ = run_match(frame, template_a, template_b, roi, threshold)
    cache[frame_index] = result
    return result


def find_first_positive_in_range(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    template_a: TemplateSet,
    template_b: TemplateSet,
    roi: Optional[tuple[int, int, int, int]],
    threshold: float,
    cache: dict[int, MatchResult],
) -> Optional[int]:
    for frame_index in range(start_frame, end_frame + 1):
        result = evaluate_frame_at(cap, frame_index, template_a, template_b, roi, threshold, cache)
        if result is not None and result.visible:
            return frame_index
    return None


def refine_start_frame(
    cap: cv2.VideoCapture,
    low_negative: int,
    high_positive: int,
    template_a: TemplateSet,
    template_b: TemplateSet,
    roi: Optional[tuple[int, int, int, int]],
    threshold: float,
    cache: dict[int, MatchResult],
) -> int:
    lo = low_negative + 1
    hi = high_positive

    while lo < hi:
        mid = (lo + hi) // 2
        left_hit = find_first_positive_in_range(cap, lo, mid, template_a, template_b, roi, threshold, cache)
        if left_hit is not None:
            hi = left_hit
        else:
            lo = mid + 1

    final_hit = find_first_positive_in_range(cap, lo, hi, template_a, template_b, roi, threshold, cache)
    return hi if final_hit is None else final_hit


def has_visible_run(
    cap: cv2.VideoCapture,
    start_frame: int,
    run_length: int,
    total_frames: int,
    template_a: TemplateSet,
    template_b: TemplateSet,
    roi: Optional[tuple[int, int, int, int]],
    threshold: float,
    cache: dict[int, MatchResult],
) -> bool:
    if start_frame + run_length > total_frames:
        return False

    for frame_index in range(start_frame, start_frame + run_length):
        result = evaluate_frame_at(cap, frame_index, template_a, template_b, roi, threshold, cache)
        if result is None or max(result.score_a, result.score_b) < threshold:
            return False
    return True


def find_confirmed_start_in_range(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    total_frames: int,
    run_length: int,
    template_a: TemplateSet,
    template_b: TemplateSet,
    roi: Optional[tuple[int, int, int, int]],
    threshold: float,
    cache: dict[int, MatchResult],
) -> Optional[int]:
    for frame_index in range(start_frame, end_frame + 1):
        if has_visible_run(
            cap,
            frame_index,
            run_length,
            total_frames,
            template_a,
            template_b,
            roi,
            threshold,
            cache,
        ):
            return frame_index
    return None


def fast_forward_to_match_start(
    cap: cv2.VideoCapture,
    current_frame: int,
    total_frames: int,
    scan_step: int,
    idle_samples_per_step: int,
    start_frames: int,
    template_a: TemplateSet,
    template_b: TemplateSet,
    roi: Optional[tuple[int, int, int, int]],
    threshold: float,
) -> Optional[int]:
    cache: dict[int, MatchResult] = {}
    probe_frame = current_frame
    last_negative = current_frame - 1
    samples_per_step = max(1, idle_samples_per_step)

    while probe_frame < total_frames:
        interval_end = min(total_frames - 1, probe_frame + scan_step - 1)
        step_span = max(0, interval_end - probe_frame)

        sample_frames: list[int] = []
        for sample_index in range(samples_per_step):
            if samples_per_step == 1:
                candidate = probe_frame
            else:
                offset = int(round((step_span * sample_index) / (samples_per_step - 1)))
                candidate = probe_frame + offset
            if not sample_frames or candidate != sample_frames[-1]:
                sample_frames.append(candidate)

        hit_frame: Optional[int] = None
        for candidate in sample_frames:
            result = evaluate_frame_at(cap, candidate, template_a, template_b, roi, threshold, cache)
            if result is not None and result.visible:
                hit_frame = candidate
                break

        if hit_frame is not None:
            search_start = max(current_frame, last_negative + 1)
            first_positive = refine_start_frame(
                cap,
                search_start - 1,
                hit_frame,
                template_a,
                template_b,
                roi,
                threshold,
                cache,
            )
            confirmed_start = find_confirmed_start_in_range(
                cap,
                search_start,
                first_positive,
                total_frames,
                max(1, start_frames),
                template_a,
                template_b,
                roi,
                threshold,
                cache,
            )
            if confirmed_start is not None:
                return confirmed_start

        last_negative = interval_end
        probe_frame += scan_step

    return None


def draw_match(
    frame: np.ndarray,
    roi_offset: tuple[int, int],
    score_a: float,
    loc_a: tuple[int, int],
    size_a: tuple[int, int],
    score_b: float,
    loc_b: tuple[int, int],
    size_b: tuple[int, int],
    threshold: float,
    state: DetectionState,
) -> np.ndarray:
    debug = frame.copy()
    offset_x, offset_y = roi_offset

    def draw_box(loc: tuple[int, int], size: tuple[int, int], score: float, color: tuple[int, int, int]) -> None:
        x1 = loc[0] + offset_x
        y1 = loc[1] + offset_y
        x2 = x1 + size[0]
        y2 = y1 + size[1]
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            debug,
            f"{score:.3f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    color_a = (0, 220, 0) if score_a >= threshold else (0, 120, 220)
    color_b = (220, 200, 0) if score_b >= threshold else (220, 120, 0)
    draw_box(loc_a, size_a, score_a, color_a)
    draw_box(loc_b, size_b, score_b, color_b)

    status = "ACTIVE" if state.active else "IDLE"
    cv2.putText(debug, f"state={status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        debug,
        f"a={score_a:.3f} b={score_b:.3f} threshold={threshold:.2f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        debug,
        f"present_streak={state.present_streak} missing_streak={state.missing_streak}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return debug


def main() -> int:
    args = parse_args()

    if not args.screen and not args.video:
        print("Use either --video or --screen.", file=sys.stderr)
        return 2

    if args.screen and args.video:
        print("Choose one source mode: either --video or --screen.", file=sys.stderr)
        return 2

    template_a = build_template_set(load_template(args.template_a), args.search_scale)
    template_b = build_template_set(load_template(args.template_b), args.search_scale)
    roi = tuple(args.roi) if args.roi else None

    cap: Optional[cv2.VideoCapture] = None
    seekable_video = False
    total_frames = 0
    scan_step = 1
    analysis_stride = 1
    if args.video:
        cap = open_video_source(args.video)
        seekable_video = is_seekable_video(args.video)
        fps = get_video_fps(cap)
        if args.analysis_fps > 0:
            analysis_stride = max(1, int(round(fps / args.analysis_fps)))
        if seekable_video:
            total_frames = get_frame_count(cap)
            scan_step = max(1, int(round(fps * args.idle_scan_seconds)))
            if total_frames <= 0:
                seekable_video = False

    state = DetectionState()
    frame_index = 0
    last_frame_time = time.time()
    video_fps = get_video_fps(cap) if cap is not None else 0.0
    best_score_seen = -1.0
    best_score_frame = -1
    saw_start_event = False

    search_scope = f"ROI {roi}" if roi else "full frame/screen"
    print(f"Searching {search_scope} for either template.")
    print(
        f"Start after {args.start_frames} matching frames, end after {args.stop_frames} missing frames, "
        f"start_threshold={args.start_threshold:.2f} end_threshold={args.threshold:.2f}"
    )
    print(f"Search scale={min(template_a.scale, template_b.scale):.2f}")
    if args.video:
        effective_analysis_fps = (video_fps / analysis_stride) if video_fps > 0 else args.analysis_fps
        print(
            f"Sequential video mode: analyzing every {analysis_stride} frame(s) "
            f"at about {effective_analysis_fps:.2f} fps."
        )

    if args.show:
        cv2.namedWindow("match-icon-test", cv2.WINDOW_NORMAL)
        startup_frame = np.zeros((360, 720, 3), dtype=np.uint8)
        cv2.putText(
            startup_frame,
            "Starting match icon preview...",
            (30, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.imshow("match-icon-test", startup_frame)
        cv2.waitKey(1)

    try:
        while True:
            if args.screen:
                frame = capture_screen_frame(roi)
                match_result, _, roi_offset = run_match(frame, template_a, template_b, roi, args.threshold)
                if args.fps > 0:
                    target_interval = 1.0 / args.fps
                    elapsed = time.time() - last_frame_time
                    if elapsed < target_interval:
                        time.sleep(target_interval - elapsed)
                    last_frame_time = time.time()
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                match_result, _, roi_offset = run_match(frame, template_a, template_b, roi, args.threshold)

            current_best_score = max(match_result.score_a, match_result.score_b)
            if current_best_score > best_score_seen:
                best_score_seen = current_best_score
                best_score_frame = frame_index
            events = update_detection_state(
                state,
                current_best_score,
                args.start_frames,
                args.stop_frames,
                args.start_threshold,
                args.threshold,
            )

            timestamp = time.strftime("%H:%M:%S")
            video_time = format_video_time(frame_index / video_fps) if video_fps > 0 else "live"

            if args.show:
                print(
                    f"[{timestamp}] frame={frame_index:06d} "
                    f"video_time={video_time} "
                    f"score_a={match_result.score_a:.3f} score_b={match_result.score_b:.3f} active={state.active}"
                )

            for event in events:
                if event == "MATCH STARTED":
                    saw_start_event = True
                print(f"[{timestamp}] {event} at video time {video_time} (frame {frame_index})")

            if args.show:
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
                cv2.imshow("match-icon-test", debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.video and cap is not None:
                skipped = 0
                while skipped < analysis_stride - 1:
                    if not cap.grab():
                        break
                    skipped += 1
                frame_index += 1 + skipped
            else:
                frame_index += 1

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    if not saw_start_event and best_score_frame >= 0:
        best_time = format_video_time(best_score_frame / video_fps) if video_fps > 0 else "live"
        print(
            f"No match start detected. Best score seen was {best_score_seen:.3f} "
            f"at video time {best_time} (frame {best_score_frame})."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
