from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .calibration import points_form_valid_quad
from .config import ARTIFACT_ROOT, FUEL_DENSITY_MAP_ROOT, FUEL_FIELD_IMAGE_PATH, FUEL_PROCESSOR_SCRIPT, TARGET_FPS
from .fuel_builtin import run_builtin_fuel_processor
from .schemas import FuelArtifactSet, FuelProcessingProgress, MatchRecord, SourceSubmission
from .storage import TrackingStore


DEFAULT_FUEL_BASE_COLOR = [255, 255, 0]
PROGRESS_JSON_PREFIX = "PROGRESS_JSON:"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_rgb_color(value: Any) -> Optional[list[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    normalized: list[int] = []
    for channel in value:
        if not isinstance(channel, (int, float)):
            return None
        normalized.append(int(max(0, min(255, round(float(channel))))))
    return normalized


def order_quad_points(points: list[list[float]]) -> list[list[float]]:
    if len(points) != 4:
        return points

    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    sorted_by_angle = sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))

    top_edge_start_index = min(
        range(len(sorted_by_angle)),
        key=lambda index: (sorted_by_angle[index][1] + sorted_by_angle[(index + 1) % len(sorted_by_angle)][1]) * 0.5,
    )
    rotated = [sorted_by_angle[(top_edge_start_index + index) % len(sorted_by_angle)] for index in range(len(sorted_by_angle))]
    edge_top_a, edge_top_b, edge_bottom_a, edge_bottom_b = rotated
    top_left, top_right = (edge_top_a, edge_top_b) if edge_top_a[0] <= edge_top_b[0] else (edge_top_b, edge_top_a)
    bottom_left, bottom_right = (edge_bottom_a, edge_bottom_b) if edge_bottom_a[0] <= edge_bottom_b[0] else (edge_bottom_b, edge_bottom_a)
    return [top_left, top_right, bottom_right, bottom_left]


def normalize_quad(value: Any) -> Optional[list[list[float]]]:
    if not isinstance(value, list) or len(value) != 4:
        return None

    quad: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        x_val = float(x)
        y_val = float(y)
        if x_val < 0.0 or x_val > 1.0 or y_val < 0.0 or y_val > 1.0:
            return None
        quad.append([x_val, y_val])

    ordered_quad = order_quad_points(quad)
    if not points_form_valid_quad([[point[0] * 1000.0, point[1] * 1000.0] for point in ordered_quad], min_area_floor=10.0):
        return None
    return ordered_quad


def quad_to_pixels(quad: list[list[float]], width: int, height: int) -> list[list[float]]:
    return [
        [round(clamp01(point[0]) * max(width - 1, 1)), round(clamp01(point[1]) * max(height - 1, 1))]
        for point in quad
    ]


def bbox_from_quad_pixels(quad: list[list[float]], width: int, height: int, padding_px: int = 6) -> tuple[int, int, int, int]:
    xs = [int(round(point[0])) for point in quad]
    ys = [int(round(point[1])) for point in quad]
    left = max(0, min(xs) - padding_px)
    top = max(0, min(ys) - padding_px)
    right = min(max(width - 1, 0), max(xs) + padding_px)
    bottom = min(max(height - 1, 0), max(ys) + padding_px)
    return left, top, max(1, right - left + 1), max(1, bottom - top + 1)


def _artifact_url(match_id: str, name: str) -> str:
    return f"/artifacts/{match_id}/{urllib.parse.quote(name)}"


def _artifact_path(match_id: str, name: str) -> Path:
    return ARTIFACT_ROOT / match_id / name


def clear_fuel_artifacts(match: MatchRecord) -> None:
    match.fuel_analysis.artifacts = FuelArtifactSet()
    match.fuel_analysis.stats = {}
    match.fuel_analysis.processing_progress = None
    match.fuel_analysis.last_error = None
    match.fuel_analysis.updated_at = time.time()


def invalidate_fuel_analysis(match: MatchRecord) -> None:
    clear_fuel_artifacts(match)
    match.fuel_analysis.status = "ready" if match.fuel_calibration.ground_quad else "idle"


def resolve_match_video_path(match: MatchRecord) -> str:
    source_name = str(match.source.get("source_name") or match.metadata.get("display_name") or f"match-{match.id}")
    stored_path = match.source.get("stored_path") if isinstance(match.source, dict) else None
    if stored_path and Path(stored_path).exists():
        return str(Path(stored_path))

    candidate_urls = [
        match.artifacts.trimmed_video,
        match.artifacts.source_video,
        match.artifacts.annotated_video,
    ]
    for url in candidate_urls:
        if not url:
            continue
        candidate = ARTIFACT_ROOT / match.id / urllib.parse.unquote(url.rsplit("/", 1)[-1])
        if candidate.exists():
            return str(candidate)

    source_url = match.source.get("source_url") if isinstance(match.source, dict) else None
    if source_url:
        from .pipeline import resolve_source

        source_kind = str(match.metadata.get("source_kind") or "youtube")
        resolved = resolve_source(
            SourceSubmission(
                source_kind=source_kind if source_kind in {"upload", "youtube", "watchbot"} else "youtube",
                source_url=source_url,
                source_name=source_name,
                stored_path=stored_path,
            )
        )
        return resolved.video_path

    raise RuntimeError("Could not resolve a video source for this match.")


def probe_video_metadata(video_path: str) -> dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the match video for fuel analysis.")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    return {
        "width": float(width),
        "height": float(height),
        "fps": float(fps),
        "frame_count": float(frame_count),
        "duration": float(duration),
    }


def sample_match_video_color(
    match: MatchRecord,
    normalized_x: float,
    normalized_y: float,
    time_sec: float,
    radius: int = 2,
) -> list[int]:
    video_path = resolve_match_video_path(match)
    metadata = probe_video_metadata(video_path)
    width = max(1, int(metadata["width"]))
    height = max(1, int(metadata["height"]))
    duration = float(metadata["duration"])

    target_x = int(round(clamp01(normalized_x) * max(width - 1, 1)))
    target_y = int(round(clamp01(normalized_y) * max(height - 1, 1)))
    seek_msec = max(0.0, min(float(time_sec), max(duration - 0.001, 0.0))) * 1000.0 if duration > 0 else 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the match video for color sampling.")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_msec)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Could not decode a frame at that timestamp.")
    finally:
        cap.release()

    sample_radius = max(0, int(radius))
    start_x = max(0, target_x - sample_radius)
    end_x = min(frame.shape[1] - 1, target_x + sample_radius)
    start_y = max(0, target_y - sample_radius)
    end_y = min(frame.shape[0] - 1, target_y + sample_radius)
    window = frame[start_y : end_y + 1, start_x : end_x + 1]
    if window.size == 0:
        raise RuntimeError("Could not sample a valid color window from the frame.")

    b, g, r = [int(np.median(channel)) for channel in cv2.split(window)]
    return [r, g, b]


def processor_python_bin() -> str:
    override = os.getenv("FUEL_PROCESSOR_PYTHON_BIN")
    if override:
        return override

    script_dir_name = "Scripts" if os.name == "nt" else "bin"
    executable_name = "python.exe" if os.name == "nt" else "python"
    cpu_candidate = FUEL_DENSITY_MAP_ROOT / ".venv" / script_dir_name / executable_name
    cuda_candidate = FUEL_DENSITY_MAP_ROOT / ".venv-opencv-cuda" / script_dir_name / executable_name
    backend = fuel_processor_backend()
    candidates = [cpu_candidate, cuda_candidate, Path(sys.executable)]
    if backend == "cuda":
        candidates = [cuda_candidate, cpu_candidate, Path(sys.executable)]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("python3") or shutil.which("python") or "python"


@lru_cache(maxsize=4)
def _processor_python_supports_cuda(python_bin: str) -> bool:
    command = [
        python_bin,
        "-c",
        (
            "import cv2, sys; "
            "count = -1; "
            "try:\n"
            "    count = int(cv2.cuda.getCudaEnabledDeviceCount())\n"
            "except Exception:\n"
            "    count = -1\n"
            "sys.stdout.write(str(count))"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return False

    if result.returncode != 0:
        return False

    try:
        return int((result.stdout or "").strip() or "-1") > 0
    except ValueError:
        return False


def fuel_processor_backend() -> str:
    value = str(os.getenv("FUEL_PROCESSOR_BACKEND", "")).strip().lower()
    if value in {"cpu", "cuda"}:
        return value

    override = os.getenv("FUEL_PROCESSOR_PYTHON_BIN", "").strip()
    if override and _processor_python_supports_cuda(override):
        return "cuda"

    cuda_candidate = FUEL_DENSITY_MAP_ROOT / ".venv-opencv-cuda" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if cuda_candidate.exists() and _processor_python_supports_cuda(str(cuda_candidate)):
        return "cuda"
    return "cpu"


def fuel_processor_overlay_output() -> str:
    value = str(os.getenv("FUEL_PROCESSOR_OVERLAY_OUTPUT", "none")).strip().lower()
    return value if value in {"video", "frames", "none"} else "none"


def fuel_processor_target_height() -> int:
    try:
        value = int(str(os.getenv("FUEL_PROCESSOR_TARGET_HEIGHT", "720")).strip() or "720")
    except ValueError:
        value = 720
    return max(0, value)


def external_fuel_processor_enabled() -> bool:
    value = str(os.getenv("FUEL_PROCESSOR_DISABLE_EXTERNAL", "")).strip().lower()
    return value not in {"1", "true", "yes", "on"}


def external_fuel_processor_fallback_enabled() -> bool:
    value = str(os.getenv("FUEL_PROCESSOR_DISABLE_BUILTIN_FALLBACK", "")).strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _cleanup_fuel_outputs(
    artifact_dir: Path,
    *,
    preserve_external_log: bool = False,
    preserve_scaled_input: bool = False,
) -> None:
    generated_names = [
        "overlay.png",
        "overlay-transparent.png",
        "raw_data.txt",
        "field-map.json",
        "air-profile.json",
        "stats.json",
        "fuel-process.log",
    ]
    if not preserve_external_log:
        generated_names.append("fuel-process-external.log")
    optional_generated_names = [
        "overlay-video.mp4",
    ]
    generated_patterns = [
        "field-frames-*.jsonpart",
        "air-profile-*.jsonpart",
    ]
    if not preserve_scaled_input:
        generated_patterns.append("fuel_input_*p.mp4")
    generated_dirs = [
        "overlay-frames",
    ]

    for name in generated_names + optional_generated_names:
        path = artifact_dir / name
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    for pattern in generated_patterns:
        for path in artifact_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    for name in generated_dirs:
        path = artifact_dir / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def _meaningful_fuel_error_line(log_lines: list[str], default_message: str) -> str:
    ignored_prefixes = (
        PROGRESS_JSON_PREFIX,
        "Processing sampled frame ",
    )
    ignored_exact = {
        "Processing video...",
    }
    for line in reversed([entry.strip() for entry in log_lines if entry.strip()]):
        if line in ignored_exact:
            continue
        if any(line.startswith(prefix) for prefix in ignored_prefixes):
            continue
        return line
    return default_message


def _transcode_fuel_video_input(input_path: str, output_path: Path, target_height: int) -> str:
    backend = fuel_processor_backend()
    base_command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"scale=-2:{int(target_height)}",
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    commands = []
    if backend == "cuda":
        commands.append(base_command + ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "28", str(output_path)])
    commands.append(base_command + ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", str(output_path)])

    for command in commands:
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return str(output_path)
        except Exception:
            continue
    return input_path


def prepare_fuel_video_input(video_path: str, artifact_dir: Path) -> str:
    target_height = fuel_processor_target_height()
    if target_height <= 0:
        return video_path

    metadata = probe_video_metadata(video_path)
    height = int(metadata["height"])
    if height <= 0 or height <= target_height:
        return video_path

    output_path = artifact_dir / f"fuel_input_{target_height}p.mp4"
    if output_path.exists() and output_path.stat().st_size > 0:
        return str(output_path)
    return _transcode_fuel_video_input(video_path, output_path, target_height)


def update_fuel_debug(match: MatchRecord) -> None:
    match.debug.setdefault("fuel", {})
    match.debug["fuel"].update(
        {
            "status": match.fuel_analysis.status,
            "updated_at": match.fuel_analysis.updated_at,
            "stats": match.fuel_analysis.stats,
            "last_error": match.fuel_analysis.last_error,
        }
    )


def _set_fuel_processing_progress(
    match: MatchRecord,
    store: TrackingStore,
    *,
    started_at: float,
    phase: str,
    current: int,
    total: int,
) -> None:
    now = time.time()
    match.fuel_analysis.processing_progress = FuelProcessingProgress(
        phase=phase,
        current=max(0, int(current)),
        total=max(1, int(total)),
        started_at=started_at,
        updated_at=now,
    )
    match.fuel_analysis.updated_at = now
    update_fuel_debug(match)
    store.save_match(match)


def _set_fuel_analysis_error(
    match: MatchRecord,
    store: TrackingStore,
    message: str,
    *,
    artifact_dir: Path,
) -> None:
    log_path = artifact_dir / "fuel-process.log"
    if not log_path.exists():
        log_path.write_text(f"{message}\n", encoding="utf-8")
    match.fuel_analysis.status = "error"
    match.fuel_analysis.last_error = message
    match.fuel_analysis.processing_progress = None
    match.fuel_analysis.updated_at = time.time()
    if log_path.exists():
        match.fuel_analysis.artifacts.process_log = _artifact_url(match.id, log_path.name)
    update_fuel_debug(match)
    store.save_match(match)


def _finalize_fuel_analysis(match: MatchRecord, store: TrackingStore, artifact_dir: Path) -> MatchRecord:
    stats_path = artifact_dir / "stats.json"
    stats_payload: dict[str, Any] = {}
    if stats_path.exists():
        stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))

    log_path = artifact_dir / "fuel-process.log"
    match.fuel_analysis.status = "completed"
    match.fuel_analysis.processing_progress = None
    match.fuel_analysis.last_error = None
    match.fuel_analysis.updated_at = time.time()
    match.fuel_analysis.stats = stats_payload
    match.fuel_analysis.artifacts = FuelArtifactSet(
        overlay_image=_artifact_url(match.id, "overlay.png") if _artifact_path(match.id, "overlay.png").exists() else None,
        overlay_transparent_image=(
            _artifact_url(match.id, "overlay-transparent.png")
            if _artifact_path(match.id, "overlay-transparent.png").exists()
            else None
        ),
        overlay_video=_artifact_url(match.id, "overlay-video.mp4") if _artifact_path(match.id, "overlay-video.mp4").exists() else None,
        raw_data=_artifact_url(match.id, "raw_data.txt") if _artifact_path(match.id, "raw_data.txt").exists() else None,
        field_map=_artifact_url(match.id, "field-map.json") if _artifact_path(match.id, "field-map.json").exists() else None,
        air_profile=_artifact_url(match.id, "air-profile.json") if _artifact_path(match.id, "air-profile.json").exists() else None,
        stats_file=_artifact_url(match.id, "stats.json") if stats_path.exists() else None,
        process_log=_artifact_url(match.id, log_path.name) if log_path.exists() else None,
    )
    update_fuel_debug(match)
    store.save_match(match)
    return match


def _run_external_fuel_processor(
    *,
    video_path: str,
    artifact_dir: Path,
    bbox: tuple[int, int, int, int],
    ground_quad_pixels: list[list[float]],
    left_wall_quad_pixels: Optional[list[list[float]]],
    right_wall_quad_pixels: Optional[list[list[float]]],
    fuel_base_color: list[int],
    match: MatchRecord,
    store: TrackingStore,
    started_at: float,
) -> None:
    command = [
        processor_python_bin(),
        "-u",
        str(FUEL_PROCESSOR_SCRIPT),
        "--video",
        video_path,
        "--session-dir",
        str(artifact_dir),
        "--bbox",
        ",".join(str(int(value)) for value in bbox),
        "--quad",
        ",".join(f"{int(point[0])},{int(point[1])}" for point in ground_quad_pixels),
        "--overlay-output",
        fuel_processor_overlay_output(),
        "--backend",
        fuel_processor_backend(),
        "--field-image",
        str(FUEL_FIELD_IMAGE_PATH),
        "--fuel-base-color",
        ",".join(str(value) for value in fuel_base_color),
    ]
    if left_wall_quad_pixels:
        command.extend(["--wall-quad-left", ",".join(f"{int(point[0])},{int(point[1])}" for point in left_wall_quad_pixels)])
    if right_wall_quad_pixels:
        command.extend(["--wall-quad-right", ",".join(f"{int(point[0])},{int(point[1])}" for point in right_wall_quad_pixels)])

    output_lines: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=str(FUEL_DENSITY_MAP_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not start fuel processor: {exc}") from exc

    try:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            if line.startswith(PROGRESS_JSON_PREFIX):
                try:
                    payload = json.loads(line[len(PROGRESS_JSON_PREFIX) :].strip())
                except json.JSONDecodeError:
                    continue
                _set_fuel_processing_progress(
                    match,
                    store,
                    started_at=started_at,
                    phase=str(payload.get("phase") or "frames"),
                    current=int(payload.get("current") or 0),
                    total=int(payload.get("total") or 1),
                )
        return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()

    log_path = artifact_dir / "fuel-process.log"
    log_text = "".join(output_lines).strip()
    log_path.write_text(f"{log_text}\n" if log_text else "", encoding="utf-8")

    if return_code != 0:
        error_message = f"Fuel processor exited with code {return_code}."
        if output_lines:
            error_message = _meaningful_fuel_error_line(output_lines, error_message)
        raise RuntimeError(error_message)


def run_fuel_analysis(match: MatchRecord, store: TrackingStore) -> MatchRecord:
    if not match.fuel_calibration.ground_quad:
        raise RuntimeError("Fuel analysis needs four calibrated ground corners first.")

    artifact_dir = store.create_match_artifact_dir(match.id)
    _cleanup_fuel_outputs(artifact_dir)
    video_path = prepare_fuel_video_input(resolve_match_video_path(match), artifact_dir)
    metadata = probe_video_metadata(video_path)
    width = max(1, int(metadata["width"]))
    height = max(1, int(metadata["height"]))

    ground_quad_pixels = quad_to_pixels(match.fuel_calibration.ground_quad, width, height)
    bbox = bbox_from_quad_pixels(ground_quad_pixels, width, height)
    left_wall_quad_pixels = (
        quad_to_pixels(match.fuel_calibration.left_wall_quad, width, height)
        if match.fuel_calibration.left_wall_quad
        else None
    )
    right_wall_quad_pixels = (
        quad_to_pixels(match.fuel_calibration.right_wall_quad, width, height)
        if match.fuel_calibration.right_wall_quad
        else None
    )
    fuel_base_color = normalize_rgb_color(match.fuel_calibration.fuel_base_color) or list(DEFAULT_FUEL_BASE_COLOR)

    started_at = time.time()
    clear_fuel_artifacts(match)
    match.fuel_analysis.status = "processing"
    match.fuel_analysis.processing_progress = FuelProcessingProgress(
        phase="starting",
        current=0,
        total=1,
        started_at=started_at,
        updated_at=started_at,
    )
    match.fuel_analysis.updated_at = started_at
    update_fuel_debug(match)
    store.save_match(match)
    try:
        progress_interval = 12

        def builtin_progress(phase: str, current: int, total: int) -> None:
            if current not in {0, total} and current % progress_interval != 0:
                return
            _set_fuel_processing_progress(
                match,
                store,
                started_at=started_at,
                phase=phase,
                current=current,
                total=total,
            )

        external_error: Exception | None = None
        if FUEL_PROCESSOR_SCRIPT.exists() and external_fuel_processor_enabled():
            try:
                _run_external_fuel_processor(
                    video_path=video_path,
                    artifact_dir=artifact_dir,
                    bbox=bbox,
                    ground_quad_pixels=ground_quad_pixels,
                    left_wall_quad_pixels=left_wall_quad_pixels,
                    right_wall_quad_pixels=right_wall_quad_pixels,
                    fuel_base_color=fuel_base_color,
                    match=match,
                    store=store,
                    started_at=started_at,
                )
            except Exception as exc:
                external_error = exc
                external_log = artifact_dir / "fuel-process.log"
                archived_external_log = artifact_dir / "fuel-process-external.log"
                if external_log.exists():
                    external_log.replace(archived_external_log)
                if not external_fuel_processor_fallback_enabled():
                    raise
                _cleanup_fuel_outputs(artifact_dir, preserve_external_log=True, preserve_scaled_input=True)
                match.debug.setdefault("fuel", {})
                match.debug["fuel"]["external_failure"] = str(exc)
                store.save_match(match)

        if external_error is None and FUEL_PROCESSOR_SCRIPT.exists() and external_fuel_processor_enabled():
            return _finalize_fuel_analysis(match, store, artifact_dir)

        run_builtin_fuel_processor(
            video_path=video_path,
            artifact_dir=artifact_dir,
            field_image_path=FUEL_FIELD_IMAGE_PATH,
            ground_quad_pixels=ground_quad_pixels,
            left_wall_quad_pixels=left_wall_quad_pixels,
            right_wall_quad_pixels=right_wall_quad_pixels,
            fuel_base_color=fuel_base_color,
            analysis_fps=float(max(1.0, TARGET_FPS)),
            progress_callback=builtin_progress,
        )
    except Exception as exc:
        _set_fuel_analysis_error(match, store, str(exc), artifact_dir=artifact_dir)
        raise RuntimeError(str(exc)) from exc
    return _finalize_fuel_analysis(match, store, artifact_dir)
