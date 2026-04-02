from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import yt_dlp
except Exception:
    yt_dlp = None

from .calibration import (
    calibration_for_match,
    load_default_calibration,
    project_detection_candidates,
    project_detection_in_view,
)
from .config import (
    FIELD_HEIGHT_IN,
    FIELD_WIDTH_IN,
    MERGE_DISTANCE_IN,
    MOTION_TRACKER_ALGORITHM,
    OCR_INTERVAL_SECONDS,
    OCR_TIMEOUT_SECONDS,
    ROBOFLOW_API_KEY,
    ROBOT_MODEL_ID,
    TARGET_FPS,
)
from .image_tracker import ImageTrackObservation, ImageTrackerManager
from .schemas import CalibrationEnvelope, DetectionRecord, JobRecord, MatchArtifactSet, MatchRecord, SourceSubmission, TrackRecord
from .storage import TrackingStore
from .tracker import FieldTracker

try:
    from inference import get_model
except Exception:
    get_model = None


TIMER_PATTERN = re.compile(r"(\d{1,2})[:.](\d{2})")
FIELD_IMAGE_SCALE_X = 1.365
FIELD_IMAGE_SCALE_Y = 1.125
FIELD_IMAGE_PATH = Path(__file__).resolve().parents[2] / "frontend" / "mais" / "public" / "2026_No-Fuel_Transparent.png"
YOUTUBE_MIN_HEIGHT = 720
ANNOTATION_DEFAULT_COLOR = (214, 224, 230)
ANNOTATION_FUSED_COLOR = (180, 90, 255)
ANNOTATION_FALLBACK_COLOR = (80, 210, 255)


@dataclass
class ResolvedSource:
    video_path: str
    display_name: str
    source_url: Optional[str]
    stream_height: Optional[int] = None


def _youtube_format_sort_key(fmt: dict) -> tuple[int, int, int, int, int, float, int, float]:
    protocol = str(fmt.get("protocol") or "").lower()
    ext = str(fmt.get("ext") or "").lower()
    vcodec = str(fmt.get("vcodec") or "").lower()
    acodec = str(fmt.get("acodec") or "").lower()
    height = int(fmt.get("height") or 0)
    fps = float(fmt.get("fps") or 0.0)
    tbr = float(fmt.get("tbr") or 0.0)

    # Prefer 720p+, non-HLS transport, MP4 containers, and H.264-compatible streams.
    meets_target = 1 if height >= YOUTUBE_MIN_HEIGHT else 0
    non_hls = 0 if "m3u8" in protocol else 1
    mp4_container = 1 if ext == "mp4" else 0
    codec_score = 2 if ("avc1" in vcodec or "h264" in vcodec) else 1 if ext == "mp4" else 0
    has_audio = 1 if acodec not in {"", "none"} else 0

    return (meets_target, non_hls, mp4_container, codec_score, height, fps, has_audio, tbr)


def _select_youtube_stream(info: dict) -> tuple[Optional[str], Optional[int]]:
    formats = info.get("formats") or []
    candidates = [
        fmt
        for fmt in formats
        if fmt.get("url") and str(fmt.get("vcodec") or "").lower() not in {"", "none"}
    ]
    if candidates:
        selected = max(candidates, key=_youtube_format_sort_key)
        return selected.get("url"), int(selected.get("height") or 0) or None

    stream_url = info.get("url")
    if stream_url:
        return stream_url, int(info.get("height") or 0) or None

    return None, None


def resolve_source(source: SourceSubmission) -> ResolvedSource:
    if source.source_kind == "upload" and source.stored_path:
        return ResolvedSource(video_path=source.stored_path, display_name=source.source_name, source_url=None)

    if source.source_kind in {"youtube", "watchbot"} and source.source_url:
        if source.source_url.startswith("http"):
            if yt_dlp is None:
                raise RuntimeError("yt-dlp is required to resolve YouTube sources.")
            ydl_opts = {
                "quiet": True,
                "noplaylist": True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source.source_url, download=False)
            except Exception as exc:
                raise RuntimeError(f"Could not resolve the YouTube source: {exc}") from exc

            stream_url, stream_height = _select_youtube_stream(info)
            if not stream_url:
                raise RuntimeError("yt-dlp resolved the source but did not return a playable stream URL.")

            display_name = info.get("title") or source.source_name
            return ResolvedSource(
                video_path=stream_url,
                display_name=display_name,
                source_url=source.source_url,
                stream_height=stream_height,
            )

    raise ValueError("Unsupported source submission.")


def read_match_timer(frame: np.ndarray, last_ocr_time: float) -> tuple[Optional[int], float]:
    if pytesseract is None:
        return None, last_ocr_time
    now = time.time()
    if now - last_ocr_time < OCR_INTERVAL_SECONDS:
        return None, last_ocr_time

    height, width, _ = frame.shape
    roi = frame[int(height * 0.02):int(height * 0.12), int(width * 0.35):int(width * 0.65)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)
    _, gray = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    try:
        text = pytesseract.image_to_string(
            gray,
            config="--psm 7 -c tessedit_char_whitelist=0123456789:",
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except RuntimeError as exc:
        if "timeout" in str(exc).lower():
            return None, now
        raise
    match = TIMER_PATTERN.search(text)
    if not match:
        return None, now
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    total = minutes * 60 + seconds
    if total > 180:
        return None, now
    return total, now


def load_robot_model():
    if get_model is None:
        raise RuntimeError(
            "The Python environment running the backend cannot import `inference`. "
            "Run uvicorn from the same environment where `backend/testVid.py` works, "
            "or install the Roboflow `inference` package there."
        )
    return get_model(model_id=ROBOT_MODEL_ID, api_key=ROBOFLOW_API_KEY)


def fuse_field_detections(detections: list[dict]) -> list[dict]:
    if not detections:
        return []

    fused: list[dict] = []

    adjacency: list[set[int]] = [set() for _ in detections]
    field_points = [
        np.array(detection["field_point"], dtype=np.float32)
        for detection in detections
    ]

    for index in range(len(detections)):
        for candidate_index in range(index + 1, len(detections)):
            distance = float(np.linalg.norm(field_points[index] - field_points[candidate_index]))
            if distance <= MERGE_DISTANCE_IN:
                adjacency[index].add(candidate_index)
                adjacency[candidate_index].add(index)

    used = [False] * len(detections)
    for index in range(len(detections)):
        if used[index]:
            continue

        cluster: list[int] = []
        stack = [index]
        used[index] = True
        while stack:
            current = stack.pop()
            cluster.append(current)
            for neighbor in adjacency[current]:
                if used[neighbor]:
                    continue
                used[neighbor] = True
                stack.append(neighbor)

        cluster_detections = [detections[item] for item in cluster]
        cluster_points = np.array([item["field_point"] for item in cluster_detections], dtype=np.float32)
        weights = np.array(
            [max(float(item["confidence"]), 1e-3) for item in cluster_detections],
            dtype=np.float32,
        )
        weighted_point = np.average(cluster_points, axis=0, weights=weights)
        primary_detection = max(cluster_detections, key=lambda item: float(item["confidence"]))
        fused.append(
            {
                "field_point": weighted_point.tolist(),
                "confidence": float(max(item["confidence"] for item in cluster_detections)),
                "view": primary_detection["view"],
                "source_views": sorted({item["view"] for item in cluster_detections}),
                "source_detection_indices": sorted(cluster),
            }
        )

    return fused


def is_reasonable_field_point(field_point: np.ndarray) -> bool:
    margin_x = 36.0
    margin_y = 36.0
    min_x = -(FIELD_WIDTH_IN / 2.0) - margin_x
    max_x = (FIELD_WIDTH_IN / 2.0) + margin_x
    min_y = -(FIELD_HEIGHT_IN / 2.0) - margin_y
    max_y = (FIELD_HEIGHT_IN / 2.0) + margin_y
    return min_x <= float(field_point[0]) <= max_x and min_y <= float(field_point[1]) <= max_y


def build_image_track_observations(tracks: list, projected_detections: list[dict]) -> list[tuple[Any, ImageTrackObservation]]:
    observations: list[tuple[Any, ImageTrackObservation]] = []
    for track in tracks:
        candidate_indices = [
            index
            for index in track.source_detection_indices
            if 0 <= index < len(projected_detections)
        ]
        if not candidate_indices:
            continue

        best_index = max(candidate_indices, key=lambda index: projected_detections[index]["confidence"])
        best_detection = projected_detections[best_index]
        bbox = best_detection.get("bbox")
        image_anchor = best_detection.get("image_anchor")
        if bbox is None or image_anchor is None:
            continue

        observations.append(
            (
                track,
                ImageTrackObservation(
                    track_id=track.track_id,
                    view=best_detection["view"],
                    bbox=[float(value) for value in bbox],
                    image_anchor=[float(value) for value in image_anchor],
                    field_point=[float(value) for value in best_detection["field_point"]],
                    confidence=float(best_detection["confidence"]),
                    source_detection_indices=list(candidate_indices),
                ),
            )
        )
    return observations


def reconcile_detection_track_ids(
    observed_tracks: list[tuple[Any, ImageTrackObservation]],
    image_tracker: ImageTrackerManager,
) -> None:
    if not image_tracker.enabled:
        return

    reserved_track_ids: set[int] = set()
    for track, observation in sorted(observed_tracks, key=lambda item: item[1].confidence, reverse=True):
        if observation.track_id in image_tracker.active_track_ids():
            reserved_track_ids.add(observation.track_id)
            continue

        matched_track_id = image_tracker.match_existing_track_id(
            observation.view,
            observation.bbox,
            reserved_track_ids,
        )
        if matched_track_id is not None:
            track.track_id = matched_track_id
            observation.track_id = matched_track_id
        reserved_track_ids.add(observation.track_id)


def to_global_detection_indices(local_indices: list[int], projected_detections: list[dict]) -> list[int]:
    resolved = []
    for index in local_indices:
        if 0 <= index < len(projected_detections):
            global_index = projected_detections[index].get("global_detection_index")
            if global_index is not None:
                resolved.append(int(global_index))
    return sorted(set(resolved))


def build_topdown_snapshot(tracks: list[TrackRecord], output_path: Path) -> None:
    width = 1280
    height = 640
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (16, 40, 30)
    padding = 80
    scale = min((width - padding * 2) / FIELD_WIDTH_IN, (height - padding * 2) / FIELD_HEIGHT_IN)
    frame_width = FIELD_WIDTH_IN * scale
    frame_height = FIELD_HEIGHT_IN * scale
    frame_left = int(round((width - frame_width) / 2.0))
    frame_top = int(round((height - frame_height) / 2.0))
    frame_right = int(round(frame_left + frame_width))
    frame_bottom = int(round(frame_top + frame_height))

    if FIELD_IMAGE_PATH.exists():
        field_image = cv2.imread(str(FIELD_IMAGE_PATH), cv2.IMREAD_UNCHANGED)
        if field_image is not None and field_image.shape[-1] == 4:
            image_width = int(round(frame_width * FIELD_IMAGE_SCALE_X))
            image_height = int(round(frame_height * FIELD_IMAGE_SCALE_Y))
            image_left = int(round(frame_left + (frame_width - image_width) / 2.0))
            image_top = int(round(frame_top + (frame_height - image_height) / 2.0))
            resized = cv2.resize(field_image, (image_width, image_height), interpolation=cv2.INTER_AREA)
            clipped_left = max(image_left, 0)
            clipped_top = max(image_top, 0)
            clipped_right = min(image_left + image_width, width)
            clipped_bottom = min(image_top + image_height, height)
            if clipped_right > clipped_left and clipped_bottom > clipped_top:
                src_left = clipped_left - image_left
                src_top = clipped_top - image_top
                src_right = src_left + (clipped_right - clipped_left)
                src_bottom = src_top + (clipped_bottom - clipped_top)
                overlay = resized[src_top:src_bottom, src_left:src_right, :3].astype(np.float32)
                alpha = (resized[src_top:src_bottom, src_left:src_right, 3:4].astype(np.float32) / 255.0) * 0.82
                base = image[clipped_top:clipped_bottom, clipped_left:clipped_right].astype(np.float32)
                image[clipped_top:clipped_bottom, clipped_left:clipped_right] = np.clip(
                    overlay * alpha + base * (1.0 - alpha),
                    0,
                    255,
                ).astype(np.uint8)

    cv2.rectangle(image, (frame_left, frame_top), (frame_right, frame_bottom), (200, 220, 180), 3)

    for track in tracks[-250:]:
        x = int(round((width / 2.0) + track.x * scale))
        y = int(round((height / 2.0) - track.y * scale))
        cv2.circle(image, (x, y), 8, (0, 185, 255), -1)
        cv2.putText(image, str(track.track_id), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(output_path), image)


def transcode_annotated_video(input_path: Path, output_path: Path) -> Path:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception:
        return input_path


def rebuild_match_tracking(match: MatchRecord, store: TrackingStore) -> MatchRecord:
    calibration_map = calibration_for_match(match)
    tracker = FieldTracker()
    tracking_mode = "bytetrack" if tracker._byte_tracker is not None else "detection_fallback"
    fallback_track_count = 0

    rebuilt_detections: list[DetectionRecord] = []
    rebuilt_tracks: list[TrackRecord] = []

    grouped: dict[tuple[int, float], list[DetectionRecord]] = {}
    for detection in match.detections:
        grouped.setdefault((detection.frame, detection.time), []).append(detection)

    for (frame_id, timestamp) in sorted(grouped.keys(), key=lambda item: (item[0], item[1])):
        frame_items = grouped[(frame_id, timestamp)]
        projected_detections: list[dict] = []

        for detection in frame_items:
            view = calibration_map.get(detection.view)
            if view is None:
                continue

            field_point = project_detection_in_view(
                (float(detection.image_anchor[0]), float(detection.image_anchor[1])),
                view,
            )
            if np.isnan(field_point).any() or not is_reasonable_field_point(field_point):
                continue

            rebuilt = DetectionRecord(
                frame=detection.frame,
                time=detection.time,
                view=detection.view,
                source_confidence=detection.source_confidence,
                image_anchor=detection.image_anchor,
                field_point=[float(field_point[0]), float(field_point[1])],
                bbox=detection.bbox,
            )
            rebuilt_detections.append(rebuilt)
            projected_detections.append(
                {
                    "view": rebuilt.view,
                    "field_point": rebuilt.field_point,
                    "confidence": rebuilt.source_confidence,
                    "image_anchor": rebuilt.image_anchor,
                    "bbox": rebuilt.bbox,
                    "global_detection_index": len(rebuilt_detections) - 1,
                }
            )

        fused = fuse_field_detections(projected_detections)
        online_tracks = tracker.update(fused)
        if fused and not online_tracks:
            fallback_track_count += 1
            online_tracks = tracker._update_with_fallback(fused)
            tracking_mode = "detection_fallback"
        elif tracker._byte_tracker is not None and online_tracks:
            tracking_mode = "bytetrack"

        for track in online_tracks:
            global_detection_indices = to_global_detection_indices(track.source_detection_indices, projected_detections)
            rebuilt_tracks.append(
                TrackRecord(
                    frame=frame_id,
                    time=timestamp,
                    track_id=track.track_id,
                    x=track.x,
                    y=track.y,
                    confidence=track.confidence,
                    source_views=track.source_views,
                    source_detection_indices=global_detection_indices,
                )
            )

    match.detections = rebuilt_detections
    match.tracks = rebuilt_tracks
    match.metadata["track_count"] = len({track.track_id for track in match.tracks})
    match.metadata["detection_count"] = len(match.detections)
    match.metadata["tracking_mode"] = tracking_mode
    match.metadata["tracker_fallback_frames"] = fallback_track_count
    match.metadata["updated_at"] = time.time()
    match.debug["tracking"] = {
        "mode": tracking_mode,
        "fallback_frames": fallback_track_count,
        "projected_detection_count": len(match.detections),
        "last_reprojection_at": time.time(),
    }

    artifact_dir = store.create_match_artifact_dir(match.id)
    topdown_path = artifact_dir / "topdown.png"
    build_topdown_snapshot(match.tracks, topdown_path)
    match.artifacts.topdown_replay = f"/artifacts/{match.id}/{topdown_path.name}"
    match.artifacts.debug_snapshot = match.artifacts.topdown_replay
    match.artifacts.calibration_preview = match.artifacts.topdown_replay
    store.save_match(match)
    return match


def process_job(job: JobRecord, store: TrackingStore) -> MatchRecord:
    job.status = "running"
    store.save_job(job)
    job = store.append_job_log(job.id, "Resolving source.")
    resolved = resolve_source(job.source)
    source_ready_message = f"Source ready: {resolved.display_name}"
    if resolved.stream_height is not None:
        source_ready_message += f" ({resolved.stream_height}p stream selected)"
    job = store.append_job_log(job.id, source_ready_message)
    model = load_robot_model()
    job = store.append_job_log(job.id, f"Loaded robot model {ROBOT_MODEL_ID}")
    existing_match: MatchRecord | None = None
    if job.match_id:
        try:
            existing_match = store.load_match(job.match_id)
        except FileNotFoundError:
            existing_match = None

    if existing_match is not None:
        match = existing_match
        match.metadata["status"] = "processing"
        match.metadata["processing"] = True
        match.metadata["fps"] = TARGET_FPS
        match.source.update(
            {
                "source_name": job.source.source_name,
                "source_url": resolved.source_url,
                "stored_path": job.source.stored_path,
            }
        )
        match.detections = []
        match.tracks = []
        match.debug = {"stages": []}
    else:
        match_id = uuid.uuid4().hex
        calibration_preset = None
        if job.source.calibration_preset_id:
            calibration_preset = store.load_calibration_preset(job.source.calibration_preset_id)
        calibration = calibration_preset.calibration if calibration_preset is not None else load_default_calibration()
        match = MatchRecord(
            id=match_id,
            created_at=time.time(),
            updated_at=time.time(),
            metadata={
                "display_name": job.source.requested_match_name or Path(job.source.source_name).stem,
                "status": "processing",
                "fps": TARGET_FPS,
                "source_kind": job.source.source_kind,
                "start_mode": "ocr_gated" if job.source.source_kind == "watchbot" else "immediate",
                "calibration_preset_id": calibration_preset.id if calibration_preset is not None else None,
                "calibration_preset_name": calibration_preset.name if calibration_preset is not None else None,
            },
            source={
                "source_name": job.source.source_name,
                "source_url": resolved.source_url,
                "stored_path": job.source.stored_path,
            },
            calibration=calibration,
            artifacts=MatchArtifactSet(),
            debug={"stages": []},
        )

    artifact_dir = store.create_match_artifact_dir(match.id)
    copied_source = store.copy_into_artifacts(job.source.stored_path, match.id, "source") if job.source.stored_path else None
    if copied_source:
        match.artifacts.source_video = copied_source
    elif match.artifacts.source_video is None and resolved.video_path.startswith("http"):
        match.artifacts.source_video = resolved.video_path

    cap = cv2.VideoCapture(resolved.video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the source video.")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    frame_interval = max(1.0 / TARGET_FPS, 0.001)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    annotated_path = artifact_dir / "annotated.mp4"
    writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (width, height))

    tracker = FieldTracker()
    image_tracker = ImageTrackerManager()
    calibration_map = calibration_for_match(match)
    tracking_mode = "bytetrack" if tracker._byte_tracker is not None else "detection_fallback"
    if image_tracker.enabled:
        tracking_mode = f"{tracking_mode}+{(image_tracker.algorithm or 'opencv').lower()}_image_fallback"
    fallback_track_count = 0
    image_fallback_frame_count = 0

    frame_id = 0
    last_processed_time = -frame_interval
    last_ocr_time = 0.0
    last_progress_log_time = time.time()
    last_progress_log_frame = 0
    use_ocr_gating = job.source.source_kind == "watchbot"
    match_active = not use_ocr_gating
    seen_timer = not use_ocr_gating

    if use_ocr_gating:
        job = store.append_job_log(job.id, "Using OCR match-start detection for this source.")
    else:
        source_label = {
            "upload": "Upload",
            "youtube": "YouTube",
            "watchbot": "Watchbot",
        }.get(job.source.source_kind, "Source")
        job = store.append_job_log(job.id, f"{source_label} source detected: starting processing at frame 0 and skipping OCR gating.")
    if image_tracker.enabled:
        job = store.append_job_log(
            job.id,
            f"Image fallback tracker enabled with OpenCV {image_tracker.algorithm} (requested {MOTION_TRACKER_ALGORITHM.upper()}).",
        )
    else:
        job = store.append_job_log(job.id, "Image fallback tracker unavailable in this OpenCV build.", level="warning")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        timer_value = None

        if use_ocr_gating:
            timer_value, last_ocr_time = read_match_timer(frame, last_ocr_time)
            if timer_value is not None:
                seen_timer = True
                if timer_value > 0:
                    match_active = True
                elif match_active:
                    break

        if timestamp - last_processed_time < frame_interval:
            continue
        last_processed_time = timestamp

        if seen_timer and not match_active:
            continue

        inference_result = model.infer(frame)[0]
        predictions = getattr(inference_result, "predictions", [])

        projected_detections: list[dict] = []
        boxes_to_draw: list[dict] = []

        for detection_index, prediction in enumerate(predictions):
            center_x = float(getattr(prediction, "x", 0.0))
            center_y = float(getattr(prediction, "y", 0.0))
            width_px = float(getattr(prediction, "width", 0.0))
            height_px = float(getattr(prediction, "height", 0.0))
            confidence = float(getattr(prediction, "confidence", 0.0))

            x1 = center_x - (width_px / 2.0)
            y1 = center_y - (height_px / 2.0)
            x2 = center_x + (width_px / 2.0)
            y2 = center_y + (height_px / 2.0)
            bbox = [x1, y1, x2, y2]
            boxes_to_draw.append(
                {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": confidence,
                    "global_detection_index": None,
                }
            )

            point = (center_x, center_y)
            projected_candidates = [
                (view, field_point)
                for view, field_point in project_detection_candidates(point, calibration_map)
                if not np.isnan(field_point).any() and is_reasonable_field_point(field_point)
            ]
            if not projected_candidates:
                continue
            for candidate_index, (view, field_point) in enumerate(projected_candidates):
                record = DetectionRecord(
                    frame=frame_id,
                    time=timestamp,
                    view=view,
                    source_confidence=confidence,
                    image_anchor=[center_x, center_y],
                    field_point=[float(field_point[0]), float(field_point[1])],
                    bbox=[float(v) for v in bbox] if bbox else None,
                )
                match.detections.append(record)
                global_detection_index = len(match.detections) - 1
                if candidate_index == 0:
                    boxes_to_draw[-1]["global_detection_index"] = global_detection_index
                projected_detections.append(
                    {
                        "view": view,
                        "field_point": record.field_point,
                        "confidence": confidence,
                        "bbox": record.bbox,
                        "image_anchor": record.image_anchor,
                        "global_detection_index": global_detection_index,
                    }
                )

        fused = fuse_field_detections(projected_detections)
        online_tracks = tracker.update(fused)
        if fused and not online_tracks:
            fallback_track_count += 1
            online_tracks = tracker._update_with_fallback(fused)
            tracking_mode = "detection_fallback"
        elif tracker._byte_tracker is not None and online_tracks:
            tracking_mode = "bytetrack"
        if image_tracker.enabled and ("image_fallback" not in tracking_mode):
            tracking_mode = f"{tracking_mode}+{(image_tracker.algorithm or 'opencv').lower()}_image_fallback"

        tracked_observations = build_image_track_observations(online_tracks, projected_detections)
        reconcile_detection_track_ids(tracked_observations, image_tracker)
        image_tracker.observe_detections(frame, [observation for _, observation in tracked_observations])

        visible_track_ids = {observation.track_id for _, observation in tracked_observations}
        image_fallback_tracks = [
            fallback
            for fallback in image_tracker.track_missing(frame, visible_track_ids, calibration_map)
            if is_reasonable_field_point(np.array(fallback.field_point, dtype=np.float32))
        ]
        if image_fallback_tracks:
            image_fallback_frame_count += 1

        track_observation_by_id = {observation.track_id: observation for _, observation in tracked_observations}
        for track in online_tracks:
            observation = track_observation_by_id.get(track.track_id)
            global_detection_indices = to_global_detection_indices(track.source_detection_indices, projected_detections)
            match.tracks.append(
                TrackRecord(
                    frame=frame_id,
                    time=timestamp,
                    track_id=track.track_id,
                    x=track.x,
                    y=track.y,
                    confidence=track.confidence,
                    source_views=track.source_views,
                    source_detection_indices=global_detection_indices,
                    tracking_source="detection",
                    image_view=observation.view if observation is not None else None,
                    image_anchor=observation.image_anchor if observation is not None else None,
                    image_bbox=observation.bbox if observation is not None else None,
                )
            )
        for fallback in image_fallback_tracks:
            match.tracks.append(
                TrackRecord(
                    frame=frame_id,
                    time=timestamp,
                    track_id=fallback.track_id,
                    x=float(fallback.field_point[0]),
                    y=float(fallback.field_point[1]),
                    confidence=fallback.confidence,
                    source_views=[fallback.view],
                    source_detection_indices=[],
                    tracking_source="image_tracker",
                    image_view=fallback.view,
                    image_anchor=fallback.image_anchor,
                    image_bbox=fallback.bbox,
                )
            )

        detection_to_track_ids: dict[int, list[int]] = {}
        fused_track_ids: set[int] = set()
        for track in online_tracks:
            if len(set(track.source_views)) > 1:
                fused_track_ids.add(track.track_id)
            for detection_index in to_global_detection_indices(track.source_detection_indices, projected_detections):
                detection_to_track_ids.setdefault(detection_index, []).append(track.track_id)

        debug_frame = frame.copy()
        for box in boxes_to_draw:
            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]
            confidence = box["confidence"]
            detection_index = box["global_detection_index"]
            track_ids = detection_to_track_ids.get(detection_index, []) if detection_index is not None else []
            label = f"robot {confidence:.2f}"
            color = ANNOTATION_DEFAULT_COLOR
            if track_ids:
                primary_track_id = track_ids[0]
                is_fused = primary_track_id in fused_track_ids
                color = ANNOTATION_FUSED_COLOR if is_fused else ANNOTATION_DEFAULT_COLOR
                label = f"T{primary_track_id}{' FUSED' if is_fused else ''} {confidence:.2f}"

            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                debug_frame,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        for fallback in image_fallback_tracks:
            x1, y1, x2, y2 = [int(round(value)) for value in fallback.bbox]
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), ANNOTATION_FALLBACK_COLOR, 2)
            cv2.putText(
                debug_frame,
                f"T{fallback.track_id} TRACK",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ANNOTATION_FALLBACK_COLOR,
                1,
            )
        for track in online_tracks:
            track_color = ANNOTATION_FUSED_COLOR if track.track_id in fused_track_ids else ANNOTATION_DEFAULT_COLOR
            cv2.putText(
                debug_frame,
                f"T{track.track_id}{' FUSED' if track.track_id in fused_track_ids else ''}",
                (20, 40 + (track.track_id % 10) * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                track_color,
                2,
            )
        cv2.putText(
            debug_frame,
            f"dets:{len(boxes_to_draw)} projected:{len(projected_detections)} tracks:{len(online_tracks)} fused:{len(fused_track_ids)} motion:{len(image_fallback_tracks)}",
            (20, height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(debug_frame)

        now = time.time()
        if frame_id - last_progress_log_frame >= 150 or now - last_progress_log_time >= 10.0:
            job = store.append_job_log(
                job.id,
                (
                    f"Processed frame {frame_id} at {timestamp:.1f}s "
                    f"(detections={len(match.detections)}, tracks={len(match.tracks)})."
                ),
            )
            last_progress_log_time = now
            last_progress_log_frame = frame_id

    cap.release()
    writer.release()

    browser_annotated_path = transcode_annotated_video(annotated_path, artifact_dir / "annotated_browser.mp4")
    match.artifacts.annotated_video = f"/artifacts/{match.id}/{browser_annotated_path.name}"
    if match.artifacts.source_video is None:
        match.artifacts.source_video = match.artifacts.annotated_video

    topdown_path = artifact_dir / "topdown.png"
    build_topdown_snapshot(match.tracks, topdown_path)
    match.artifacts.topdown_replay = f"/artifacts/{match.id}/{topdown_path.name}"
    match.artifacts.debug_snapshot = match.artifacts.topdown_replay
    match.artifacts.calibration_preview = match.artifacts.topdown_replay
    match.metadata["status"] = "complete"
    match.metadata["processing"] = False
    match.metadata["completed_at"] = time.time()
    match.metadata["frame_count"] = frame_id
    match.metadata["track_count"] = len({track.track_id for track in match.tracks})
    match.metadata["detection_count"] = len(match.detections)
    match.metadata["source_fps"] = source_fps
    match.metadata["tracking_mode"] = tracking_mode
    match.metadata["tracker_fallback_frames"] = fallback_track_count
    match.metadata["image_tracker_fallback_frames"] = image_fallback_frame_count
    match.metadata["image_tracker_algorithm"] = image_tracker.algorithm
    match.debug["stages"] = [
        "source_ingest",
        "match_detection",
        "robot_detection",
        "manual_fallback_calibration",
        "cross_view_fusion",
        "field_space_tracking",
        "image_space_fallback_tracking",
        "artifact_export",
    ]
    match.debug["tracking"] = {
        "mode": tracking_mode,
        "fallback_frames": fallback_track_count,
        "image_fallback_frames": image_fallback_frame_count,
        "image_tracker_algorithm": image_tracker.algorithm,
        "projected_detection_count": len(match.detections),
    }
    store.save_match(match)

    job.status = "completed"
    job.match_id = match.id
    store.save_job(job)
    job = store.append_job_log(
        job.id,
        f"Match processed with {match.metadata['track_count']} tracks using {tracking_mode}.",
    )
    return match


def capture_stream_segment(stream_url: str, output_path: str, stop_flag) -> tuple[bool, str]:
    resolved = resolve_source(
        SourceSubmission(
            source_kind="watchbot",
            source_url=stream_url,
            source_name="watchbot-stream",
        )
    )
    cap = cv2.VideoCapture(resolved.video_path)
    if not cap.isOpened():
        return False, "Could not open watchbot stream."

    writer = None
    last_ocr_time = 0.0
    active = False
    while not stop_flag():
        ret, frame = cap.read()
        if not ret:
            break

        timer_value, last_ocr_time = read_match_timer(frame, last_ocr_time)
        if timer_value is not None and timer_value > 0 and not active:
            active = True
            height, width, _ = frame.shape
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (width, height))
        if active and writer is not None:
            writer.write(frame)
        if active and timer_value == 0:
            break

    cap.release()
    if writer is not None:
        writer.release()
        return True, output_path
    return False, "No match segment was detected."
