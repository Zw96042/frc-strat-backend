from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .calibration import inside_roi, project_detection_in_view
from .config import (
    MOTION_TRACKER_ALGORITHM,
    MOTION_TRACKER_CONFIDENCE_DECAY,
    MOTION_TRACKER_MAX_ASSOCIATION_DISTANCE_PX,
    MOTION_TRACKER_MAX_FALLBACK_FRAMES,
    MOTION_TRACKER_MIN_BOX_SIZE_PX,
    MOTION_TRACKER_MIN_CONFIDENCE,
    MOTION_TRACKER_MIN_IOU,
)


TRACKER_CANDIDATES = ("CSRT", "KCF", "MOSSE", "MIL")


@dataclass
class ImageTrackObservation:
    track_id: int
    view: str
    bbox: list[float]
    image_anchor: list[float]
    field_point: list[float]
    confidence: float
    source_detection_indices: list[int]


@dataclass
class ImageTrackFallback:
    track_id: int
    view: str
    bbox: list[float]
    image_anchor: list[float]
    field_point: list[float]
    confidence: float
    fallback_age: int


@dataclass
class _ActiveImageTrack:
    track_id: int
    view: str
    bbox_xywh: tuple[float, float, float, float]
    tracker: Any
    confidence: float
    fallback_age: int = 0


def _get_opencv_attr(path: str) -> Optional[Any]:
    value: Any = cv2
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _available_tracker_algorithm(preferred: str) -> Optional[str]:
    ordered = []
    normalized = (preferred or "").strip().upper()
    if normalized:
        ordered.append(normalized)
    for candidate in TRACKER_CANDIDATES:
        if candidate not in ordered:
            ordered.append(candidate)

    for candidate in ordered:
        for attr_name in (f"Tracker{candidate}_create", f"legacy.Tracker{candidate}_create"):
            factory = _get_opencv_attr(attr_name)
            if callable(factory):
                return candidate
    return None


def _create_tracker(algorithm: str) -> Optional[Any]:
    for attr_name in (f"Tracker{algorithm}_create", f"legacy.Tracker{algorithm}_create"):
        factory = _get_opencv_attr(attr_name)
        if callable(factory):
            return factory()
    return None


def _clamp_bbox_xywh(bbox: tuple[float, float, float, float], frame_shape: tuple[int, ...]) -> Optional[tuple[float, float, float, float]]:
    height, width = frame_shape[:2]
    x, y, w, h = bbox
    x1 = min(max(float(x), 0.0), max(float(width - 1), 0.0))
    y1 = min(max(float(y), 0.0), max(float(height - 1), 0.0))
    x2 = min(max(float(x + w), x1 + 1.0), float(width))
    y2 = min(max(float(y + h), y1 + 1.0), float(height))
    width_px = x2 - x1
    height_px = y2 - y1
    if width_px < MOTION_TRACKER_MIN_BOX_SIZE_PX or height_px < MOTION_TRACKER_MIN_BOX_SIZE_PX:
        return None
    return (x1, y1, width_px, height_px)


def _bbox_xyxy_to_xywh(bbox: list[float], frame_shape: tuple[int, ...]) -> Optional[tuple[float, float, float, float]]:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return _clamp_bbox_xywh((x1, y1, x2 - x1, y2 - y1), frame_shape)


def _bbox_xywh_to_xyxy(bbox: tuple[float, float, float, float]) -> list[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def _bbox_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    union = (aw * ah) + (bw * bh) - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _bbox_center_distance(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    center_a = np.array([ax + (aw / 2.0), ay + (ah / 2.0)], dtype=np.float32)
    center_b = np.array([bx + (bw / 2.0), by + (bh / 2.0)], dtype=np.float32)
    return float(np.linalg.norm(center_a - center_b))


class ImageTrackerManager:
    def __init__(self, preferred_algorithm: str = MOTION_TRACKER_ALGORITHM) -> None:
        self.algorithm = _available_tracker_algorithm(preferred_algorithm)
        self.enabled = self.algorithm is not None
        self._tracks: dict[int, _ActiveImageTrack] = {}

    def active_track_ids(self) -> set[int]:
        return set(self._tracks.keys())

    def match_existing_track_id(self, view: str, bbox: list[float], reserved_track_ids: set[int]) -> Optional[int]:
        if not self.enabled:
            return None

        candidate_bbox = _bbox_xyxy_to_xywh(bbox, (int(1e9), int(1e9), 3))
        if candidate_bbox is None:
            return None

        best_track_id = None
        best_score = float("-inf")
        for track_id, track in self._tracks.items():
            if track_id in reserved_track_ids or track.view != view:
                continue
            iou = _bbox_iou(track.bbox_xywh, candidate_bbox)
            distance = _bbox_center_distance(track.bbox_xywh, candidate_bbox)
            if iou < MOTION_TRACKER_MIN_IOU and distance > MOTION_TRACKER_MAX_ASSOCIATION_DISTANCE_PX:
                continue
            score = (iou * 1000.0) - distance
            if score > best_score:
                best_score = score
                best_track_id = track_id
        return best_track_id

    def observe_detections(self, frame: np.ndarray, observations: list[ImageTrackObservation]) -> None:
        if not self.enabled:
            return

        for observation in observations:
            bbox_xywh = _bbox_xyxy_to_xywh(observation.bbox, frame.shape)
            if bbox_xywh is None:
                continue

            tracker = _create_tracker(self.algorithm or "")
            if tracker is None:
                self.enabled = False
                self._tracks.clear()
                return

            try:
                tracker.init(frame, tuple(int(round(value)) for value in bbox_xywh))
            except Exception:
                continue

            self._tracks[observation.track_id] = _ActiveImageTrack(
                track_id=observation.track_id,
                view=observation.view,
                bbox_xywh=bbox_xywh,
                tracker=tracker,
                confidence=observation.confidence,
                fallback_age=0,
            )

    def track_missing(self, frame: np.ndarray, visible_track_ids: set[int], calibration_map: dict[str, Any]) -> list[ImageTrackFallback]:
        if not self.enabled:
            return []

        fallback_tracks: list[ImageTrackFallback] = []
        stale_track_ids: list[int] = []

        for track_id, active in list(self._tracks.items()):
            if track_id in visible_track_ids:
                continue
            if active.fallback_age >= MOTION_TRACKER_MAX_FALLBACK_FRAMES:
                stale_track_ids.append(track_id)
                continue

            ok, updated_bbox = active.tracker.update(frame)
            if not ok:
                stale_track_ids.append(track_id)
                continue

            clamped_bbox = _clamp_bbox_xywh(tuple(float(value) for value in updated_bbox), frame.shape)
            if clamped_bbox is None:
                stale_track_ids.append(track_id)
                continue

            active.bbox_xywh = clamped_bbox
            active.fallback_age += 1
            active.confidence = max(active.confidence * MOTION_TRACKER_CONFIDENCE_DECAY, MOTION_TRACKER_MIN_CONFIDENCE)

            center_x = clamped_bbox[0] + (clamped_bbox[2] / 2.0)
            center_y = clamped_bbox[1] + (clamped_bbox[3] / 2.0)
            image_anchor = [float(center_x), float(center_y)]
            view = calibration_map.get(active.view)
            if view is None or not inside_roi((image_anchor[0], image_anchor[1]), view.roi):
                stale_track_ids.append(track_id)
                continue

            field_point = project_detection_in_view((image_anchor[0], image_anchor[1]), view)
            if np.isnan(field_point).any():
                stale_track_ids.append(track_id)
                continue

            fallback_tracks.append(
                ImageTrackFallback(
                    track_id=track_id,
                    view=active.view,
                    bbox=_bbox_xywh_to_xyxy(clamped_bbox),
                    image_anchor=image_anchor,
                    field_point=[float(field_point[0]), float(field_point[1])],
                    confidence=float(active.confidence),
                    fallback_age=active.fallback_age,
                )
            )

        for track_id in stale_track_ids:
            self._tracks.pop(track_id, None)

        return fallback_tracks
