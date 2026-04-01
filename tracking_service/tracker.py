from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import TRACK_BOX_SIZE_IN, TRACKER_FRAME_RATE, TRACK_DEDUPE_DISTANCE_IN

try:
    from byte_tracker import BYTETracker as NativeBYTETracker  # type: ignore
except Exception:
    NativeBYTETracker = None


@dataclass
class TrackState:
    track_id: int
    x: float
    y: float
    confidence: float
    source_views: list[str] = field(default_factory=list)
    source_detection_indices: list[int] = field(default_factory=list)


class FieldTracker:
    def __init__(self) -> None:
        self._fallback_tracks: dict[int, tuple[float, float]] = {}
        self._next_track_id = 1
        self._byte_tracker = None
        if NativeBYTETracker is not None:
            self._byte_tracker = NativeBYTETracker(track_thresh=0.45, match_thresh=0.8, frame_rate=TRACKER_FRAME_RATE)

    def update(self, detections: list[dict]) -> list[TrackState]:
        if self._byte_tracker is not None:
            tracks = self._update_with_byte_track(detections)
            if detections and not tracks:
                tracks = self._update_with_fallback(detections)
        else:
            tracks = self._update_with_fallback(detections)
        return self._dedupe_tracks(tracks)

    def _update_with_byte_track(self, detections: list[dict]) -> list[TrackState]:
        boxes = []
        for detection in detections:
            x = detection["field_point"][0]
            y = detection["field_point"][1]
            size = TRACK_BOX_SIZE_IN / 2
            boxes.append([x - size, y - size, x + size, y + size, detection["confidence"]])
        if not boxes:
            online_tracks = self._byte_tracker.update(np.empty((0, 5), dtype=np.float32))
        else:
            online_tracks = self._byte_tracker.update(np.array(boxes, dtype=np.float32))

        tracks: list[TrackState] = []
        for track in online_tracks:
            center_x = float((track.x1 + track.x2) / 2)
            center_y = float((track.y1 + track.y2) / 2)
            track_id = int(track.track_id)
            matching_indices = [
                index
                for index, detection in enumerate(detections)
                if np.linalg.norm(np.array(detection["field_point"]) - np.array([center_x, center_y])) < TRACK_BOX_SIZE_IN
            ]
            expanded_detection_indices = sorted(
                {
                    source_index
                    for index in matching_indices
                    for source_index in detections[index].get("source_detection_indices", [index])
                }
            )
            source_views = sorted({detections[index]["view"] for index in matching_indices})
            expanded_source_views = sorted(
                {
                    view_name
                    for index in matching_indices
                    for view_name in detections[index].get("source_views", [detections[index]["view"]])
                }
            )
            tracks.append(
                TrackState(
                    track_id=track_id,
                    x=center_x,
                    y=center_y,
                    confidence=float(max((detections[index]["confidence"] for index in matching_indices), default=0.0)),
                    source_views=expanded_source_views or source_views,
                    source_detection_indices=expanded_detection_indices,
                )
            )
        return tracks

    def _update_with_fallback(self, detections: list[dict]) -> list[TrackState]:
        if not detections:
            return []

        updated: dict[int, tuple[float, float]] = {}
        results: list[TrackState] = []

        for index, detection in enumerate(detections):
            point = np.array(detection["field_point"], dtype=np.float32)
            best_track = None
            best_distance = float("inf")
            for track_id, previous in self._fallback_tracks.items():
                distance = float(np.linalg.norm(point - np.array(previous, dtype=np.float32)))
                if distance < best_distance and distance < 36:
                    best_distance = distance
                    best_track = track_id
            if best_track is None:
                best_track = self._next_track_id
                self._next_track_id += 1
            updated[best_track] = (float(point[0]), float(point[1]))
            results.append(
                TrackState(
                    track_id=best_track,
                    x=float(point[0]),
                    y=float(point[1]),
                    confidence=float(detection["confidence"]),
                    source_views=list(detection.get("source_views", [detection["view"]])),
                    source_detection_indices=list(detection.get("source_detection_indices", [index])),
                )
            )

        self._fallback_tracks = updated
        return results

    def _dedupe_tracks(self, tracks: list[TrackState]) -> list[TrackState]:
        deduped: list[TrackState] = []
        for track in sorted(tracks, key=lambda item: item.confidence, reverse=True):
            is_duplicate = False
            for kept in deduped:
                distance = float(np.linalg.norm(np.array([track.x, track.y]) - np.array([kept.x, kept.y])))
                if distance <= TRACK_DEDUPE_DISTANCE_IN:
                    kept.source_views = sorted(set(kept.source_views + track.source_views))
                    kept.source_detection_indices = sorted(set(kept.source_detection_indices + track.source_detection_indices))
                    kept.confidence = max(kept.confidence, track.confidence)
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(track)
        return deduped
