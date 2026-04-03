from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


CalibrationMode = Literal["auto_calibration", "manual_override", "blended", "manual_fallback"]
CalibrationSource = Literal["manual", "apriltag_pose_seeded", "apriltag_homography"]
JobStatus = Literal["queued", "running", "completed", "failed"]
SourceKind = Literal["upload", "youtube", "watchbot"]
ViewName = Literal["left", "main", "right"]
FuelAnalysisStatus = Literal["idle", "ready", "processing", "completed", "error"]


class SourceSubmission(BaseModel):
    source_kind: SourceKind
    source_url: Optional[str] = None
    source_name: str
    stored_path: Optional[str] = None
    requested_match_name: Optional[str] = None
    event_key: Optional[str] = None
    calibration_preset_id: Optional[str] = None


class JobLogEntry(BaseModel):
    timestamp: float
    level: Literal["info", "warning", "error"] = "info"
    message: str


class JobRecord(BaseModel):
    id: str
    created_at: float
    updated_at: float
    status: JobStatus = "queued"
    source: SourceSubmission
    logs: list[JobLogEntry] = Field(default_factory=list)
    match_id: Optional[str] = None
    error: Optional[str] = None


class FieldLandmark(BaseModel):
    name: str
    image_point: list[float]
    field_point: list[float]
    confidence: float
    source_type: Literal["manual", "apriltag"] = "manual"
    tag_id: Optional[int] = None
    corner_index: Optional[int] = None
    decision_margin: Optional[float] = None


class ViewCalibration(BaseModel):
    view: ViewName
    roi: list[float]
    homography: list[list[float]]
    landmarks: list[FieldLandmark] = Field(default_factory=list)
    distortion_strength: float = 0.0
    distortion_x: float = 0.0
    distortion_y: float = 0.0
    reprojection_error: Optional[float] = None
    confidence: float = 0.0
    calibration_source: CalibrationSource = "manual"
    detected_tag_ids: list[int] = Field(default_factory=list)
    pose_debug: dict[str, Any] = Field(default_factory=dict)
    fallback_reason: Optional[str] = None


class CalibrationEnvelope(BaseModel):
    mode: CalibrationMode
    created_at: float
    updated_at: float
    quality_checks: dict[str, Any] = Field(default_factory=dict)
    views: list[ViewCalibration]


class CalibrationPreset(BaseModel):
    id: str
    name: str
    created_at: float
    updated_at: float
    calibration: CalibrationEnvelope


class DetectionRecord(BaseModel):
    frame: int
    time: float
    view: ViewName
    source_confidence: float
    image_anchor: list[float]
    field_point: list[float]
    bbox: Optional[list[float]] = None


class TrackRecord(BaseModel):
    frame: int
    time: float
    track_id: int
    x: float
    y: float
    confidence: float
    source_views: list[ViewName] = Field(default_factory=list)
    source_detection_indices: list[int] = Field(default_factory=list)
    tracking_source: Literal["detection", "image_tracker"] = "detection"
    image_view: Optional[ViewName] = None
    image_anchor: Optional[list[float]] = None
    image_bbox: Optional[list[float]] = None


class MatchArtifactSet(BaseModel):
    source_video: Optional[str] = None
    trimmed_video: Optional[str] = None
    annotated_video: Optional[str] = None
    topdown_replay: Optional[str] = None
    calibration_preview: Optional[str] = None
    debug_snapshot: Optional[str] = None


class FuelCalibration(BaseModel):
    ground_quad: Optional[list[list[float]]] = None
    left_wall_quad: Optional[list[list[float]]] = None
    right_wall_quad: Optional[list[list[float]]] = None
    fuel_base_color: list[int] = Field(default_factory=lambda: [255, 255, 0])
    updated_at: Optional[float] = None


class FuelArtifactSet(BaseModel):
    overlay_image: Optional[str] = None
    overlay_transparent_image: Optional[str] = None
    overlay_video: Optional[str] = None
    raw_data: Optional[str] = None
    field_map: Optional[str] = None
    air_profile: Optional[str] = None
    stats_file: Optional[str] = None
    process_log: Optional[str] = None


class FuelProcessingProgress(BaseModel):
    phase: str
    current: int
    total: int
    started_at: float
    updated_at: float


class FuelAnalysisRecord(BaseModel):
    status: FuelAnalysisStatus = "idle"
    artifacts: FuelArtifactSet = Field(default_factory=FuelArtifactSet)
    stats: dict[str, Any] = Field(default_factory=dict)
    last_error: Optional[str] = None
    processing_progress: Optional[FuelProcessingProgress] = None
    updated_at: Optional[float] = None


class MatchLabelUpdate(BaseModel):
    labels: dict[str, str]


class MatchRecord(BaseModel):
    id: str
    created_at: float
    updated_at: float
    metadata: dict[str, Any]
    source: dict[str, Any]
    calibration: CalibrationEnvelope
    detections: list[DetectionRecord] = Field(default_factory=list)
    tracks: list[TrackRecord] = Field(default_factory=list)
    artifacts: MatchArtifactSet = Field(default_factory=MatchArtifactSet)
    fuel_calibration: FuelCalibration = Field(default_factory=FuelCalibration)
    fuel_analysis: FuelAnalysisRecord = Field(default_factory=FuelAnalysisRecord)
    labels: dict[str, str] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


class WatchbotState(BaseModel):
    active: bool = False
    stream_url: Optional[str] = None
    started_at: Optional[float] = None
    updated_at: Optional[float] = None
    last_message: Optional[str] = None
    capture_directory: Optional[str] = None
