from __future__ import annotations

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent
DATA_ROOT = BACKEND_ROOT / "data"
UPLOAD_ROOT = DATA_ROOT / "uploads"
MATCH_ROOT = DATA_ROOT / "matches"
JOB_ROOT = DATA_ROOT / "jobs"
WATCHBOT_ROOT = DATA_ROOT / "watchbot"
ARTIFACT_ROOT = DATA_ROOT / "artifacts"
CALIBRATION_PRESET_ROOT = DATA_ROOT / "calibrations"
FUEL_CALIBRATION_PRESET_ROOT = DATA_ROOT / "fuel_calibrations"


def _resolve_default_calibration_file() -> Path:
    configured = os.getenv("TRACKING_CALIBRATION_FILE", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        BACKEND_ROOT / "field_calibration.json",
        REPO_ROOT / "backend" / "field_calibration.json",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    return (BACKEND_ROOT / "field_calibration.json").resolve()


DEFAULT_CALIBRATION_FILE = _resolve_default_calibration_file()


def _resolve_fuel_density_map_root() -> Path:
    configured = os.getenv("FUEL_DENSITY_MAP_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (REPO_ROOT / "fuel-density-map").resolve()


def _resolve_fuel_processor_script(root: Path) -> Path:
    configured = os.getenv("FUEL_PROCESSOR_SCRIPT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (root / "processor_cli.py").resolve()


def _resolve_fuel_field_image_path(root: Path) -> Path:
    configured = os.getenv("FUEL_FIELD_IMAGE_PATH", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        root / "webui" / "public" / "assets" / "rebuilt-field.png",
        REPO_ROOT / "frc-strat-frontend" / "public" / "assets" / "rebuilt-field.png",
        REPO_ROOT / "frontend" / "mais" / "public" / "assets" / "rebuilt-field.png",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / "frc-strat-frontend" / "public" / "assets" / "rebuilt-field.png").resolve()


def _resolve_watchbot_field_template_root() -> Path:
    configured = os.getenv("WATCHBOT_FIELD_TEMPLATE_ROOT", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        BACKEND_ROOT / "media" / "field_dataset",
        REPO_ROOT / "backend" / "media" / "field_dataset",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / "backend" / "media" / "field_dataset").resolve()


FUEL_DENSITY_MAP_ROOT = _resolve_fuel_density_map_root()
FUEL_PROCESSOR_SCRIPT = _resolve_fuel_processor_script(FUEL_DENSITY_MAP_ROOT)
FUEL_FIELD_IMAGE_PATH = _resolve_fuel_field_image_path(FUEL_DENSITY_MAP_ROOT)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "EocOtkmJYz5Hlu7QosUB")
ROBOT_MODEL_ID = os.getenv("ROBOT_MODEL_ID", "bumpers-detection-v2/3")
TARGET_FPS = float(os.getenv("TRACKING_TARGET_FPS", "10"))
FIELD_WIDTH_IN = 651.25
FIELD_HEIGHT_IN = 323.25
MERGE_DISTANCE_IN = float(os.getenv("TRACKING_MERGE_DISTANCE_IN", "36"))
TRACK_BOX_SIZE_IN = float(os.getenv("TRACKING_BOX_SIZE_IN", "12"))
TRACKER_FRAME_RATE = int(os.getenv("TRACKING_FRAME_RATE", "10"))
TRACK_DEDUPE_DISTANCE_IN = float(os.getenv("TRACKING_TRACK_DEDUPE_DISTANCE_IN", "18"))
MOTION_TRACKER_ALGORITHM = os.getenv("TRACKING_MOTION_TRACKER_ALGO", "CSRT")
MOTION_TRACKER_MAX_FALLBACK_FRAMES = int(os.getenv("TRACKING_MOTION_MAX_FALLBACK_FRAMES", "12"))
MOTION_TRACKER_MAX_ASSOCIATION_DISTANCE_PX = float(os.getenv("TRACKING_MOTION_ASSOCIATION_DISTANCE_PX", "140"))
MOTION_TRACKER_MIN_IOU = float(os.getenv("TRACKING_MOTION_MIN_IOU", "0.08"))
MOTION_TRACKER_CONFIDENCE_DECAY = float(os.getenv("TRACKING_MOTION_CONFIDENCE_DECAY", "0.9"))
MOTION_TRACKER_MIN_CONFIDENCE = float(os.getenv("TRACKING_MOTION_MIN_CONFIDENCE", "0.15"))
MOTION_TRACKER_MIN_BOX_SIZE_PX = float(os.getenv("TRACKING_MOTION_MIN_BOX_SIZE_PX", "10"))
WATCHBOT_TEMPLATE_ANALYSIS_FPS = float(os.getenv("WATCHBOT_TEMPLATE_ANALYSIS_FPS", "6.0"))
WATCHBOT_TEMPLATE_START_FRAMES = int(os.getenv("WATCHBOT_TEMPLATE_START_FRAMES", "3"))
WATCHBOT_MATCH_DURATION_SECONDS = float(os.getenv("WATCHBOT_MATCH_DURATION_SECONDS", "160.0"))
WATCHBOT_FIELD_TEMPLATE_MARGIN = float(os.getenv("WATCHBOT_FIELD_TEMPLATE_MARGIN", "0.02"))
WATCHBOT_FIELD_TEMPLATE_ROOT = _resolve_watchbot_field_template_root()
WATCHBOT_FIELD_TEMPLATE_SIZE = (96, 54)
WATCHBOT_FIELD_TEMPLATE_ROIS = (
    (160, 90, 320, 240),
    (0, 270, 320, 180),
    (320, 180, 960, 240),
)


def ensure_data_dirs() -> None:
    for path in (
        DATA_ROOT,
        UPLOAD_ROOT,
        MATCH_ROOT,
        JOB_ROOT,
        WATCHBOT_ROOT,
        ARTIFACT_ROOT,
        CALIBRATION_PRESET_ROOT,
        FUEL_CALIBRATION_PRESET_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
