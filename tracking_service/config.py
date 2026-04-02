from __future__ import annotations

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = BACKEND_ROOT / "data"
UPLOAD_ROOT = DATA_ROOT / "uploads"
MATCH_ROOT = DATA_ROOT / "matches"
JOB_ROOT = DATA_ROOT / "jobs"
WATCHBOT_ROOT = DATA_ROOT / "watchbot"
ARTIFACT_ROOT = DATA_ROOT / "artifacts"
DEFAULT_CALIBRATION_FILE = BACKEND_ROOT / "field_calibration.json"
FUEL_DENSITY_MAP_ROOT = Path(os.getenv("FUEL_DENSITY_MAP_ROOT", str(BACKEND_ROOT.parent / "fuel-density-map"))).resolve()
FUEL_PROCESSOR_SCRIPT = FUEL_DENSITY_MAP_ROOT / "processor_cli.py"
FUEL_FIELD_IMAGE_PATH = FUEL_DENSITY_MAP_ROOT / "webui" / "public" / "assets" / "rebuilt-field.png"

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "EocOtkmJYz5Hlu7QosUB")
ROBOT_MODEL_ID = os.getenv("ROBOT_MODEL_ID", "bumpers-detection-v2/3")
TARGET_FPS = float(os.getenv("TRACKING_TARGET_FPS", "10"))
FIELD_WIDTH_IN = 651.25
FIELD_HEIGHT_IN = 323.25
MERGE_DISTANCE_IN = float(os.getenv("TRACKING_MERGE_DISTANCE_IN", "36"))
TRACK_BOX_SIZE_IN = float(os.getenv("TRACKING_BOX_SIZE_IN", "12"))
TRACKER_FRAME_RATE = int(os.getenv("TRACKING_FRAME_RATE", "10"))
OCR_INTERVAL_SECONDS = float(os.getenv("TRACKING_OCR_INTERVAL_SECONDS", "0.5"))
OCR_TIMEOUT_SECONDS = float(os.getenv("TRACKING_OCR_TIMEOUT_SECONDS", "0.75"))
TRACK_DEDUPE_DISTANCE_IN = float(os.getenv("TRACKING_TRACK_DEDUPE_DISTANCE_IN", "18"))
MOTION_TRACKER_ALGORITHM = os.getenv("TRACKING_MOTION_TRACKER_ALGO", "CSRT")
MOTION_TRACKER_MAX_FALLBACK_FRAMES = int(os.getenv("TRACKING_MOTION_MAX_FALLBACK_FRAMES", "12"))
MOTION_TRACKER_MAX_ASSOCIATION_DISTANCE_PX = float(os.getenv("TRACKING_MOTION_ASSOCIATION_DISTANCE_PX", "140"))
MOTION_TRACKER_MIN_IOU = float(os.getenv("TRACKING_MOTION_MIN_IOU", "0.08"))
MOTION_TRACKER_CONFIDENCE_DECAY = float(os.getenv("TRACKING_MOTION_CONFIDENCE_DECAY", "0.9"))
MOTION_TRACKER_MIN_CONFIDENCE = float(os.getenv("TRACKING_MOTION_MIN_CONFIDENCE", "0.15"))
MOTION_TRACKER_MIN_BOX_SIZE_PX = float(os.getenv("TRACKING_MOTION_MIN_BOX_SIZE_PX", "10"))


def ensure_data_dirs() -> None:
    for path in (
    DATA_ROOT,
    UPLOAD_ROOT,
    MATCH_ROOT,
    JOB_ROOT,
    WATCHBOT_ROOT,
    ARTIFACT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
