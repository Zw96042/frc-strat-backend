from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .calibration import undistort_point
from .config import FIELD_HEIGHT_IN, FIELD_WIDTH_IN
from .schemas import FieldLandmark, ViewCalibration

logger = logging.getLogger(__name__)

APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
APRILTAG_SIZE_M = 0.1651
DEFAULT_CAMERA_HFOV_DEG = 70.0
POSE_REPROJECTION_MAX_ERROR_PX = 10.0
POSE_PRIOR_WEIGHT = 1.0
POSE_PRIOR_MAX_DELTA_IN = 80.0
POSE_PRIOR_CLUSTER_SPREAD_IN = 60.0
POSE_PRIOR_MAX_WEIGHT = 4.0
POSE_PRIOR_PLANAR_CLUSTER_WEIGHT = 10.0
PLANAR_LAYOUT_MAX_RELATIVE_THICKNESS = 0.08
MIN_DETECTED_TAGS = 1
MIN_ACCEPTED_TAGS = 1

# Official 2026 WPILib field layout (AndyMark field), stored locally to avoid runtime network dependence.
# Coordinates are in meters in the WPILib field frame.
FIELD_LENGTH_M = 16.518
FIELD_WIDTH_M = 8.043
TAG_POSES_2026: dict[int, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    1: ((11.863959, 7.411491399999999, 0.889), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    2: ((11.9013986, 4.6247558, 1.12395), (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)),
    3: ((11.2978438, 4.3769534, 1.12395), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    4: ((11.2978438, 4.0213534, 1.12395), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    5: ((11.9013986, 3.417951, 1.12395), (-0.7071067811865475, -0.0, 0.0, 0.7071067811865476)),
    6: ((11.863959, 0.6312154, 0.889), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    7: ((11.9388636, 0.6312154, 0.889), (1.0, 0.0, 0.0, 0.0)),
    8: ((12.2569986, 3.417951, 1.12395), (-0.7071067811865475, -0.0, 0.0, 0.7071067811865476)),
    9: ((12.5051566, 3.6657534, 1.12395), (1.0, 0.0, 0.0, 0.0)),
    10: ((12.5051566, 4.0213534, 1.12395), (1.0, 0.0, 0.0, 0.0)),
    11: ((12.2569986, 4.6247558, 1.12395), (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)),
    12: ((11.9388636, 7.411491399999999, 0.889), (1.0, 0.0, 0.0, 0.0)),
    13: ((16.499332, 7.391907999999999, 0.55245), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    14: ((16.499332, 6.960107999999999, 0.55245), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    15: ((16.4989764, 4.3124882, 0.55245), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    16: ((16.4989764, 3.8806881999999994, 0.55245), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    17: ((4.6490636, 0.6312154, 0.889), (1.0, 0.0, 0.0, 0.0)),
    18: ((4.6115986, 3.417951, 1.12395), (-0.7071067811865475, -0.0, 0.0, 0.7071067811865476)),
    19: ((5.2151534, 3.6657534, 1.12395), (1.0, 0.0, 0.0, 0.0)),
    20: ((5.2151534, 4.0213534, 1.12395), (1.0, 0.0, 0.0, 0.0)),
    21: ((4.6115986, 4.6247558, 1.12395), (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)),
    22: ((4.6490636, 7.411491399999999, 0.889), (1.0, 0.0, 0.0, 0.0)),
    23: ((4.574159, 7.411491399999999, 0.889), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    24: ((4.2559986, 4.6247558, 1.12395), (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)),
    25: ((4.007866, 4.3769534, 1.12395), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    26: ((4.007866, 4.0213534, 1.12395), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    27: ((4.2559986, 3.417951, 1.12395), (-0.7071067811865475, -0.0, 0.0, 0.7071067811865476)),
    28: ((4.574159, 0.6312154, 0.889), (6.123233995736766e-17, 0.0, 0.0, 1.0)),
    29: ((0.0136906, 0.6507734, 0.55245), (1.0, 0.0, 0.0, 0.0)),
    30: ((0.0136906, 1.0825734, 0.55245), (1.0, 0.0, 0.0, 0.0)),
    31: ((0.0140462, 3.7301932, 0.55245), (1.0, 0.0, 0.0, 0.0)),
    32: ((0.0140462, 4.1619931999999995, 0.55245), (1.0, 0.0, 0.0, 0.0)),
}


@dataclass
class DetectedTagCorner:
    tag_id: int
    corner_index: int
    image_point: list[float]
    corrected_image_point: list[float]
    corrected_point: list[float]
    field_point: list[float]
    object_point: list[float]
    confidence: float


@dataclass(frozen=True)
class FieldCoordinateMapping:
    name: str
    x_sign: float
    y_sign: float


FIELD_COORDINATE_MAPPINGS: tuple[FieldCoordinateMapping, ...] = (
    FieldCoordinateMapping("x_flipped_y_flipped", -1.0, -1.0),
    FieldCoordinateMapping("x_native_y_flipped", 1.0, -1.0),
    FieldCoordinateMapping("x_flipped_y_native", -1.0, 1.0),
    FieldCoordinateMapping("x_native_y_native", 1.0, 1.0),
)
DEFAULT_FIELD_COORDINATE_MAPPING = FIELD_COORDINATE_MAPPINGS[2]
LEFT_VIEW_FIELD_COORDINATE_MAPPING = FIELD_COORDINATE_MAPPINGS[2]
RIGHT_VIEW_FIELD_COORDINATE_MAPPING = FIELD_COORDINATE_MAPPINGS[3]

CORNER_PERMUTATIONS: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
    ("identity", (0, 1, 2, 3)),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _quaternion_to_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def infer_camera_matrix(
    width: int,
    height: int,
    horizontal_fov_deg: float = DEFAULT_CAMERA_HFOV_DEG,
    focal_y_scale: float = 1.0,
    principal_x: Optional[float] = None,
    principal_y: Optional[float] = None,
) -> np.ndarray:
    half_angle = math.radians(horizontal_fov_deg / 2.0)
    focal_x = (width / 2.0) / max(math.tan(half_angle), 1e-6)
    focal_y = focal_x * max(float(focal_y_scale), 1e-6)
    return np.array(
        [
            [focal_x, 0.0, (width / 2.0) if principal_x is None else float(principal_x)],
            [0.0, focal_y, (height / 2.0) if principal_y is None else float(principal_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _solve_pose_with_fov_search(
    object_points: list[list[float]],
    image_points: list[list[float]],
    width: int,
    height: int,
    field_mapping: FieldCoordinateMapping,
    prior_homography: Optional[list[list[float]]] = None,
    prior_delta_weight: float = POSE_PRIOR_WEIGHT,
    prefer_prior_solution: bool = False,
) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
    def evaluate_candidate(
        method_name: str,
        camera_matrix: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> None:
        nonlocal best_result

        candidate_rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
        candidate_tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)
        if hasattr(cv2, "solvePnPRefineLM"):
            try:
                candidate_rvec, candidate_tvec = cv2.solvePnPRefineLM(
                    object_points_array,
                    image_points_array,
                    camera_matrix,
                    np.zeros((5, 1), dtype=np.float64),
                    candidate_rvec,
                    candidate_tvec,
                )
            except cv2.error:
                pass
        projected_points, _ = cv2.projectPoints(
            object_points_array,
            candidate_rvec,
            candidate_tvec,
            camera_matrix,
            np.zeros((5, 1), dtype=np.float64),
        )
        projected_points = projected_points.reshape(-1, 2)
        reprojection_error = float(
            np.mean(
                np.linalg.norm(projected_points - image_points_array, axis=1)
            )
        )
        candidate_homography = derive_ground_plane_homography(camera_matrix, candidate_rvec, candidate_tvec, field_mapping)
        prior_delta = (
            _homography_delta(candidate_homography, prior_homography, width, height)
            if prior_homography is not None
            else None
        )
        score = reprojection_error + (
            0.0
            if prior_delta is None or math.isinf(prior_delta)
            else (float(prior_delta_weight) * prior_delta)
        )
        result = {
            "success": True,
            "rvec": candidate_rvec,
            "tvec": candidate_tvec,
            "camera_matrix": camera_matrix,
            "candidate_homography": candidate_homography,
            "reprojection_error_px": reprojection_error,
            "prior_delta_in": None if prior_delta is None or math.isinf(prior_delta) else prior_delta,
            "score": score,
            "camera_hfov_deg": fov,
            "camera_focal_y_scale": focal_y_scale,
            "camera_principal_x_offset": principal_x_offset,
            "camera_principal_y_offset": principal_y_offset,
            "pnp_method": method_name,
            "prior_delta_weight": float(prior_delta_weight),
        }
        candidate_summaries.append(
            {
                "score": float(score),
                "reprojection_error_px": float(reprojection_error),
                "prior_delta_in": None if prior_delta is None or math.isinf(prior_delta) else float(prior_delta),
                "prior_delta_weight": float(prior_delta_weight),
                "camera_hfov_deg": float(fov),
                "camera_focal_y_scale": float(focal_y_scale),
                "camera_principal_x_offset": float(principal_x_offset),
                "camera_principal_y_offset": float(principal_y_offset),
                "pnp_method": method_name,
            }
        )
        if best_result is None or score < float(best_result["score"]):
            best_result = result

    object_points_array = np.array(object_points, dtype=np.float64)
    image_points_array = np.array(image_points, dtype=np.float64)
    centered_points = object_points_array - np.mean(object_points_array, axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered_points)
    planar_thickness = float(singular_values[-1]) if len(singular_values) else 0.0
    planar_span = float(singular_values[-2]) if len(singular_values) >= 2 else 0.0
    relative_thickness = 0.0 if planar_span <= 1e-9 else float(planar_thickness / planar_span)
    planar_layout = bool(relative_thickness <= PLANAR_LAYOUT_MAX_RELATIVE_THICKNESS)

    if prefer_prior_solution and prior_homography is not None:
        candidate_fovs = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
        candidate_focal_y_scales = [1.0]
        candidate_principal_x_offsets = [0.0]
        candidate_principal_y_offsets = [0.0]
        candidate_flags = [
            ("iterative", cv2.SOLVEPNP_ITERATIVE),
        ]
    else:
        candidate_fovs = [40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0]
        candidate_focal_y_scales = [0.9, 1.0, 1.1]
        candidate_principal_x_offsets = [-0.1, 0.0, 0.1]
        candidate_principal_y_offsets = [-0.1, 0.0, 0.1]
        candidate_flags = [
            ("iterative", cv2.SOLVEPNP_ITERATIVE),
            ("sqpnp", getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)),
            ("epnp", cv2.SOLVEPNP_EPNP),
        ]

    best_result: dict[str, Any] | None = None
    candidate_summaries: list[dict[str, Any]] = []

    for fov in candidate_fovs:
        for focal_y_scale in candidate_focal_y_scales:
            for principal_x_offset in candidate_principal_x_offsets:
                for principal_y_offset in candidate_principal_y_offsets:
                    camera_matrix = infer_camera_matrix(
                        width,
                        height,
                        horizontal_fov_deg=fov,
                        focal_y_scale=focal_y_scale,
                        principal_x=(width / 2.0) + (principal_x_offset * width),
                        principal_y=(height / 2.0) + (principal_y_offset * height),
                    )
                    prior_guess_rvec, prior_guess_tvec = _pose_guess_from_prior_homography(
                        prior_homography,
                        camera_matrix,
                        field_mapping,
                    )
                    if prior_guess_rvec is not None and prior_guess_tvec is not None:
                        evaluate_candidate("prior_refine", camera_matrix, prior_guess_rvec, prior_guess_tvec)
                    if prior_guess_rvec is not None and prior_guess_tvec is not None:
                        try:
                            success, rvec, tvec = cv2.solvePnP(
                                object_points_array,
                                image_points_array,
                                camera_matrix,
                                np.zeros((5, 1), dtype=np.float64),
                                rvec=prior_guess_rvec,
                                tvec=prior_guess_tvec,
                                useExtrinsicGuess=True,
                                flags=cv2.SOLVEPNP_ITERATIVE,
                            )
                        except cv2.error:
                            success = False
                        if success:
                            evaluate_candidate("iterative_prior", camera_matrix, rvec, tvec)
                    if prefer_prior_solution and prior_homography is not None and prior_guess_rvec is not None and prior_guess_tvec is not None:
                        continue

                    for method_name, flag in candidate_flags:
                        try:
                            success, rvec, tvec = cv2.solvePnP(
                                object_points_array,
                                image_points_array,
                                camera_matrix,
                                np.zeros((5, 1), dtype=np.float64),
                                flags=flag,
                            )
                        except cv2.error:
                            continue
                        if success:
                            evaluate_candidate(method_name, camera_matrix, rvec, tvec)

    if best_result is None:
        return False, None, None, {
            "reason": "No PnP pose candidate converged.",
            "camera_hfov_candidates": candidate_fovs,
            "camera_focal_y_scale_candidates": candidate_focal_y_scales,
            "camera_principal_x_offset_candidates": candidate_principal_x_offsets,
            "camera_principal_y_offset_candidates": candidate_principal_y_offsets,
            "field_coordinate_mapping": field_mapping.name,
            "prior_delta_weight": float(prior_delta_weight),
            "tag_layout_is_planar": planar_layout,
            "tag_layout_relative_thickness": relative_thickness,
            "prefer_prior_solution": bool(prefer_prior_solution),
        }

    best_result["tag_layout_is_planar"] = planar_layout
    best_result["tag_layout_relative_thickness"] = relative_thickness
    best_result["prefer_prior_solution"] = bool(prefer_prior_solution)
    best_result["candidate_summaries"] = sorted(candidate_summaries, key=lambda item: float(item["score"]))[:5]
    return True, best_result["rvec"], best_result["tvec"], best_result


def _repo_field_point_from_wpilib(
    point_xyz_m: list[float],
    mapping: FieldCoordinateMapping = DEFAULT_FIELD_COORDINATE_MAPPING,
) -> list[float]:
    x_in = point_xyz_m[0] * 39.3700787402
    y_in = point_xyz_m[1] * 39.3700787402
    return [
        float(mapping.x_sign * (x_in - (FIELD_WIDTH_IN / 2.0))),
        float(mapping.y_sign * (y_in - (FIELD_HEIGHT_IN / 2.0))),
    ]


def _candidate_object_point(tag_id: int, detected_corner_index: int, permutation: tuple[int, int, int, int]) -> list[float]:
    world_corners = get_tag_world_corners(tag_id)
    return world_corners[permutation[detected_corner_index]]


def _tag_center_points(tag_ids: list[int]) -> list[list[float]]:
    centers: list[list[float]] = []
    for tag_id in sorted(set(tag_ids)):
        if tag_id not in TAG_POSES_2026:
            continue
        translation, _ = TAG_POSES_2026[tag_id]
        centers.append(_repo_field_point_from_wpilib(list(translation)))
    return centers


def _describe_detected_tag_geometry(tag_ids: list[int]) -> dict[str, Any]:
    centers = _tag_center_points(tag_ids)
    world_points = np.array(
        [point for tag_id in sorted(set(tag_ids)) if tag_id in TAG_POSES_2026 for point in get_tag_world_corners(tag_id)],
        dtype=np.float64,
    )
    if world_points.size:
        centered = world_points - np.mean(world_points, axis=0)
        _, singular_values, _ = np.linalg.svd(centered)
        planar_thickness = float(singular_values[-1]) if len(singular_values) else 0.0
        planar_span = float(singular_values[-2]) if len(singular_values) >= 2 else 0.0
        relative_thickness = 0.0 if planar_span <= 1e-9 else float(planar_thickness / planar_span)
        is_planar = bool(relative_thickness <= PLANAR_LAYOUT_MAX_RELATIVE_THICKNESS)
    else:
        planar_thickness = 0.0
        planar_span = 0.0
        relative_thickness = 0.0
        is_planar = False
    if not centers:
        return {
            "tag_center_points": [],
            "tag_center_spread_in": 0.0,
            "tag_center_x_span_in": 0.0,
            "tag_center_y_span_in": 0.0,
            "clustered_tag_spread_in": POSE_PRIOR_CLUSTER_SPREAD_IN,
            "tag_layout_is_planar": is_planar,
            "tag_layout_planar_thickness_m": planar_thickness,
            "tag_layout_planar_span_m": planar_span,
            "tag_layout_relative_thickness": relative_thickness,
        }

    xs = [float(point[0]) for point in centers]
    ys = [float(point[1]) for point in centers]
    max_pairwise_distance = 0.0
    for index, point_a in enumerate(centers):
        for point_b in centers[index + 1:]:
            max_pairwise_distance = max(
                max_pairwise_distance,
                math.hypot(float(point_b[0]) - float(point_a[0]), float(point_b[1]) - float(point_a[1])),
            )

    return {
        "tag_center_points": centers,
        "tag_center_spread_in": float(max_pairwise_distance),
        "tag_center_x_span_in": float(max(xs) - min(xs)),
        "tag_center_y_span_in": float(max(ys) - min(ys)),
        "clustered_tag_spread_in": POSE_PRIOR_CLUSTER_SPREAD_IN,
        "tag_layout_is_planar": is_planar,
        "tag_layout_planar_thickness_m": planar_thickness,
        "tag_layout_planar_span_m": planar_span,
        "tag_layout_relative_thickness": relative_thickness,
    }


def _prior_weight_for_tag_geometry(tag_center_spread_in: float, prior_homography: Optional[list[list[float]]], tag_layout_is_planar: bool = False) -> float:
    if prior_homography is None:
        return POSE_PRIOR_WEIGHT
    if tag_layout_is_planar and tag_center_spread_in < POSE_PRIOR_CLUSTER_SPREAD_IN:
        return POSE_PRIOR_PLANAR_CLUSTER_WEIGHT
    spread = max(float(tag_center_spread_in), 15.0)
    if spread >= POSE_PRIOR_CLUSTER_SPREAD_IN:
        return POSE_PRIOR_WEIGHT
    scaled_weight = POSE_PRIOR_WEIGHT * (POSE_PRIOR_CLUSTER_SPREAD_IN / spread)
    return float(min(POSE_PRIOR_MAX_WEIGHT, max(POSE_PRIOR_WEIGHT, scaled_weight)))


def _max_prior_delta_for_tag_geometry(tag_center_spread_in: float, prior_homography: Optional[list[list[float]]]) -> float:
    return float(POSE_PRIOR_MAX_DELTA_IN)


def get_tag_world_corners(tag_id: int) -> list[list[float]]:
    translation, quaternion = TAG_POSES_2026[tag_id]
    rotation = _quaternion_to_matrix(quaternion)
    half_size = APRILTAG_SIZE_M / 2.0
    # WPILib/PhotonVision AprilTag frames point +X out of the visible tag face,
    # +Y to the tag's right, and +Z up when viewing the tag head-on.
    # OpenCV ArUco returns corners in image order: top-left, top-right, bottom-right, bottom-left.
    local_corners = np.array(
        [
            [0.0, -half_size, half_size],
            [0.0, half_size, half_size],
            [0.0, half_size, -half_size],
            [0.0, -half_size, -half_size],
        ],
        dtype=np.float64,
    )
    origin = np.array(translation, dtype=np.float64)
    world_corners = []
    for local_corner in local_corners:
        world = rotation @ local_corner + origin
        world_corners.append([float(world[0]), float(world[1]), float(world[2])])
    return world_corners


def decode_data_url_image(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("Frame image must be a data URL.")
    _, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the captured frame image.")
    return image


def _build_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 35
    parameters.adaptiveThreshWinSizeStep = 5
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def _summarize_attempts(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "No detection attempts were recorded."
    best = sorted(attempts, key=lambda attempt: (int(attempt.get("tag_count", 0)), float(attempt.get("stddev", 0.0))), reverse=True)[:3]
    return "; ".join(
        f"{attempt.get('variant')} @{attempt.get('width')}x{attempt.get('height')} -> {attempt.get('tag_count')} tags"
        for attempt in best
    )


def _detect_tag_corners(
    image: np.ndarray,
    roi: list[float],
    distortion_x: float,
    distortion_y: float,
) -> tuple[list[DetectedTagCorner], list[int], dict[str, Any]]:
    if len(roi) < 4:
        return [], [], {"accepted": False, "reason": "ROI must be defined before auto-detection."}

    x1, y1, x2, y2 = [int(round(value)) for value in roi[:4]]
    if x2 <= x1 or y2 <= y1:
        return [], [], {"accepted": False, "reason": "ROI must be a valid box before auto-detection."}

    crop = image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if crop.size == 0:
        return [], [], {"accepted": False, "reason": "ROI crop is empty."}

    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    corrected_origin = undistort_point([float(x1), float(y1)], roi, distortion_x, distortion_y)
    detector = _build_detector()
    variants: list[tuple[str, np.ndarray]] = [
        ("gray_1x", grayscale),
        ("equalized_1x", cv2.equalizeHist(grayscale)),
        ("adaptive_1x", cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)),
        ("gray_2x", cv2.resize(grayscale, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)),
        ("equalized_2x", cv2.resize(cv2.equalizeHist(grayscale), None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)),
    ]
    best_corners: list[np.ndarray] = []
    best_ids: Optional[np.ndarray] = None
    best_scale = 1.0
    attempt_debug: list[dict[str, Any]] = []
    for variant_name, variant_image in variants:
        corners, ids, _ = detector.detectMarkers(variant_image)
        scale = float(variant_image.shape[1]) / float(grayscale.shape[1]) if grayscale.shape[1] else 1.0
        attempt_info = {
            "variant": variant_name,
            "width": int(variant_image.shape[1]),
            "height": int(variant_image.shape[0]),
            "mean": float(np.mean(variant_image)),
            "stddev": float(np.std(variant_image)),
            "tag_count": int(0 if ids is None else len(ids)),
            "detected_tag_ids": [] if ids is None else [int(value[0]) for value in ids.tolist()],
        }
        attempt_debug.append(attempt_info)
        if ids is not None and (best_ids is None or len(ids) > len(best_ids)):
            best_corners = corners
            best_ids = ids
            best_scale = scale

    ids = best_ids
    corners = best_corners
    if ids is None or len(ids) == 0:
        debug = {
            "accepted": False,
            "reason": "No AprilTags detected in this ROI.",
            "roi": [x1, y1, x2, y2],
            "crop_width": int(crop.shape[1]),
            "crop_height": int(crop.shape[0]),
            "crop_mean": float(np.mean(grayscale)),
            "crop_stddev": float(np.std(grayscale)),
            "attempts": attempt_debug,
            "attempt_summary": _summarize_attempts(attempt_debug),
        }
        logger.info("AprilTag detection miss: %s", debug)
        return [], [], debug

    detected: list[DetectedTagCorner] = []
    detected_ids: list[int] = []
    for marker_corners, marker_id_array in zip(corners, ids):
        tag_id = int(marker_id_array[0])
        if tag_id not in TAG_POSES_2026:
            continue
        detected_ids.append(tag_id)
        world_corners = get_tag_world_corners(tag_id)
        for corner_index, point in enumerate(marker_corners.reshape(-1, 2).tolist()):
            raw_image_point = [float((point[0] / best_scale) + x1), float((point[1] / best_scale) + y1)]
            corrected = undistort_point(raw_image_point, roi, distortion_x, distortion_y)
            detected.append(
                DetectedTagCorner(
                    tag_id=tag_id,
                    corner_index=corner_index,
                    image_point=raw_image_point,
                    corrected_image_point=[float(corrected[0]), float(corrected[1])],
                    corrected_point=[
                        float(corrected[0] - corrected_origin[0]),
                        float(corrected[1] - corrected_origin[1]),
                    ],
                    field_point=_repo_field_point_from_wpilib(world_corners[corner_index]),
                    object_point=world_corners[corner_index],
                    confidence=1.0,
                )
            )

    detected_ids = sorted(set(detected_ids))
    debug = {
        "accepted": False,
        "tag_count": len(detected_ids),
        "corner_count": len(detected),
        "detected_tag_ids": detected_ids,
        "roi": [x1, y1, x2, y2],
        "crop_width": int(crop.shape[1]),
        "crop_height": int(crop.shape[0]),
        "crop_mean": float(np.mean(grayscale)),
        "crop_stddev": float(np.std(grayscale)),
        "attempts": attempt_debug,
        "attempt_summary": _summarize_attempts(attempt_debug),
    }
    logger.info("AprilTag detection hit: %s", {k: debug[k] for k in ("tag_count", "corner_count", "detected_tag_ids", "attempt_summary")})
    return detected, detected_ids, debug


def _homography_error(
    homography: list[list[float]],
    image_points: list[list[float]],
    field_points: list[list[float]],
) -> float:
    src = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.array(field_points, dtype=np.float32)
    projected = cv2.perspectiveTransform(src, np.array(homography, dtype=np.float32)).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - dst, axis=1)))


def _normalize_prior_homography(homography: list[list[float]], width: int, height: int) -> list[list[float]]:
    sample_points = np.array(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [0.0, float(height)],
            [float(width), float(height)],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(sample_points, np.array(homography, dtype=np.float32)).reshape(-1, 2)
    xs = projected[:, 0]
    ys = projected[:, 1]
    normalized = np.array(homography, dtype=np.float64)
    edge_tolerance = 8.0

    # Older side-view calibrations are centered in X but still top-left-origin in Y.
    y_looks_top_left = (
        float(np.min(ys)) >= -edge_tolerance and
        float(np.max(ys)) <= (FIELD_HEIGHT_IN + edge_tolerance) and
        (
            float(np.min(ys)) <= edge_tolerance or
            float(np.max(ys)) >= (FIELD_HEIGHT_IN - edge_tolerance)
        ) and
        (float(np.max(ys)) - float(np.min(ys))) >= (FIELD_HEIGHT_IN * 0.5)
    )
    if y_looks_top_left:
        normalized = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, -(FIELD_HEIGHT_IN / 2.0)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ) @ normalized

    x_looks_top_left = (
        float(np.min(xs)) >= -edge_tolerance and
        float(np.max(xs)) <= (FIELD_WIDTH_IN + edge_tolerance) and
        (
            float(np.min(xs)) <= edge_tolerance or
            float(np.max(xs)) >= (FIELD_WIDTH_IN - edge_tolerance)
        ) and
        (float(np.max(xs)) - float(np.min(xs))) >= (FIELD_WIDTH_IN * 0.5)
    )
    if x_looks_top_left:
        normalized = np.array(
            [
                [1.0, 0.0, -(FIELD_WIDTH_IN / 2.0)],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ) @ normalized

    return normalized.tolist()


def _meters_to_repo_matrix(field_mapping: FieldCoordinateMapping) -> np.ndarray:
    return np.array(
        [
            [field_mapping.x_sign * 39.3700787402, 0.0, -field_mapping.x_sign * (FIELD_WIDTH_IN / 2.0)],
            [0.0, field_mapping.y_sign * 39.3700787402, -field_mapping.y_sign * (FIELD_HEIGHT_IN / 2.0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _pose_guess_from_prior_homography(
    prior_homography: Optional[list[list[float]]],
    camera_matrix: np.ndarray,
    field_mapping: FieldCoordinateMapping,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if prior_homography is None:
        return None, None

    try:
        prior_h = np.array(prior_homography, dtype=np.float64)
        repo_to_image = np.linalg.inv(prior_h)
        world_to_image = repo_to_image @ _meters_to_repo_matrix(field_mapping)
        normalized = np.linalg.inv(camera_matrix) @ world_to_image
        col0 = normalized[:, 0]
        col1 = normalized[:, 1]
        scale = (np.linalg.norm(col0) + np.linalg.norm(col1)) / 2.0
        if scale <= 1e-9:
            return None, None
        r1 = col0 / scale
        r2 = col1 / scale
        t = normalized[:, 2] / scale
        r3 = np.cross(r1, r2)
        rotation_approx = np.column_stack((r1, r2, r3))
        u, _, vt = np.linalg.svd(rotation_approx)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            rotation *= -1.0
            t *= -1.0
        rvec, _ = cv2.Rodrigues(rotation)
        return rvec.reshape(3, 1), t.reshape(3, 1)
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return None, None


def _homography_delta(
    candidate_homography: list[list[float]],
    prior_homography: Optional[list[list[float]]],
    width: int,
    height: int,
) -> float:
    if prior_homography is None:
        return math.inf

    sample_points = np.array(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [0.0, float(height)],
            [float(width), float(height)],
            [float(width) / 2.0, float(height) / 2.0],
            [float(width) * 0.25, float(height) * 0.5],
            [float(width) * 0.75, float(height) * 0.5],
            [float(width) * 0.5, float(height) * 0.25],
            [float(width) * 0.5, float(height) * 0.75],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    candidate = cv2.perspectiveTransform(sample_points, np.array(candidate_homography, dtype=np.float32)).reshape(-1, 2)
    prior = cv2.perspectiveTransform(sample_points, np.array(prior_homography, dtype=np.float32)).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(candidate - prior, axis=1)))


def derive_ground_plane_homography(
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    field_mapping: FieldCoordinateMapping = DEFAULT_FIELD_COORDINATE_MAPPING,
) -> list[list[float]]:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    world_to_image = camera_matrix @ np.column_stack((rotation_matrix[:, 0], rotation_matrix[:, 1], tvec.reshape(3)))
    image_to_world = np.linalg.inv(world_to_image)
    meters_to_repo = _meters_to_repo_matrix(field_mapping)
    return (meters_to_repo @ image_to_world).tolist()


def solve_apriltag_view_calibration(
    view_name: str,
    frame_image: np.ndarray,
    roi: list[float],
    distortion_x: float = 0.0,
    distortion_y: float = 0.0,
    prior_homography: Optional[list[list[float]]] = None,
) -> tuple[Optional[ViewCalibration], dict[str, Any]]:
    detected, detected_tag_ids, debug = _detect_tag_corners(frame_image, roi, distortion_x, distortion_y)
    tag_geometry_debug = _describe_detected_tag_geometry(detected_tag_ids)
    debug.update(tag_geometry_debug)
    if len(detected_tag_ids) < MIN_DETECTED_TAGS or len(detected) < 4:
        debug["reason"] = "Need at least one visible AprilTag before solving a side-camera calibration."
        return None, debug

    image_points = [corner.corrected_point for corner in detected]
    prior_delta_weight = _prior_weight_for_tag_geometry(
        float(tag_geometry_debug["tag_center_spread_in"]),
        prior_homography,
        bool(tag_geometry_debug.get("tag_layout_is_planar")),
    )
    max_prior_delta_in = _max_prior_delta_for_tag_geometry(
        float(tag_geometry_debug["tag_center_spread_in"]),
        prior_homography,
    )
    tag_geometry_debug["prior_delta_weight"] = float(prior_delta_weight)
    tag_geometry_debug["max_prior_delta_in"] = float(max_prior_delta_in)
    debug.update(tag_geometry_debug)

    crop_width = int(round(max(float(roi[2]) - float(roi[0]), 1.0)))
    crop_height = int(round(max(float(roi[3]) - float(roi[1]), 1.0)))
    normalized_prior_homography = (
        _normalize_prior_homography(prior_homography, crop_width, crop_height)
        if prior_homography is not None
        else None
    )
    permutation_name, permutation = CORNER_PERMUTATIONS[0]
    field_mapping = (
        LEFT_VIEW_FIELD_COORDINATE_MAPPING
        if view_name == "left"
        else RIGHT_VIEW_FIELD_COORDINATE_MAPPING
    )
    prefer_prior_solution = (
        normalized_prior_homography is not None and
        bool(tag_geometry_debug.get("tag_layout_is_planar")) and
        float(tag_geometry_debug["tag_center_spread_in"]) < POSE_PRIOR_CLUSTER_SPREAD_IN
    )
    object_points = [
        _candidate_object_point(corner.tag_id, corner.corner_index, permutation)
        for corner in detected
    ]
    success, rvec, tvec, pose_search_debug = _solve_pose_with_fov_search(
        object_points,
        image_points,
        crop_width,
        crop_height,
        field_mapping,
        normalized_prior_homography,
        prior_delta_weight,
        prefer_prior_solution=prefer_prior_solution,
    )
    field_mapping_debug = {
        "selected_field_coordinate_mapping": field_mapping.name,
        "field_coordinate_mapping_candidates": [
            {
                "mapping": field_mapping.name,
                "success": bool(success),
                "score": None if not success else float(pose_search_debug.get("score", math.inf)),
                "prior_delta_in": None if not success else pose_search_debug.get("prior_delta_in"),
                "reprojection_error_px": None if not success else pose_search_debug.get("reprojection_error_px"),
                "pnp_method": None if not success else pose_search_debug.get("pnp_method"),
            }
        ],
        "field_coordinate_mapping_locked": True,
        "field_coordinate_mapping_reason": "Side-view AprilTag previews use a fixed per-view repo map orientation.",
    }
    if not success:
        debug.update(
            {
                "reason": "No AprilTag pose candidate converged for the fixed side-view field mapping.",
                "corner_permutation_candidates": [name for name, _ in CORNER_PERMUTATIONS],
                **field_mapping_debug,
            }
        )
        return None, _json_safe(debug)
    pose_search_debug = {
        **pose_search_debug,
        "corner_permutation": permutation_name,
    }
    camera_matrix = pose_search_debug.get("camera_matrix") if success else None
    serializable_pose_search_debug = _json_safe({
        key: value
        for key, value in pose_search_debug.items()
        if key not in {"camera_matrix", "candidate_homography"}
    })

    pose_homography_list: Optional[list[list[float]]] = None
    pose_error = math.inf
    pose_reprojection_error = math.inf
    pose_prior_delta = math.inf
    if success and camera_matrix is not None:
        pose_homography_list = pose_search_debug.get("candidate_homography") or derive_ground_plane_homography(camera_matrix, rvec, tvec, field_mapping)
        pose_reprojection_error = float(pose_search_debug["reprojection_error_px"])
        if pose_search_debug.get("prior_delta_in") is not None:
            pose_prior_delta = float(pose_search_debug["prior_delta_in"])

    accepted_source = "manual"
    accepted_homography: Optional[list[list[float]]] = None
    accepted_error = math.inf
    if (
        len(detected_tag_ids) >= MIN_ACCEPTED_TAGS and
        pose_homography_list is not None and
        pose_reprojection_error <= POSE_REPROJECTION_MAX_ERROR_PX and
        (
            (
                prefer_prior_solution and
                str(pose_search_debug.get("pnp_method") or "").startswith("prior")
            ) or
            math.isinf(pose_prior_delta) or
            pose_prior_delta <= max_prior_delta_in
        )
    ):
        accepted_source = "apriltag_pose_seeded"
        accepted_homography = pose_homography_list
        accepted_error = pose_reprojection_error

    if accepted_homography is None:
        debug.update(
            {
                "reason": (
                    "Need at least one visible AprilTag before auto-calibrating a side view."
                    if len(detected_tag_ids) < MIN_ACCEPTED_TAGS else
                    (
                        f"Detected tags are clustered on the field map, so this preview must stay within {max_prior_delta_in:.1f} in "
                        f"of the current side-view calibration. The candidate drifted {pose_prior_delta:.1f} in."
                        if float(tag_geometry_debug["tag_center_spread_in"]) < POSE_PRIOR_CLUSTER_SPREAD_IN
                        else "AprilTag pose solve was rejected because the pose fit drifted too far from the current side-view calibration."
                    )
                    if not math.isinf(pose_prior_delta) and pose_prior_delta > max_prior_delta_in
                    else "AprilTag pose solve was rejected because the pose fit was too unstable."
                ),
                "pose_error_in": None if math.isinf(pose_error) else pose_error,
                "pose_reprojection_error_px": None if math.isinf(pose_reprojection_error) else pose_reprojection_error,
                "pose_prior_delta_in": None if math.isinf(pose_prior_delta) else pose_prior_delta,
                "min_required_tags": MIN_ACCEPTED_TAGS,
                "max_reprojection_error_px": POSE_REPROJECTION_MAX_ERROR_PX,
                "max_prior_delta_in": max_prior_delta_in,
                "inferred_intrinsics": True,
                **tag_geometry_debug,
                **field_mapping_debug,
                **serializable_pose_search_debug,
            }
        )
        return None, _json_safe(debug)

    view = ViewCalibration(
        view=view_name,  # type: ignore[arg-type]
        roi=[float(value) for value in roi],
        homography=accepted_homography,
        landmarks=[],
        distortion_x=float(distortion_x),
        distortion_y=float(distortion_y),
        distortion_strength=float(distortion_y),
        reprojection_error=float(accepted_error),
        confidence=max(0.6, min(0.98, 1.0 - (accepted_error / 32.0))),
        calibration_source=accepted_source,  # type: ignore[arg-type]
        detected_tag_ids=detected_tag_ids,
        pose_debug={
            "accepted": True,
            "accepted_source": accepted_source,
            "preserve_homography": True,
            "inferred_intrinsics": True,
            "camera_hfov_deg": float(serializable_pose_search_debug.get("camera_hfov_deg", DEFAULT_CAMERA_HFOV_DEG)),
            "pnp_method": serializable_pose_search_debug.get("pnp_method"),
            "camera_focal_y_scale": float(serializable_pose_search_debug.get("camera_focal_y_scale", 1.0)),
            "camera_principal_x_offset": float(serializable_pose_search_debug.get("camera_principal_x_offset", 0.0)),
            "camera_principal_y_offset": float(serializable_pose_search_debug.get("camera_principal_y_offset", 0.0)),
            "field_coordinate_mapping": field_mapping.name,
            "tag_count": len(detected_tag_ids),
            "corner_count": len(detected),
            "pose_error_in": None if math.isinf(pose_error) else float(pose_error),
            "pose_reprojection_error_px": None if math.isinf(pose_reprojection_error) else float(pose_reprojection_error),
            "pose_prior_delta_in": None if math.isinf(pose_prior_delta) else float(pose_prior_delta),
            "max_reprojection_error_px": POSE_REPROJECTION_MAX_ERROR_PX,
            "max_prior_delta_in": float(max_prior_delta_in),
            "field_layout": "2026-rebuilt-andymark",
            **tag_geometry_debug,
            **field_mapping_debug,
        },
        fallback_reason=f"AprilTag preview accepted for {view_name} using {accepted_source.replace('_', ' ')}.",
    )
    return view, _json_safe({
        **debug,
        "accepted": True,
        "accepted_source": accepted_source,
        "pose_error_in": None if math.isinf(pose_error) else float(pose_error),
        "pose_reprojection_error_px": None if math.isinf(pose_reprojection_error) else float(pose_reprojection_error),
        "inferred_intrinsics": True,
        "camera_hfov_deg": float(serializable_pose_search_debug.get("camera_hfov_deg", DEFAULT_CAMERA_HFOV_DEG)),
        "pnp_method": serializable_pose_search_debug.get("pnp_method"),
        "field_layout": "2026-rebuilt-andymark",
        **tag_geometry_debug,
        **field_mapping_debug,
    })
