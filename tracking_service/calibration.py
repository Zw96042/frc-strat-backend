from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .config import DEFAULT_CALIBRATION_FILE, FIELD_HEIGHT_IN, FIELD_WIDTH_IN
from .schemas import CalibrationEnvelope, FieldLandmark, MatchRecord, ViewCalibration


LEGACY_TARGET_PRESETS: dict[str, list[list[float]]] = {
    "main": [
        [-121.3, 108.5],
        [121.3, 108.5],
        [-121.3, -108.5],
        [121.3, -108.5],
    ],
    "left": [
        [-166.4564, 23.4334],
        [-166.4564, -23.3678],
        [-281.5264, 8.3038],
        [-281.5264, -30.0962],
    ],
    "right": [
        [166.4564, 23.4334],
        [166.4564, -23.3678],
        [281.5264, 30.0962],
        [281.5264, -8.3038],
    ],
}

MIXED_AXIS_TARGET_PRESETS: dict[str, list[list[float]]] = {
    "main": LEGACY_TARGET_PRESETS["main"],
    "left": [
        [-149.8488, 281.5264],
        [-119.5896, 166.4564],
        [-213.1920, 166.4564],
        [-226.6488, 281.5264],
    ],
    "right": [
        [226.6488, 281.5264],
        [213.3232, 166.4564],
        [119.7208, 166.4564],
        [149.8488, 281.5264],
    ],
}


def _center_field_y(points: list[list[float]]) -> list[list[float]]:
    return [[float(x), float(y - (FIELD_HEIGHT_IN / 2.0))] for x, y in points]


TARGET_PRESETS: dict[str, list[list[float]]] = {
    "main": LEGACY_TARGET_PRESETS["main"],
    "left": _center_field_y([
        MIXED_AXIS_TARGET_PRESETS["left"][3],
        MIXED_AXIS_TARGET_PRESETS["left"][0],
        MIXED_AXIS_TARGET_PRESETS["left"][2],
        MIXED_AXIS_TARGET_PRESETS["left"][1],
    ]),
    "right": _center_field_y([
        MIXED_AXIS_TARGET_PRESETS["right"][3],
        MIXED_AXIS_TARGET_PRESETS["right"][0],
        MIXED_AXIS_TARGET_PRESETS["right"][2],
        MIXED_AXIS_TARGET_PRESETS["right"][1],
    ]),
}


def _roi_to_landmarks(view: str, roi: list[float]) -> list[FieldLandmark]:
    x1, y1, x2, y2 = roi
    image_points = [
        [x1, y1],
        [x2, y1],
        [x1, y2],
        [x2, y2],
    ]
    return [
        FieldLandmark(
            name=f"{view}_{index + 1}",
            image_point=image_point,
            field_point=TARGET_PRESETS[view][index],
            confidence=0.42,
        )
        for index, image_point in enumerate(image_points)
    ]


def get_solve_field_points(view_name: str, landmarks: list[FieldLandmark]) -> list[list[float]]:
    return [list(landmark.field_point) for landmark in landmarks[:4]]


def distort_point(point: list[float], roi: list[float], x_strength: float = 0.0, y_strength: float = 0.0) -> list[float]:
    if len(roi) < 4 or (abs(x_strength) < 1e-9 and abs(y_strength) < 1e-9):
        return [float(point[0]), float(point[1])]

    x1, y1, x2, y2 = roi
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    half_width = max((x2 - x1) / 2.0, 1.0)
    half_height = max((y2 - y1) / 2.0, 1.0)
    delta_x = float(point[0]) - center_x
    delta_y = float(point[1]) - center_y
    normalized_x = (float(point[0]) - center_x) / half_width
    normalized_y = (float(point[1]) - center_y) / half_height
    curved_offset_x = x_strength * (normalized_y ** 2) * (delta_x / half_width) * half_width
    curved_offset_y = y_strength * (normalized_x ** 2) * (delta_y / half_height) * half_height
    return [
        float(point[0]) + curved_offset_x,
        float(point[1]) + curved_offset_y,
    ]


def undistort_point(
    point: list[float],
    roi: list[float],
    x_strength: float = 0.0,
    y_strength: float = 0.0,
    iterations: int = 8,
) -> list[float]:
    if len(roi) < 4 or (abs(x_strength) < 1e-9 and abs(y_strength) < 1e-9):
        return [float(point[0]), float(point[1])]

    estimate = [float(point[0]), float(point[1])]
    target_x = float(point[0])
    target_y = float(point[1])
    for _ in range(iterations):
        distorted_x, distorted_y = distort_point(estimate, roi, x_strength, y_strength)
        estimate[0] += target_x - distorted_x
        estimate[1] += target_y - distorted_y
    return estimate


def _points_match(a: list[list[float]], b: list[list[float]], tolerance: float = 1e-3) -> bool:
    if len(a) != len(b):
        return False
    return all(
        len(point_a) == len(point_b) == 2
        and abs(point_a[0] - point_b[0]) <= tolerance
        and abs(point_a[1] - point_b[1]) <= tolerance
        for point_a, point_b in zip(a, b)
    )


def _triangle_area(a: list[float], b: list[float], c: list[float]) -> float:
    return abs(
        a[0] * (b[1] - c[1]) +
        b[0] * (c[1] - a[1]) +
        c[0] * (a[1] - b[1])
    ) / 2.0


def _point_inside_triangle(point: list[float], triangle: list[list[float]], tolerance: float = 1e-3) -> bool:
    whole = _triangle_area(triangle[0], triangle[1], triangle[2])
    parts = (
        _triangle_area(point, triangle[0], triangle[1]) +
        _triangle_area(point, triangle[1], triangle[2]) +
        _triangle_area(point, triangle[2], triangle[0])
    )
    return abs(parts - whole) <= tolerance


def points_form_valid_quad(points: list[list[float]], min_area_floor: float = 100.0, min_area_ratio: float = 0.05) -> bool:
    if len(points) < 4:
        return False

    first_four = [list(point) for point in points[:4]]
    unique_points = {
        (round(point[0], 3), round(point[1], 3))
        for point in first_four
    }
    if len(unique_points) < 4:
        return False

    for index, point in enumerate(first_four):
        triangle = [first_four[other] for other in range(4) if other != index]
        if _point_inside_triangle(point, triangle):
            return False

    triangle_areas = [
        _triangle_area(first_four[0], first_four[1], first_four[2]),
        _triangle_area(first_four[0], first_four[1], first_four[3]),
        _triangle_area(first_four[0], first_four[2], first_four[3]),
        _triangle_area(first_four[1], first_four[2], first_four[3]),
    ]
    max_triangle_area = max(triangle_areas)
    min_triangle_area = min(triangle_areas)
    if max_triangle_area <= 1e-3:
        return False
    if min_triangle_area < max(min_area_floor, max_triangle_area * min_area_ratio):
        return False

    return True


def field_points_form_valid_quad(points: list[list[float]]) -> bool:
    return points_form_valid_quad(points)


def image_landmarks_form_valid_quad(roi: list[float], landmarks: list[FieldLandmark]) -> bool:
    if len(roi) < 4 or len(landmarks) < 4:
        return False

    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return False

    relative_points: list[list[float]] = []
    for landmark in landmarks[:4]:
        point_x = float(landmark.image_point[0])
        point_y = float(landmark.image_point[1])
        if not (x1 <= point_x <= x2 and y1 <= point_y <= y2):
            return False
        relative_points.append([point_x - x1, point_y - y1])

    return points_form_valid_quad(relative_points)


def upgrade_legacy_landmarks(view_name: str, landmarks: list[FieldLandmark]) -> list[FieldLandmark]:
    if view_name not in {"left", "right"} or len(landmarks) < 4:
        return landmarks

    current_points = [list(landmark.field_point) for landmark in landmarks[:4]]
    legacy_points = LEGACY_TARGET_PRESETS[view_name]
    mixed_axis_points = MIXED_AXIS_TARGET_PRESETS[view_name]
    should_upgrade = (
        _points_match(current_points, legacy_points) or
        _points_match(current_points, mixed_axis_points) or
        not field_points_form_valid_quad(current_points)
    )
    if not should_upgrade:
        return landmarks

    upgraded = []
    for index, landmark in enumerate(landmarks):
        if index < 4:
            upgraded.append(
                landmark.copy(
                    update={
                        "field_point": TARGET_PRESETS[view_name][index],
                        "confidence": max(float(landmark.confidence), 0.5),
                    }
                )
            )
        else:
            upgraded.append(landmark)
    return upgraded


def solve_view_homography(
    view_name: str,
    roi: list[float],
    landmarks: list[FieldLandmark],
    distortion_x: float = 0.0,
    distortion_y: float = 0.0,
) -> tuple[list[list[float]], float]:
    if len(landmarks) < 4:
        raise ValueError("At least four landmarks are required to solve a homography.")
    x1, y1, _, _ = roi
    corrected_roi_origin = undistort_point([x1, y1], roi, distortion_x, distortion_y)
    relative_image_points = [
        [
            undistort_point(landmark.image_point, roi, distortion_x, distortion_y)[0] - corrected_roi_origin[0],
            undistort_point(landmark.image_point, roi, distortion_x, distortion_y)[1] - corrected_roi_origin[1],
        ]
        for landmark in landmarks[:4]
    ]
    field_points = get_solve_field_points(view_name, landmarks)
    return solve_homography(relative_image_points, field_points)


def load_default_calibration(file_path: Optional[Union[Path, str]] = None) -> CalibrationEnvelope:
    path = Path(file_path or DEFAULT_CALIBRATION_FILE)
    raw = json.loads(path.read_text())
    views: list[ViewCalibration] = []

    for view_name in ("left", "main", "right"):
        roi = raw[f"{view_name}_bbox"]
        landmarks = _roi_to_landmarks(view_name, roi)
        homography, error = solve_view_homography(view_name, roi, landmarks, 0.0)
        views.append(
            ViewCalibration(
                view=view_name,  # type: ignore[arg-type]
                roi=roi,
                homography=homography,
                landmarks=landmarks,
                reprojection_error=error,
                confidence=0.42,
                fallback_reason="Seeded from manual calibration file until the 2026 landmark model is trained.",
            )
        )

    return CalibrationEnvelope(
        mode="manual_fallback",
        created_at=time.time(),
        updated_at=time.time(),
        quality_checks={
            "reprojection_error_ok": True,
            "field_bounds_ok": True,
            "overlap_consistency_ok": True,
            "notes": [
                "Using baseline manual calibration as the automatic fallback.",
                "The backend contract is ready for future landmark-model outputs.",
            ],
        },
        views=views,
    )


def calibration_for_match(match: MatchRecord) -> dict[str, ViewCalibration]:
    calibration: dict[str, ViewCalibration] = {}
    for view in match.calibration.views:
        upgraded_landmarks = upgrade_legacy_landmarks(view.view, view.landmarks)
        if upgraded_landmarks is view.landmarks:
            calibration[view.view] = view
            continue

        homography, error = solve_view_homography(
            view.view,
            view.roi,
            upgraded_landmarks,
            float(getattr(view, "distortion_x", 0.0)),
            float(getattr(view, "distortion_y", getattr(view, "distortion_strength", 0.0))),
        )
        calibration[view.view] = view.copy(
            update={
                "landmarks": upgraded_landmarks,
                "homography": homography,
                "reprojection_error": error,
                "fallback_reason": "Side-view field targets were normalized into centered field coordinates.",
            }
        )
    return calibration


def project_point(point: np.ndarray, homography: np.ndarray) -> np.ndarray:
    p = np.array([point[0], point[1], 1.0], dtype=np.float32)
    output = homography @ p
    if abs(float(output[2])) < 1e-6:
        return np.array([np.nan, np.nan], dtype=np.float32)
    output /= output[2]
    return output[:2]


def inside_roi(point: tuple[float, float], roi: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2


def normalize_field_point(projected: np.ndarray, view: Optional[ViewCalibration] = None) -> np.ndarray:
    if np.isnan(projected).any():
        return projected

    x = float(projected[0])
    y = float(projected[1])

    if view is not None and len(view.landmarks) >= 4:
        landmark_points = [landmark.field_point for landmark in view.landmarks[:4]]
        xs = [float(point[0]) for point in landmark_points]
        ys = [float(point[1]) for point in landmark_points]
        in_top_left_bounds = all(
            0.0 <= float(point[0]) <= FIELD_WIDTH_IN and 0.0 <= float(point[1]) <= FIELD_HEIGHT_IN
            for point in landmark_points
        )
        edge_tolerance = 6.0
        touches_field_edges = (
            min(xs) <= edge_tolerance or
            max(xs) >= FIELD_WIDTH_IN - edge_tolerance or
            min(ys) <= edge_tolerance or
            max(ys) >= FIELD_HEIGHT_IN - edge_tolerance
        )
        spans_large_fraction_of_field = (
            (max(xs) - min(xs)) >= (FIELD_WIDTH_IN * 0.5) or
            (max(ys) - min(ys)) >= (FIELD_HEIGHT_IN * 0.5)
        )
        uses_top_left_origin = in_top_left_bounds and touches_field_edges and spans_large_fraction_of_field
    else:
        uses_top_left_origin = False

    # Support older calibration files that project entirely in top-left field coordinates.
    if uses_top_left_origin:
        x = x - (FIELD_WIDTH_IN / 2.0)
        y = (FIELD_HEIGHT_IN / 2.0) - y

    return np.array([x, y], dtype=np.float32)


def project_detection_in_view(point: Tuple[float, float], view: ViewCalibration) -> np.ndarray:
    if not inside_roi(point, view.roi):
        return np.array([np.nan, np.nan], dtype=np.float32)

    x1, y1, _, _ = view.roi
    distortion_x = float(getattr(view, "distortion_x", 0.0))
    distortion_y = float(getattr(view, "distortion_y", getattr(view, "distortion_strength", 0.0)))
    corrected_point = undistort_point([float(point[0]), float(point[1])], view.roi, distortion_x, distortion_y)
    corrected_origin = undistort_point([x1, y1], view.roi, distortion_x, distortion_y)
    homography = np.array(view.homography, dtype=np.float32)
    projected = project_point(
        np.array([corrected_point[0] - corrected_origin[0], corrected_point[1] - corrected_origin[1]], dtype=np.float32),
        homography,
    )
    return normalize_field_point(projected, view)


def project_detection(point: Tuple[float, float], calibration: dict[str, ViewCalibration]) -> tuple[Optional[str], np.ndarray]:
    for view_name in ("left", "main", "right"):
        view = calibration[view_name]
        if inside_roi(point, view.roi):
            return view_name, project_detection_in_view(point, view)
    return None, np.array([np.nan, np.nan], dtype=np.float32)


def solve_homography(image_points: list[list[float]], field_points: list[list[float]]) -> tuple[list[list[float]], float]:
    if len(image_points) < 4 or len(field_points) < 4:
        raise ValueError("Homography solve requires at least four image points and four field points.")

    src = np.array(image_points, dtype=np.float64)
    dst = np.array(field_points, dtype=np.float64)
    homography, _ = cv2.findHomography(src, dst)
    if homography is None:
        raise ValueError("Could not compute homography from provided points.")

    projected = cv2.perspectiveTransform(src.astype(np.float32).reshape(-1, 1, 2), homography.astype(np.float32)).reshape(-1, 2)
    error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))
    return homography.tolist(), error
