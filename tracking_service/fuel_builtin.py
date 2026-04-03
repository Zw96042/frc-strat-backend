from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np


FIELD_IMAGE_NORM_BOUNDS = {
    "minX": 0.133,
    "maxX": 0.866,
    "minY": 0.053,
    "maxY": 0.946,
}
DEFAULT_FIELD_IMAGE_WIDTH = 3901
DEFAULT_FIELD_IMAGE_HEIGHT = 1583
MAX_OVERLAY_WIDTH = 1280


@dataclass(frozen=True)
class FuelColorModel:
    base_lab: np.ndarray
    base_hsv: np.ndarray
    hue_tolerance: float
    saturation_min: float
    value_min: float
    lab_tolerance: float


@dataclass(frozen=True)
class WallProjector:
    name: str
    mask: np.ndarray
    homography_to_local: np.ndarray
    ground_edge_start: np.ndarray
    ground_edge_end: np.ndarray


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _prepare_field_image(field_image_path: Path) -> np.ndarray:
    field_image = cv2.imread(str(field_image_path), cv2.IMREAD_COLOR)
    if field_image is not None and field_image.size:
        return field_image
    return np.zeros((DEFAULT_FIELD_IMAGE_HEIGHT, DEFAULT_FIELD_IMAGE_WIDTH, 3), dtype=np.uint8)


def _build_color_model(fuel_base_color: list[int]) -> FuelColorModel:
    rgb = np.array([[fuel_base_color]], dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    saturation = float(hsv[1])
    value = float(hsv[2])
    hue_tolerance = 16.0 if saturation >= 150 else 22.0 if saturation >= 90 else 30.0
    lab_tolerance = 48.0 if value >= 180 else 58.0 if saturation >= 90 else 68.0
    return FuelColorModel(
        base_lab=lab,
        base_hsv=hsv,
        hue_tolerance=hue_tolerance,
        saturation_min=max(26.0, saturation * 0.22),
        value_min=max(32.0, value * 0.22),
        lab_tolerance=lab_tolerance,
    )


def _build_polygon_mask(height: int, width: int, quad: Optional[list[list[float]]], offset_x: int, offset_y: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not quad:
        return mask
    polygon = np.array(
        [[int(round(point[0] - offset_x)), int(round(point[1] - offset_y))] for point in quad],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def _union_bbox(
    width: int,
    height: int,
    quads: list[Optional[list[list[float]]]],
    padding: int = 12,
) -> tuple[int, int, int, int]:
    points: list[list[float]] = []
    for quad in quads:
        if quad:
            points.extend(quad)
    if not points:
        return 0, 0, width, height
    xs = [int(round(point[0])) for point in points]
    ys = [int(round(point[1])) for point in points]
    left = max(0, min(xs) - padding)
    top = max(0, min(ys) - padding)
    right = min(width, max(xs) + padding + 1)
    bottom = min(height, max(ys) + padding + 1)
    return left, top, max(1, right - left), max(1, bottom - top)


def _build_ground_homography(
    ground_quad_pixels: list[list[float]],
    field_width: int,
    field_height: int,
) -> np.ndarray:
    destination = np.array(
        [
            [FIELD_IMAGE_NORM_BOUNDS["minX"] * field_width, FIELD_IMAGE_NORM_BOUNDS["minY"] * field_height],
            [FIELD_IMAGE_NORM_BOUNDS["maxX"] * field_width, FIELD_IMAGE_NORM_BOUNDS["minY"] * field_height],
            [FIELD_IMAGE_NORM_BOUNDS["maxX"] * field_width, FIELD_IMAGE_NORM_BOUNDS["maxY"] * field_height],
            [FIELD_IMAGE_NORM_BOUNDS["minX"] * field_width, FIELD_IMAGE_NORM_BOUNDS["maxY"] * field_height],
        ],
        dtype=np.float32,
    )
    source = np.array(ground_quad_pixels, dtype=np.float32)
    return cv2.getPerspectiveTransform(source, destination)


def _build_wall_projector(
    name: str,
    quad_pixels: Optional[list[list[float]]],
    mask: np.ndarray,
    ground_quad_pixels: list[list[float]],
) -> Optional[WallProjector]:
    if not quad_pixels:
        return None

    if name == "left":
        destination = np.array([[1, 1], [0, 1], [0, 0], [1, 0]], dtype=np.float32)
        ground_edge_start = np.array(ground_quad_pixels[0], dtype=np.float32)
        ground_edge_end = np.array(ground_quad_pixels[3], dtype=np.float32)
    else:
        destination = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        ground_edge_start = np.array(ground_quad_pixels[1], dtype=np.float32)
        ground_edge_end = np.array(ground_quad_pixels[2], dtype=np.float32)

    return WallProjector(
        name=name,
        mask=mask,
        homography_to_local=cv2.getPerspectiveTransform(np.array(quad_pixels, dtype=np.float32), destination),
        ground_edge_start=ground_edge_start,
        ground_edge_end=ground_edge_end,
    )


def _build_color_mask(frame_bgr: np.ndarray, color_model: FuelColorModel, roi_mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    hue_delta = np.abs(hsv[:, :, 0] - color_model.base_hsv[0])
    hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    lab_distance = np.linalg.norm(lab - color_model.base_lab, axis=2)
    primary = (
        (hue_delta <= color_model.hue_tolerance)
        & (saturation >= color_model.saturation_min)
        & (value >= color_model.value_min)
        & (lab_distance <= color_model.lab_tolerance)
    )
    highlight = (lab_distance <= color_model.lab_tolerance * 0.85) & (value >= max(90.0, color_model.value_min * 1.6))
    combined = np.where(primary | highlight, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(combined, roi_mask)


def _project_ground_point(
    x: float,
    y: float,
    homography_to_field: np.ndarray,
    field_width: int,
    field_height: int,
) -> list[int]:
    point = np.array([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, homography_to_field)[0, 0]
    fx = _clamp(projected[0] / max(field_width, 1), 0.0, 1.0)
    fy = _clamp(projected[1] / max(field_height, 1), 0.0, 1.0)
    return [int(round(fx * 10000.0)), int(round(fy * 10000.0)), 0]


def _project_wall_point(
    x: float,
    y: float,
    wall_projector: WallProjector,
    ground_homography: np.ndarray,
    field_width: int,
    field_height: int,
) -> list[int]:
    point = np.array([[[x, y]]], dtype=np.float32)
    local = cv2.perspectiveTransform(point, wall_projector.homography_to_local)[0, 0]
    along_edge = _clamp(local[0], 0.0, 1.0)
    height_norm = _clamp(local[1], 0.0, 1.0)
    ground_anchor = wall_projector.ground_edge_start + (wall_projector.ground_edge_end - wall_projector.ground_edge_start) * along_edge
    projected = cv2.perspectiveTransform(np.array([[ground_anchor]], dtype=np.float32), ground_homography)[0, 0]
    fx = _clamp(projected[0] / max(field_width, 1), 0.0, 1.0)
    fy = _clamp(projected[1] / max(field_height, 1), 0.0, 1.0)
    return [int(round(fx * 10000.0)), int(round(fy * 10000.0)), int(round(height_norm * 10000.0))]


def _make_overlay_writer(path: Path, width: int, height: int, fps: float) -> tuple[Optional[cv2.VideoWriter], tuple[int, int]]:
    scale = 1.0
    if width > MAX_OVERLAY_WIDTH:
        scale = MAX_OVERLAY_WIDTH / float(width)
    output_size = (max(2, int(round(width * scale))), max(2, int(round(height * scale))))
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(fps, 1.0)),
        output_size,
    )
    if not writer.isOpened():
        writer.release()
        return None, output_size
    return writer, output_size


def _draw_overlay_frame(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    output_writer: Optional[cv2.VideoWriter],
    output_size: tuple[int, int],
    frame_number: int,
    time_sec: float,
    ground_quad: list[list[float]],
    left_wall_quad: Optional[list[list[float]]],
    right_wall_quad: Optional[list[list[float]]],
) -> None:
    if output_writer is None:
        return

    preview = frame_bgr.copy()
    for quad, color in (
        (ground_quad, (52, 186, 255)),
        (left_wall_quad, (201, 133, 255)),
        (right_wall_quad, (201, 133, 255)),
    ):
        if not quad:
            continue
        polygon = np.array(quad, dtype=np.int32)
        cv2.polylines(preview, [polygon], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    for detection in detections:
        center = (int(round(detection["image_x"])), int(round(detection["image_y"])))
        radius = max(4, int(round(detection["radius"])))
        z_norm = float(detection["point"][2]) / 10000.0
        color = (214, 116, 255) if z_norm >= 0.02 else (70, 215, 255)
        cv2.circle(preview, center, radius, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(preview, center, 2, color, -1, lineType=cv2.LINE_AA)

    cv2.putText(
        preview,
        f"frame {frame_number}  t={time_sec:0.2f}s  detections={len(detections)}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (236, 240, 244),
        2,
        cv2.LINE_AA,
    )

    if preview.shape[1] != output_size[0] or preview.shape[0] != output_size[1]:
        preview = cv2.resize(preview, output_size, interpolation=cv2.INTER_AREA)
    output_writer.write(preview)


def _create_overlay_images(field_image: np.ndarray, frames: list[list[list[int]]], artifact_dir: Path) -> None:
    field_height, field_width = field_image.shape[:2]
    ground_heat = np.zeros((field_height, field_width), dtype=np.float32)
    air_heat = np.zeros((field_height, field_width), dtype=np.float32)
    radius = max(8, int(round(field_width / 220.0)))

    for frame in frames:
        for x_norm, y_norm, z_norm in frame:
            x = int(round(_clamp(x_norm / 10000.0, 0.0, 1.0) * max(field_width - 1, 1)))
            y = int(round(_clamp(y_norm / 10000.0, 0.0, 1.0) * max(field_height - 1, 1)))
            intensity = 1.0 + (z_norm / 10000.0) * 0.8
            target = air_heat if z_norm >= 200 else ground_heat
            cv2.circle(target, (x, y), radius, intensity, -1, lineType=cv2.LINE_AA)

    ground_blur = cv2.GaussianBlur(ground_heat, (0, 0), sigmaX=max(4.0, radius * 0.75), sigmaY=max(4.0, radius * 0.75))
    air_blur = cv2.GaussianBlur(air_heat, (0, 0), sigmaX=max(4.0, radius * 0.75), sigmaY=max(4.0, radius * 0.75))
    ground_norm = ground_blur / max(float(ground_blur.max()), 1.0)
    air_norm = air_blur / max(float(air_blur.max()), 1.0)

    transparent = np.zeros((field_height, field_width, 4), dtype=np.uint8)
    color = (
        ground_norm[:, :, None] * np.array([58.0, 214.0, 246.0], dtype=np.float32)
        + air_norm[:, :, None] * np.array([255.0, 130.0, 193.0], dtype=np.float32)
    )
    alpha = np.clip((ground_norm * 175.0) + (air_norm * 210.0), 0.0, 255.0)
    transparent[:, :, :3] = np.clip(color, 0.0, 255.0).astype(np.uint8)
    transparent[:, :, 3] = alpha.astype(np.uint8)

    composite = field_image.astype(np.float32)
    alpha_float = transparent[:, :, 3:4].astype(np.float32) / 255.0
    composite = composite * (1.0 - alpha_float) + transparent[:, :, :3].astype(np.float32) * alpha_float

    cv2.imwrite(str(artifact_dir / "overlay-transparent.png"), transparent)
    cv2.imwrite(str(artifact_dir / "overlay.png"), composite.astype(np.uint8))


def run_builtin_fuel_processor(
    *,
    video_path: str,
    artifact_dir: Path,
    field_image_path: Path,
    ground_quad_pixels: list[list[float]],
    left_wall_quad_pixels: Optional[list[list[float]]],
    right_wall_quad_pixels: Optional[list[list[float]]],
    fuel_base_color: list[int],
    analysis_fps: float,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the match video for built-in fuel processing.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError("Match video metadata is invalid for built-in fuel processing.")

    field_image = _prepare_field_image(field_image_path)
    field_height, field_width = field_image.shape[:2]
    ground_homography = _build_ground_homography(ground_quad_pixels, field_width, field_height)
    crop_left, crop_top, crop_width, crop_height = _union_bbox(
        frame_width,
        frame_height,
        [ground_quad_pixels, left_wall_quad_pixels, right_wall_quad_pixels],
    )

    ground_mask = _build_polygon_mask(crop_height, crop_width, ground_quad_pixels, crop_left, crop_top)
    left_mask = _build_polygon_mask(crop_height, crop_width, left_wall_quad_pixels, crop_left, crop_top)
    right_mask = _build_polygon_mask(crop_height, crop_width, right_wall_quad_pixels, crop_left, crop_top)
    roi_mask = cv2.bitwise_or(ground_mask, left_mask)
    roi_mask = cv2.bitwise_or(roi_mask, right_mask)
    if not np.any(roi_mask):
        cap.release()
        raise RuntimeError("Fuel calibration did not produce a usable search region.")

    left_projector = _build_wall_projector("left", left_wall_quad_pixels, left_mask, ground_quad_pixels)
    right_projector = _build_wall_projector("right", right_wall_quad_pixels, right_mask, ground_quad_pixels)
    color_model = _build_color_model(fuel_base_color)
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=240, varThreshold=18, detectShadows=False)

    effective_source_fps = source_fps if source_fps > 0.0 else float(max(analysis_fps, 1.0))
    target_fps = float(max(1.0, min(analysis_fps, effective_source_fps)))
    frame_stride = max(1, int(round(effective_source_fps / target_fps)))
    total_samples = max(1, int(math.ceil(frame_count / frame_stride))) if frame_count > 0 else 1
    target_fps = effective_source_fps / frame_stride

    overlay_writer, overlay_size = _make_overlay_writer(artifact_dir / "overlay-video.mp4", crop_width, crop_height, target_fps)

    frames: list[list[list[int]]] = []
    raw_lines = [
        "# Built-in fuel processor output",
        f"# video={video_path}",
        f"# analysis_fps={target_fps:.3f}",
        f"# stride={frame_stride}",
        f"# fuel_base_color={fuel_base_color}",
    ]
    log_lines = [
        "Built-in fuel processor enabled because the external fuel-density-map processor is unavailable.",
        f"Video path: {video_path}",
        f"Field image: {field_image_path}",
        f"Analysis fps: {target_fps:.3f}",
        f"Frame stride: {frame_stride}",
    ]

    processed_frames = 0
    total_detections = 0
    frames_with_detections = 0
    ground_detections = 0
    left_wall_detections = 0
    right_wall_detections = 0
    previous_gray: Optional[np.ndarray] = None
    component_max_area = max(240, int(crop_width * crop_height * 0.0035))
    component_min_area = max(6, int(crop_width * crop_height * 0.000015))

    if progress_callback is not None:
        progress_callback("initializing", 0, total_samples)

    frame_index = 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break
        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        ok, frame = cap.retrieve()
        if not ok or frame is None:
            break

        crop = frame[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]
        masked_crop = cv2.bitwise_and(crop, crop, mask=roi_mask)
        blurred = cv2.GaussianBlur(masked_crop, (5, 5), 0)
        color_mask = _build_color_mask(blurred, color_model, roi_mask)

        learning_rate = 0.09 if processed_frames < 8 else 0.003
        foreground = background_subtractor.apply(blurred, learningRate=learning_rate)
        foreground_mask = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)[1]

        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            delta_mask = np.zeros_like(gray, dtype=np.uint8)
        else:
            delta = cv2.absdiff(gray, previous_gray)
            delta_mask = cv2.threshold(delta, 16, 255, cv2.THRESH_BINARY)[1]
        previous_gray = gray

        motion_mask = cv2.bitwise_or(foreground_mask, delta_mask)
        candidate_mask = cv2.bitwise_and(color_mask, motion_mask)
        candidate_mask = cv2.bitwise_and(candidate_mask, roi_mask)
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
        frame_points: list[list[int]] = []
        detection_debug: list[dict[str, Any]] = []
        for label in range(1, labels_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < component_min_area or area > component_max_area:
                continue

            width_px = int(stats[label, cv2.CC_STAT_WIDTH])
            height_px = int(stats[label, cv2.CC_STAT_HEIGHT])
            if width_px > 96 or height_px > 96:
                continue

            centroid_x = float(centroids[label][0])
            centroid_y = float(centroids[label][1])
            sample_x = int(round(centroid_x))
            sample_y = int(round(centroid_y))
            if sample_x < 0 or sample_x >= crop_width or sample_y < 0 or sample_y >= crop_height:
                continue

            full_x = centroid_x + crop_left
            full_y = centroid_y + crop_top
            point: Optional[list[int]] = None
            region = "ground"
            if ground_mask[sample_y, sample_x] > 0:
                point = _project_ground_point(full_x, full_y, ground_homography, field_width, field_height)
                ground_detections += 1
            elif left_projector is not None and left_projector.mask[sample_y, sample_x] > 0:
                point = _project_wall_point(full_x, full_y, left_projector, ground_homography, field_width, field_height)
                region = "left_wall"
                left_wall_detections += 1
            elif right_projector is not None and right_projector.mask[sample_y, sample_x] > 0:
                point = _project_wall_point(full_x, full_y, right_projector, ground_homography, field_width, field_height)
                region = "right_wall"
                right_wall_detections += 1

            if point is None:
                continue

            frame_points.append(point)
            detection_debug.append(
                {
                    "region": region,
                    "image_x": centroid_x,
                    "image_y": centroid_y,
                    "radius": math.sqrt(area / math.pi),
                    "point": point,
                }
            )

        frames.append(frame_points)
        total_detections += len(frame_points)
        if frame_points:
            frames_with_detections += 1

        time_sec = frame_index / max(effective_source_fps, 1.0)
        raw_lines.append(
            f"{processed_frames}\t{frame_index}\t{time_sec:.3f}\t{len(frame_points)}\t"
            + " ".join(f"{point[0]},{point[1]},{point[2]}" for point in frame_points)
        )
        _draw_overlay_frame(
            crop,
            detection_debug,
            overlay_writer,
            overlay_size,
            frame_index,
            time_sec,
            [[point[0] - crop_left, point[1] - crop_top] for point in ground_quad_pixels],
            [[point[0] - crop_left, point[1] - crop_top] for point in left_wall_quad_pixels] if left_wall_quad_pixels else None,
            [[point[0] - crop_left, point[1] - crop_top] for point in right_wall_quad_pixels] if right_wall_quad_pixels else None,
        )

        processed_frames += 1
        if progress_callback is not None:
            progress_callback("frames", min(processed_frames, total_samples), total_samples)
        frame_index += 1

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()
    elif (artifact_dir / "overlay-video.mp4").exists():
        (artifact_dir / "overlay-video.mp4").unlink(missing_ok=True)

    _create_overlay_images(field_image, frames, artifact_dir)

    air_profile_frames = [[[point[0], point[2]] for point in frame] for frame in frames]
    field_map_payload = {
        "imageWidth": int(field_width),
        "imageHeight": int(field_height),
        "fps": round(target_fps, 4),
        "frameCount": len(frames),
        "frames": frames,
    }
    air_profile_payload = {
        "fps": round(target_fps, 4),
        "frameCount": len(air_profile_frames),
        "wallSide": "mixed" if (left_wall_quad_pixels or right_wall_quad_pixels) else "bottom",
        "frames": air_profile_frames,
    }
    stats_payload = {
        "processor": "builtin",
        "backend": "internal",
        "usedExternalProcessor": False,
        "videoWidth": frame_width,
        "videoHeight": frame_height,
        "overlayWidth": overlay_size[0],
        "overlayHeight": overlay_size[1],
        "overlayFps": round(target_fps, 4),
        "overlayFrameCount": len(frames),
        "analysisFps": round(target_fps, 4),
        "sourceFps": round(effective_source_fps, 4),
        "sampleStrideFrames": frame_stride,
        "durationSec": round(frame_count / max(effective_source_fps, 1.0), 4) if frame_count > 0 else 0.0,
        "totalDetections": total_detections,
        "framesWithDetections": frames_with_detections,
        "groundDetections": ground_detections,
        "leftWallDetections": left_wall_detections,
        "rightWallDetections": right_wall_detections,
        "cropBox": {
            "x": crop_left,
            "y": crop_top,
            "width": crop_width,
            "height": crop_height,
        },
        "fuelBaseColor": fuel_base_color,
    }
    log_lines.extend(
        [
            f"Processed frames: {len(frames)}",
            f"Total detections: {total_detections}",
            f"Frames with detections: {frames_with_detections}",
            f"Ground detections: {ground_detections}",
            f"Left wall detections: {left_wall_detections}",
            f"Right wall detections: {right_wall_detections}",
        ]
    )

    (artifact_dir / "field-map.json").write_text(json.dumps(field_map_payload, separators=(",", ":")), encoding="utf-8")
    (artifact_dir / "air-profile.json").write_text(json.dumps(air_profile_payload, separators=(",", ":")), encoding="utf-8")
    (artifact_dir / "stats.json").write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
    (artifact_dir / "raw_data.txt").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    (artifact_dir / "fuel-process.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return {
        "stats": stats_payload,
        "log_lines": log_lines,
    }
