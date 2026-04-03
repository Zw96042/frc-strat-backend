from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FieldTemplateBank:
    roi: tuple[int, int, int, int]
    start_mean: np.ndarray
    end_mean: np.ndarray


class WatchbotStartDetector:
    def __init__(
        self,
        dataset_root: Path,
        rois: tuple[tuple[int, int, int, int], ...],
        template_size: tuple[int, int],
    ) -> None:
        start_paths = sorted(dataset_root.glob("*_start.jpg"))
        end_paths = sorted(dataset_root.glob("*_end.jpg"))
        if not start_paths or not end_paths:
            raise FileNotFoundError(f"Could not find watchbot field templates in {dataset_root}")

        self.template_size = template_size
        self.banks = tuple(self._build_bank(roi, start_paths, end_paths) for roi in rois)

    def _build_bank(
        self,
        roi: tuple[int, int, int, int],
        start_paths: list[Path],
        end_paths: list[Path],
    ) -> FieldTemplateBank:
        start_vectors = np.stack([self._vectorize_path(path, roi) for path in start_paths])
        end_vectors = np.stack([self._vectorize_path(path, roi) for path in end_paths])

        start_mean = start_vectors.mean(axis=0)
        start_mean /= np.linalg.norm(start_mean) + 1e-6
        end_mean = end_vectors.mean(axis=0)
        end_mean /= np.linalg.norm(end_mean) + 1e-6

        return FieldTemplateBank(
            roi=roi,
            start_mean=start_mean.astype(np.float32),
            end_mean=end_mean.astype(np.float32),
        )

    def _vectorize_path(self, path: Path, roi: tuple[int, int, int, int]) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read field template: {path}")
        return self._vectorize_frame(image, roi)

    def _vectorize_frame(self, image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = roi
        height, width = image.shape[:2]
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))
        w = max(1, min(int(w), width - x))
        h = max(1, min(int(h), height - y))

        crop = image[y:y + h, x:x + w]
        crop = cv2.resize(crop, self.template_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        crop -= crop.mean()
        crop /= np.linalg.norm(crop) + 1e-6
        return crop.reshape(-1)

    def score_frame(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        total_margin = 0.0
        for bank in self.banks:
            vector = self._vectorize_frame(gray, bank.roi)
            total_margin += float(np.dot(vector, bank.start_mean) - np.dot(vector, bank.end_mean))
        return total_margin
