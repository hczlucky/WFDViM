"""Model-independent BUSI weak-boundary robustness protocol."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pywt
from PIL import Image
from scipy import ndimage


Pair = tuple[str | Path, str | Path]


class BUSIWeakBoundaryProtocol:
    """Prepare 25 dB DWT perturbations and a train-calibrated subset."""

    def __init__(
        self,
        image_size: int = 256,
        target_psnr: float = 25.0,
        base_seed: int = 2026,
    ) -> None:
        self.image_size = int(image_size)
        self.target_psnr = float(target_psnr)
        self.base_seed = int(base_seed)
        self.wavelet = "bior2.2"
        self.dwt_mode = "periodization"
        self.score_quantile = 0.25
        self.smoothing_sigma = 1.0
        self.epsilon = 1e-6

    def load_pair(
        self,
        image_path: str | Path,
        mask_path: str | Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load one RGB image and binary mask at the protocol resolution."""
        size = (self.image_size, self.image_size)
        image = Image.open(image_path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(size, Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0 >= 0.5
        return image_array, mask_array

    def boundary_score(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Compute the normalized gradient magnitude around the GT boundary."""
        image = self._validate_image(image)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != image.shape[:2]:
            raise ValueError("The image and mask must have matching spatial dimensions.")

        gray = image.mean(axis=2)
        smooth = ndimage.gaussian_filter(gray, sigma=self.smoothing_sigma, mode="reflect")
        grad_x = ndimage.sobel(smooth, axis=1, mode="reflect")
        grad_y = ndimage.sobel(smooth, axis=0, mode="reflect")
        magnitude = np.hypot(grad_x, grad_y)

        boundary = mask ^ ndimage.binary_erosion(mask, border_value=0)
        boundary = ndimage.binary_dilation(boundary, iterations=1)
        if not np.any(boundary):
            return math.inf

        robust_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        return float(magnitude[boundary].mean() / max(robust_range, self.epsilon))

    def calibrate_threshold(self, training_pairs: Iterable[Pair]) -> float:
        """Calibrate the first-quartile threshold from training images only."""
        scores = []
        for image_path, mask_path in training_pairs:
            image, mask = self.load_pair(image_path, mask_path)
            score = self.boundary_score(image, mask)
            if np.isfinite(score):
                scores.append(score)
        if not scores:
            raise ValueError("No finite training boundary scores were produced.")
        return float(np.quantile(np.asarray(scores, dtype=np.float64), self.score_quantile))

    def select_weak_boundary_subset(
        self,
        training_pairs: Sequence[Pair],
        test_pairs: Sequence[Pair],
    ) -> tuple[float, list[dict[str, object]]]:
        """Return the train-calibrated threshold and per-test-image records."""
        threshold = self.calibrate_threshold(training_pairs)
        records = []
        for image_path, mask_path in test_pairs:
            image, mask = self.load_pair(image_path, mask_path)
            score = self.boundary_score(image, mask)
            records.append(
                {
                    "filename": Path(image_path).name,
                    "score": score,
                    "is_weak_boundary": bool(score <= threshold),
                }
            )
        return threshold, records

    def perturb(
        self,
        image: np.ndarray,
        filename: str,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Inject deterministic DWT high-frequency noise at the target PSNR."""
        image = self._validate_image(image)
        seed = self._stable_seed(filename)
        rng = np.random.default_rng(seed)
        delta = self._high_frequency_delta(image, rng)
        perturbed = self._scale_to_target_psnr(image, delta)
        metadata: dict[str, float | int] = {
            "seed": seed,
            "target_psnr": self.target_psnr,
            "actual_psnr": self._psnr(image, perturbed),
        }
        return perturbed, metadata

    def perturb_file(self, image_path: str | Path) -> tuple[np.ndarray, dict[str, float | int]]:
        """Load and perturb one image using its filename for deterministic noise."""
        size = (self.image_size, self.image_size)
        image = Image.open(image_path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return self.perturb(image_array, Path(image_path).name)

    def _high_frequency_delta(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        coefficients = [
            pywt.dwt2(image[:, :, channel], self.wavelet, mode=self.dwt_mode)
            for channel in range(image.shape[2])
        ]
        detail_shape = coefficients[0][1][0].shape
        orientation_noise = [
            rng.standard_normal(detail_shape, dtype=np.float32)
            for _ in range(3)
        ]

        reconstructed_channels = []
        for approximation, details in coefficients:
            perturbed_details = tuple(
                detail + self._robust_scale(detail) * orientation_noise[index]
                for index, detail in enumerate(details)
            )
            reconstructed = pywt.idwt2(
                (approximation, perturbed_details),
                self.wavelet,
                mode=self.dwt_mode,
            )
            reconstructed_channels.append(
                reconstructed[: self.image_size, : self.image_size]
            )

        reconstructed = np.stack(reconstructed_channels, axis=2).astype(np.float32)
        return reconstructed - image

    def _scale_to_target_psnr(self, image: np.ndarray, delta: np.ndarray) -> np.ndarray:
        target_mse = 10.0 ** (-self.target_psnr / 10.0)

        def clipped_mse(gain: float) -> float:
            candidate = np.clip(image + np.float32(gain) * delta, 0.0, 1.0)
            return float(np.mean((candidate.astype(np.float64) - image) ** 2))

        high = 1.0
        while clipped_mse(high) < target_mse and high < 1048576.0:
            high *= 2.0
        if clipped_mse(high) < target_mse:
            raise RuntimeError("Unable to reach the requested target PSNR.")

        low = 0.0
        for _ in range(50):
            middle = (low + high) / 2.0
            if clipped_mse(middle) < target_mse:
                low = middle
            else:
                high = middle
        return np.clip(image + np.float32(high) * delta, 0.0, 1.0).astype(np.float32)

    def _stable_seed(self, filename: str) -> int:
        payload = f"{self.base_seed}|{filename}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    @staticmethod
    def _robust_scale(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float32)
        median = np.median(values)
        scale = float(1.4826 * np.median(np.abs(values - median)))
        if scale <= 1e-8:
            scale = float(np.std(values))
        return max(scale, 1e-8)

    @staticmethod
    def _psnr(reference: np.ndarray, perturbed: np.ndarray) -> float:
        mse = float(np.mean((reference.astype(np.float64) - perturbed) ** 2))
        if mse <= 0.0:
            return math.inf
        return float(-10.0 * math.log10(mse))

    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        expected_shape = (self.image_size, self.image_size, 3)
        if image.shape != expected_shape:
            raise ValueError(f"Expected an RGB image with shape {expected_shape}.")
        if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("Image values must be finite and within [0, 1].")
        return image
