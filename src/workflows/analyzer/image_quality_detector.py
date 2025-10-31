"""
Image Quality Detector for OCR Routing

This module detects image/document quality to help route to the appropriate OCR engine.
Low quality documents should be routed to advanced OCR (Qwen-VL) while high quality
documents can use classic OCR.

Quality metrics:
- Resolution (DPI)
- Sharpness (variance of Laplacian)
- Contrast (standard deviation)
- Noise level
- Skew/rotation
- Brightness distribution
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np

try:
    from PIL import Image
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


class ImageQualityMetrics:
    """Image quality metrics for a document."""

    def __init__(
        self,
        resolution_dpi: float,
        sharpness_score: float,
        contrast_score: float,
        noise_level: float,
        skew_angle: float,
        brightness_score: float,
        overall_quality_score: float,
        quality_category: str,
    ):
        self.resolution_dpi = resolution_dpi
        self.sharpness_score = sharpness_score
        self.contrast_score = contrast_score
        self.noise_level = noise_level
        self.skew_angle = skew_angle
        self.brightness_score = brightness_score
        self.overall_quality_score = overall_quality_score
        self.quality_category = quality_category

    def to_dict(self) -> Dict[str, any]:
        """Convert metrics to dictionary."""
        return {
            "resolution_dpi": self.resolution_dpi,
            "sharpness_score": self.sharpness_score,
            "contrast_score": self.contrast_score,
            "noise_level": self.noise_level,
            "skew_angle": self.skew_angle,
            "brightness_score": self.brightness_score,
            "overall_quality_score": self.overall_quality_score,
            "quality_category": self.quality_category,
        }


class ImageQualityDetector:
    """
    Detect image quality to help route OCR processing.

    Quality categories:
    - HIGH (score >= 0.7): Good resolution, sharp, good contrast → Classic OCR
    - MEDIUM (0.4 <= score < 0.7): Acceptable quality → Classic OCR with preprocessing
    - LOW (score < 0.4): Poor quality, blurry, low resolution → Advanced OCR (Qwen-VL)
    """

    def __init__(
        self,
        min_acceptable_dpi: int = 150,
        min_sharpness: float = 100.0,
        min_contrast: float = 30.0,
    ):
        """
        Initialize the quality detector.

        Args:
            min_acceptable_dpi: Minimum DPI for good quality (default: 150)
            min_sharpness: Minimum sharpness score for good quality (default: 100)
            min_contrast: Minimum contrast score for good quality (default: 30)
        """
        self.min_acceptable_dpi = min_acceptable_dpi
        self.min_sharpness = min_sharpness
        self.min_contrast = min_contrast
        self.logger = logging.getLogger(self.__class__.__name__)

        if not OPENCV_AVAILABLE:
            self.logger.warning("OpenCV not available. Image quality detection will use fallback methods.")

    def detect_quality(self, image_path: Union[str, Path]) -> ImageQualityMetrics:
        """
        Detect image quality metrics.

        Args:
            image_path: Path to the image file

        Returns:
            ImageQualityMetrics object
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.logger.info(f"Detecting quality for: {image_path}")

        # Load image
        img_pil = Image.open(image_path)

        # Detect resolution
        resolution_dpi = self._detect_resolution(img_pil)

        if OPENCV_AVAILABLE:
            # Convert to OpenCV format
            img_cv = cv2.imread(str(image_path))
            if img_cv is None:
                # Fallback: convert PIL to CV2
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Detect sharpness
            sharpness_score = self._detect_sharpness(gray)

            # Detect contrast
            contrast_score = self._detect_contrast(gray)

            # Detect noise
            noise_level = self._detect_noise(gray)

            # Detect skew
            skew_angle = self._detect_skew(gray)

            # Detect brightness
            brightness_score = self._detect_brightness(gray)
        else:
            # Fallback to PIL-based detection
            gray_array = np.array(img_pil.convert('L'))
            sharpness_score = self._detect_sharpness_fallback(gray_array)
            contrast_score = self._detect_contrast_fallback(gray_array)
            noise_level = 0.0  # Cannot detect without OpenCV
            skew_angle = 0.0   # Cannot detect without OpenCV
            brightness_score = self._detect_brightness_fallback(gray_array)

        # Calculate overall quality score (0.0 to 1.0)
        overall_quality_score = self._calculate_overall_quality(
            resolution_dpi,
            sharpness_score,
            contrast_score,
            noise_level,
            skew_angle,
            brightness_score
        )

        # Categorize quality
        quality_category = self._categorize_quality(overall_quality_score)

        metrics = ImageQualityMetrics(
            resolution_dpi=resolution_dpi,
            sharpness_score=sharpness_score,
            contrast_score=contrast_score,
            noise_level=noise_level,
            skew_angle=skew_angle,
            brightness_score=brightness_score,
            overall_quality_score=overall_quality_score,
            quality_category=quality_category,
        )

        self.logger.info(
            f"Quality detected: {quality_category} "
            f"(score={overall_quality_score:.3f}, "
            f"dpi={resolution_dpi:.0f}, "
            f"sharpness={sharpness_score:.1f}, "
            f"contrast={contrast_score:.1f})"
        )

        return metrics

    def _detect_resolution(self, img: Image.Image) -> float:
        """
        Detect image resolution (DPI).

        Args:
            img: PIL Image

        Returns:
            DPI value
        """
        # Get DPI from image metadata
        dpi = img.info.get('dpi', None)

        if dpi:
            # DPI is a tuple (x_dpi, y_dpi)
            return float(dpi[0])

        # Fallback: estimate from image size
        # Assume A4 page (8.27 x 11.69 inches)
        width, height = img.size
        estimated_dpi = width / 8.27  # Rough estimate

        self.logger.debug(f"DPI metadata not found, estimated: {estimated_dpi:.1f}")
        return estimated_dpi

    def _detect_sharpness(self, gray: np.ndarray) -> float:
        """
        Detect image sharpness using Laplacian variance.

        Higher values indicate sharper images.
        Typical values:
        - < 50: Very blurry
        - 50-100: Blurry
        - 100-500: Acceptable
        - > 500: Sharp

        Args:
            gray: Grayscale image array

        Returns:
            Sharpness score
        """
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Variance of Laplacian indicates sharpness
        sharpness = laplacian.var()

        return float(sharpness)

    def _detect_sharpness_fallback(self, gray: np.ndarray) -> float:
        """
        Fallback sharpness detection using numpy gradient.

        Args:
            gray: Grayscale image array

        Returns:
            Sharpness score
        """
        # Calculate gradient magnitude
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Variance indicates sharpness
        sharpness = magnitude.var()

        return float(sharpness)

    def _detect_contrast(self, gray: np.ndarray) -> float:
        """
        Detect image contrast using standard deviation.

        Higher values indicate better contrast.
        Typical values:
        - < 20: Very low contrast
        - 20-40: Low contrast
        - 40-60: Good contrast
        - > 60: High contrast

        Args:
            gray: Grayscale image array

        Returns:
            Contrast score
        """
        contrast = gray.std()
        return float(contrast)

    def _detect_contrast_fallback(self, gray: np.ndarray) -> float:
        """Fallback contrast detection."""
        return float(gray.std())

    def _detect_noise(self, gray: np.ndarray) -> float:
        """
        Detect noise level in the image.

        Uses median filtering to estimate noise.
        Lower values are better.

        Args:
            gray: Grayscale image array

        Returns:
            Noise level (0.0 = no noise, higher = more noise)
        """
        # Apply median filter
        median = cv2.medianBlur(gray, 5)

        # Calculate difference from original
        noise = np.abs(gray.astype(float) - median.astype(float))

        # Mean absolute deviation indicates noise
        noise_level = noise.mean()

        return float(noise_level)

    def _detect_skew(self, gray: np.ndarray) -> float:
        """
        Detect skew/rotation angle of the document.

        Args:
            gray: Grayscale image array

        Returns:
            Skew angle in degrees (0 = no skew)
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return 0.0

        # Calculate dominant angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            # Only consider angles close to horizontal
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return 0.0

        # Median angle
        skew_angle = np.median(angles)

        return float(abs(skew_angle))

    def _detect_brightness(self, gray: np.ndarray) -> float:
        """
        Detect brightness distribution quality.

        Good brightness should be centered around 128 (middle gray).

        Args:
            gray: Grayscale image array

        Returns:
            Brightness score (0-100, higher is better)
        """
        mean_brightness = gray.mean()

        # Optimal brightness is around 128 (middle gray)
        # Score is higher when closer to 128
        deviation_from_optimal = abs(mean_brightness - 128)

        # Convert to 0-100 scale (0 deviation = 100 score)
        brightness_score = max(0, 100 - deviation_from_optimal)

        return float(brightness_score)

    def _detect_brightness_fallback(self, gray: np.ndarray) -> float:
        """Fallback brightness detection."""
        mean_brightness = gray.mean()
        deviation_from_optimal = abs(mean_brightness - 128)
        brightness_score = max(0, 100 - deviation_from_optimal)
        return float(brightness_score)

    def _calculate_overall_quality(
        self,
        resolution_dpi: float,
        sharpness_score: float,
        contrast_score: float,
        noise_level: float,
        skew_angle: float,
        brightness_score: float,
    ) -> float:
        """
        Calculate overall quality score (0.0 to 1.0).

        Weighted average of individual metrics.

        Returns:
            Overall quality score
        """
        # Normalize each metric to 0-1 scale

        # Resolution: good if >= min_acceptable_dpi
        resolution_norm = min(1.0, resolution_dpi / self.min_acceptable_dpi)

        # Sharpness: good if >= min_sharpness
        sharpness_norm = min(1.0, sharpness_score / self.min_sharpness)

        # Contrast: good if >= min_contrast
        contrast_norm = min(1.0, contrast_score / self.min_contrast)

        # Noise: good if low (inverse)
        noise_norm = max(0.0, 1.0 - (noise_level / 50.0))  # 50 is arbitrary threshold

        # Skew: good if low (inverse)
        skew_norm = max(0.0, 1.0 - (abs(skew_angle) / 10.0))  # 10 degrees threshold

        # Brightness: already normalized to 0-100
        brightness_norm = brightness_score / 100.0

        # Weighted average
        weights = {
            'resolution': 0.25,
            'sharpness': 0.25,
            'contrast': 0.20,
            'noise': 0.15,
            'skew': 0.10,
            'brightness': 0.05,
        }

        overall_score = (
            weights['resolution'] * resolution_norm +
            weights['sharpness'] * sharpness_norm +
            weights['contrast'] * contrast_norm +
            weights['noise'] * noise_norm +
            weights['skew'] * skew_norm +
            weights['brightness'] * brightness_norm
        )

        return float(overall_score)

    def _categorize_quality(self, score: float) -> str:
        """
        Categorize quality based on overall score.

        Args:
            score: Overall quality score (0.0 to 1.0)

        Returns:
            Quality category: HIGH, MEDIUM, or LOW
        """
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def get_recommended_ocr_engine(self, quality_metrics: ImageQualityMetrics) -> str:
        """
        Get recommended OCR engine based on quality.

        Args:
            quality_metrics: Image quality metrics

        Returns:
            Recommended OCR engine: 'classic_ocr', 'classic_ocr_with_preprocessing', or 'qwen_vl'
        """
        if quality_metrics.quality_category == "HIGH":
            return "classic_ocr"
        elif quality_metrics.quality_category == "MEDIUM":
            return "classic_ocr_with_preprocessing"
        else:  # LOW
            return "qwen_vl"


def detect_image_quality(image_path: Union[str, Path]) -> ImageQualityMetrics:
    """
    Convenience function to detect image quality.

    Args:
        image_path: Path to the image file

    Returns:
        ImageQualityMetrics object
    """
    detector = ImageQualityDetector()
    return detector.detect_quality(image_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python image_quality_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        detector = ImageQualityDetector()
        metrics = detector.detect_quality(image_path)

        print(f"Image Quality Analysis: {image_path}")
        print(f"=" * 60)
        print(f"Resolution (DPI):      {metrics.resolution_dpi:.1f}")
        print(f"Sharpness Score:       {metrics.sharpness_score:.1f}")
        print(f"Contrast Score:        {metrics.contrast_score:.1f}")
        print(f"Noise Level:           {metrics.noise_level:.1f}")
        print(f"Skew Angle:            {metrics.skew_angle:.1f}°")
        print(f"Brightness Score:      {metrics.brightness_score:.1f}")
        print(f"")
        print(f"Overall Quality Score: {metrics.overall_quality_score:.3f}")
        print(f"Quality Category:      {metrics.quality_category}")
        print(f"")
        print(f"Recommended OCR:       {detector.get_recommended_ocr_engine(metrics)}")

    except Exception as e:
        print(f"Error analyzing image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
