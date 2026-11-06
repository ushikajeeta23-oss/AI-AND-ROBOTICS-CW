"""HOG shape features + HSV colour histograms for a portable baseline."""
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog


def load_image(path: str | Path) -> np.ndarray:
    """Read a photograph and return uint8 RGB. Raises for corrupt images."""
    path = Path(path)
    # imdecode handles unicode filenames more reliably than cv2.imread on Windows.
    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f'Unable to decode image: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """Transform RGB HxWx3 image to fixed-length, finite float32 feature vector."""
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected an RGB image of shape (height, width, 3).')
    if rgb.size == 0 or not np.isfinite(rgb).all():
        raise ValueError('Image is empty or contains invalid pixel values.')
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    shape = hog(grey, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist_parts = []
    for channel, nbins, bounds in ((0, 18, (0, 180)), (1, 16, (0, 256)), (2, 16, (0, 256))):
        histogram = cv2.calcHist([hsv], [channel], None, [nbins], list(bounds)).flatten()
        histogram /= max(1.0, float(histogram.sum()))
        hist_parts.append(histogram)
    return np.concatenate([shape, *hist_parts]).astype(np.float32)
