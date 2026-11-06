"""Common prediction API for YOLOv8n-cls (.pt) and optional SVM (.joblib).

Reject a model with wrong class labels rather than silently showing a false
10-class result. A generic ImageNet checkpoint is NOT a trained OfficeMate model.
"""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from .config import CLASSES
from .features import extract_features

@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]

class Predictor:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f'Model not found: {self.model_path}. Run training first.')
        self.kind = self.model_path.suffix.lower()
        if self.kind == '.joblib':
            import joblib
            self.model = joblib.load(self.model_path)  # ONLY use trusted local files.
            names = [str(x) for x in self.model.classes_]
        elif self.kind == '.pt':
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError('Install requirements-yolo.txt for YOLO .pt models.') from exc
            self.model = YOLO(str(self.model_path))
            self.image_size = self.model.model.args.get("imgsz") or 224
            names = [str(self.model.names[i]) for i in sorted(self.model.names)]
        else:
            raise ValueError('Expected an OfficeMate .pt or trusted .joblib model.')
        if sorted(names) != sorted(CLASSES):
            raise ValueError(f'Model class labels do not match the report. Got {names}; expected {CLASSES}. '
                             'Train YOLOv8n-cls on all ten OfficeMate classes first.')

    def predict_rgb(self, rgb: np.ndarray) -> Prediction:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Image must be an RGB image with three channels.')
        if self.kind == '.joblib':
            values = self.model.predict_proba(extract_features(rgb).reshape(1, -1))[0]
            labels = [str(k) for k in self.model.classes_]
        else:
            import cv2
            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            result = self.model.predict(source=bgr, verbose=False, imgsz=self.image_size)[0]
            if result.probs is None:
                raise RuntimeError('These weights are not for classification (expected -cls).')
            values = result.probs.data.cpu().numpy()
            labels = [str(result.names[i]) for i in range(len(values))]
        scores = {label: float(score) for label, score in zip(labels, values)}
        highest = max(scores, key=scores.get)
        return Prediction(label=highest, confidence=scores[highest], probabilities=scores)
