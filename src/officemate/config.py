"""Values taken from the group's OfficeMate AI system-design report (5 Nov 2026).

Only the classification portion is implemented in Part B. A prediction is NOT a
command to move a physical robot.
"""
from pathlib import Path

CLASSES = ('bin', 'book', 'bottle', 'chair', 'keyboard', 'mouse', 'mug', 'notebook', 'phone', 'stapler')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
DEFAULT_DATASET = Path('data/office_dataset')
DEFAULT_MODEL = Path('models/officemate_yolov8n_cls.pt')
DEFAULT_THRESHOLD = 0.80  # Recognition-review score, NOT model test accuracy.
TARGET_ACCURACY = 0.90    # Proposed success condition, not a measured result.
MAX_ITEM_MASS_G = 200
GRIP_MIN_MM = 25
GRIP_MAX_MM = 65
WORKSPACE_REACH_MM = 300
SEED = 42
