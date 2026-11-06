"""Check that the code does not drift from the submitted student report."""
from pathlib import Path
from officemate.config import CLASSES, DEFAULT_THRESHOLD, DEFAULT_MODEL
from officemate.policy import decide


def test_report_yolo_threshold_and_classes():
    assert len(CLASSES) == 10
    assert DEFAULT_THRESHOLD == 0.80
    assert DEFAULT_MODEL.name == 'officemate_yolov8n_cls.pt'
    assert Path('scripts/train_yolo.py').read_text().count('yolov8n-cls.pt') >= 1


def test_report_approved_three_categories_only():
    for cls in ('bottle', 'mouse', 'stapler'):
        assert decide(cls, 0.95).action == 'CHECK_BEFORE_LIFT'
    for cls in ('bin','book','chair','keyboard','mug','notebook','phone'):
        assert decide(cls, 0.95).action == 'REPORT_ONLY'
    for cls in CLASSES:
        assert decide(cls, 0.799).action == 'REVIEW'


def test_physical_limits_are_not_guessed_from_images():
    d = decide('bottle', .99)
    assert '200 g' in d.reason and '25-65 mm' in d.reason and '300 mm' in d.reason
    assert 'cannot verify' in d.reason
