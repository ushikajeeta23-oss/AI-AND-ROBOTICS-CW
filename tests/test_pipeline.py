from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from officemate.config import CLASSES
from officemate.features import extract_features, load_image
from officemate.policy import decide
from officemate.dataset import split_raw, audit


def test_features_same_length_for_varied_image_sizes():
    small = extract_features(np.zeros((16, 28, 3), dtype=np.uint8))
    large = extract_features(np.full((360, 512, 3), 220, dtype=np.uint8))
    assert small.shape == large.shape
    assert np.all(np.isfinite(small))


def test_policy_unknown_low_confidence_personal():
    assert decide('phone', .95).action == 'REPORT_ONLY'
    assert decide('book', .30).action == 'REVIEW'
    assert decide('book', .95).action == 'REPORT_ONLY'
    assert decide('stapler', .95).action == 'CHECK_BEFORE_LIFT'
    assert '200 g' in decide('mouse', .96).reason
    assert decide('chair', .95).action == 'REPORT_ONLY'
    assert decide('bin', .95).action == 'REPORT_ONLY'


def test_invalid_image():
    with pytest.raises(ValueError):
        extract_features(np.zeros((12, 12), dtype=np.uint8))


def test_group_aware_split_and_hash_audit(tmp_path):
    raw, dest = tmp_path / 'raw', tmp_path / 'prepared'
    for class_i, cls in enumerate(CLASSES):
        (raw / cls).mkdir(parents=True)
        for group_i in range(7):
            for frame in range(2):
                # Distinct content by class, object group and frame for exact hash checks.
                arr = np.zeros((25, 25, 3), dtype=np.uint8)
                arr[:,:,0] = class_i * 28 + frame
                arr[:,:,1] = group_i * 31 + 3
                arr[:,:,2] = (class_i * 19 + group_i * 10 + frame) % 256
                Image.fromarray(arr).save(raw / cls / f'item{group_i:02}__view{frame}.png')
    records = split_raw(raw, dest)
    assert len(records) == 10 * 7 * 2
    counts, overlaps = audit(dest)
    assert not overlaps
    assert all(counts[s][c] > 0 for s in ('train','val','test') for c in CLASSES)
    assignments = {}
    for row in records:
        key = (row['class'],row['group'])
        assert key not in assignments or assignments[key] == row['split']
        assignments[key] = row['split']
    with pytest.raises(FileExistsError):
        split_raw(raw, dest)
