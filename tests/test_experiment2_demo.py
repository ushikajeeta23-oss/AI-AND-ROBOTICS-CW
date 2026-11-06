from pathlib import Path

import numpy as np
import pytest

from officemate.config import CLASSES, DEFAULT_MODEL
from officemate.inference import Predictor
from officemate.policy import decide
from train_experiment2 import validate_config


def test_dashboard_loads_checkpoint_and_camera_mode():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=45).run()
    assert not app.exception and not app.error
    assert [tab.label for tab in app.tabs] == ["Recognition", "Model performance", "Session history"]
    for model in app.selectbox[0].options:
        app.selectbox[0].set_value(model)
        app.radio[0].set_value("Camera snapshot").run()
        assert not app.exception and not app.error


def test_model_classes_invalid_image_and_score_boundary():
    model = Predictor(DEFAULT_MODEL)
    prediction = model.predict_rgb(np.zeros((160, 160, 3), dtype=np.uint8))
    assert set(prediction.probabilities) == set(CLASSES)
    assert sum(prediction.probabilities.values()) == pytest.approx(1, abs=1e-5)
    with pytest.raises(ValueError, match="RGB image"):
        model.predict_rgb(np.zeros((160, 160), dtype=np.uint8))
    assert decide("bottle", .8).action == "CHECK_BEFORE_LIFT"
    assert decide("bottle", .799999).action == "REVIEW"
    assert decide("phone", 1).action == "REPORT_ONLY"


def test_training_refuses_existing_output(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    output = tmp_path / "checkpoint.pt"
    output.write_bytes(b"previous checkpoint")
    with pytest.raises(FileExistsError):
        validate_config({"data": str(tmp_path)}, output)
    assert output.read_bytes() == b"previous checkpoint"
