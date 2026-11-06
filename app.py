"""OfficeMate recognition dashboard for uploaded photos and camera snapshots."""
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

from officemate.config import DEFAULT_MODEL, DEFAULT_THRESHOLD
from officemate.inference import Predictor
from officemate.policy import decide

ROOT = Path(__file__).resolve().parent
MODELS = {
    "Experiment 1": (ROOT / DEFAULT_MODEL, ROOT / "results/REAL_YOLOV8N_TEST",
                     "Provisional historical test · potential duplicate leakage"),
    "Experiment 2": (ROOT / "models/experiment2/officemate_yolov8n_cls.pt",
                     ROOT / "results/experiment2/validation", "Development validation"),
}


@st.cache_resource(show_spinner="Loading classifier…")
def load_predictor(model_path):
    return Predictor(model_path)


st.set_page_config(page_title="OfficeMate AI", layout="wide")
st.title("OfficeMate AI")
st.caption("PDE3802 Assessment 1 | Office object recognition")
st.info("This app classifies one object per image. Handling requires independent "
        "authorization, weight, grip-width, reach and safety checks. This software cannot operate motors "
        "or verify a grasp.")

available = {label: info for label, info in MODELS.items() if info[0].is_file()}
if not available:
    st.error("No OfficeMate .pt checkpoint found. See README for model locations.")
    st.stop()
selected = st.sidebar.selectbox("YOLOv8n-cls checkpoint", list(reversed(available)))
path, evidence, protocol = available[selected]
st.sidebar.caption(f"{path.relative_to(ROOT).as_posix()} · Load only trusted .pt weights.")
threshold = st.sidebar.slider("Human-review threshold", 0.0, 1.0, DEFAULT_THRESHOLD, step=0.05)
mode = st.sidebar.radio("Input mode", ("Upload image(s)", "Camera snapshot"))
try:
    predictor = load_predictor(path)
except Exception as exc:
    st.error(f"Model failed to load: {exc}")
    st.stop()

recognition, performance, history = st.tabs(["Recognition", "Model performance", "Session history"])
with performance:
    st.subheader(selected)
    st.caption(protocol)
    metrics_path = evidence / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for column, (label, key) in zip(st.columns(4), (
            ("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Review rate", "review_rate"),
            ("Accepted accuracy", "accepted_accuracy"),
        )):
            value = metrics.get(key)
            column.metric(label, f"{value:.2%}" if value is not None else "Unavailable")
        st.caption(f'{metrics["n_images"]} images · evaluation threshold {metrics["review_threshold"]:.0%}')
        if (evidence / "confusion_matrix.png").is_file():
            st.image(str(evidence / "confusion_matrix.png"), caption="Rows are true classes. Columns are predictions.")
    else:
        st.caption("Evaluation artifacts are pending.")
    st.caption("Independent final test evaluation pending.")

records = st.session_state.setdefault("office_history", {})
with recognition:
    if mode == "Upload image(s)":
        files = st.file_uploader("Choose office photos", type=["png", "jpg", "jpeg", "webp"],
                                 accept_multiple_files=True)
    else:
        snapshot = st.camera_input("Place one office item in the frame")
        files = [snapshot] if snapshot is not None else []
    st.caption("Use one clearly visible object per photo. Model scores are uncalibrated.")
    rows = []
    for upload in files:
        try:
            raw = upload.getvalue()
            with Image.open(BytesIO(raw)) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
            result = predictor.predict_rgb(rgb)
            decision = decide(result.label, result.confidence, threshold)
        except Exception as exc:
            st.error(f"Could not process {upload.name}: {exc}")
            continue
        review = "Human review required" if decision.action == "REVIEW" else "Passed score threshold"
        preview, details = st.columns(2)
        with preview:
            st.image(rgb, caption=upload.name, use_container_width=True)
        with details:
            st.subheader(result.label.title())
            st.metric("Model confidence", f"{result.confidence:.1%}")
            st.write(f"**Recognition status** · {review}")
            st.write(f"**Handling policy** · {decision.action}")
            st.write(f"**Suggested destination** · {decision.destination}")
            st.caption(decision.reason)
            with st.expander("All ten class scores"):
                for name, score in sorted(result.probabilities.items(), key=lambda pair: -pair[1]):
                    st.progress(score, text=f"{name.title()} · {score:.1%}")
        row = {"file": upload.name, "class": result.label, "confidence": f"{result.confidence:.4f}",
               "decision": decision.action, "destination": decision.destination, "review": review,
               "model": selected, "threshold": threshold}
        rows.append(row)
        records[(selected, upload.name, sha256(raw).hexdigest(), threshold)] = row
        st.divider()
    if rows:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Download classification CSV", output.getvalue().encode("utf-8"),
                           "officemate_predictions.csv", "text/csv")
with history:
    if records:
        st.dataframe(list(reversed(records.values())), hide_index=True)
    else:
        st.caption("Session predictions will appear here.")
