"""Phase 2 visual review desk.

Run from the project folder:
    .venv\\Scripts\\python.exe -m streamlit run scripts\\phase2_review_app.py --server.port 8502

The app is intentionally separate from app.py. It never deletes an image and
never marks one reviewed unless the reviewer chooses a decision and presses
Save. Original photographs, the Experiment 1 dataset, the model and the
Experiment 1 results are read-only inputs.
"""
from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from officemate.config import CLASSES  # noqa: E402
from officemate.review_store import (  # noqa: E402
    LABEL_DECISION_FIELDS,
    PAIR_DECISION_FIELDS,
    load_decisions,
    upsert_decision,
)

REVIEW_DIR = Path(os.environ.get("OFFICEMATE_REVIEW_DIR", ROOT / "results" / "phase2" / "review"))
PAYLOAD_PATH = REVIEW_DIR / "review_items.json"
DECISION_DIR = REVIEW_DIR / "decisions"
LABEL_DECISIONS = DECISION_DIR / "label_decisions.csv"
PAIR_DECISIONS = DECISION_DIR / "duplicate_decisions.csv"
SOURCE_FOLLOWUP = REVIEW_DIR / "source_followup.csv"
DATASET = ROOT / "data" / "office_dataset"

LABEL_OPTIONS = ["Correct", "Incorrect", "Uncertain"]
PAIR_OPTIONS = ["Duplicate", "Near-duplicate", "Distinct", "Uncertain"]
SPLITS = ["train", "val", "test"]


st.set_page_config(
    page_title="OfficeMate Phase 2 review",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --paper: #f5f2e9;
  --paper-deep: #e9e3d5;
  --ink: #1d241f;
  --muted: #647068;
  --line: #d4cdbd;
  --signal: #b8422c;
  --signal-soft: #f4ddd5;
  --teal: #1d6c64;
  --teal-soft: #dcece7;
  --mustard: #d7a62f;
  --mustard-soft: #f6ebc9;
}

.stApp {
  background: var(--paper);
  color: var(--ink);
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
  background: var(--paper-deep);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * {
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}
h1, h2, h3 {
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
  letter-spacing: -0.035em;
  color: var(--ink);
}
h2 { margin-bottom: 0.25rem; }
p, li, label, .stMarkdown { color: var(--ink); }
[data-testid="stImage"] img {
  border: 1px solid var(--line);
  background: white;
  box-shadow: 0 8px 28px rgba(29, 36, 31, 0.09);
}
.om-kicker {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.76rem;
  color: var(--muted);
  letter-spacing: 0.04em;
  margin-bottom: 0.2rem;
}
.om-title {
  font-size: clamp(1.45rem, 2.2vw, 2.1rem);
  line-height: 1.12;
  font-weight: 700;
  letter-spacing: -0.045em;
  margin: 0 0 0.5rem 0;
}
.om-badges { margin: 0.35rem 0 0.8rem 0; }
.om-badge {
  display: inline-block;
  padding: 0.16rem 0.55rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 600;
  margin-right: 0.3rem;
  border: 1px solid transparent;
}
.om-badge.high { background: var(--signal-soft); color: #8d2c1c; border-color: #e5b8ab; }
.om-badge.medium { background: var(--mustard-soft); color: #725313; border-color: #e8d18e; }
.om-badge.low { background: #ece8dc; color: #59635c; border-color: var(--line); }
.om-badge.teal { background: var(--teal-soft); color: #15544e; border-color: #b9d9d2; }
.om-badge.plain { background: #fffdf7; color: var(--muted); border-color: var(--line); }
.om-callout {
  border-left: 4px solid var(--signal);
  background: #fffaf3;
  padding: 0.9rem 1rem;
  margin: 0.5rem 0 1rem 0;
  line-height: 1.45;
}
.om-callout strong { display: block; margin-bottom: 0.2rem; }
.om-small { color: var(--muted); font-size: 0.86rem; }
.om-rule { border-top: 1px solid var(--line); margin: 1rem 0; }
.om-saved {
  background: var(--teal-soft);
  border: 1px solid #b9d9d2;
  padding: 0.6rem 0.75rem;
  color: #15544e;
  font-size: 0.9rem;
}
.om-path {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.76rem;
  color: var(--muted);
  overflow-wrap: anywhere;
}
div[data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.05rem;
}
.stButton button[kind="primary"] {
  background: var(--signal);
  border-color: var(--signal);
}
.stButton button[kind="primary"]:hover {
  background: #963522;
  border-color: #963522;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_payload(path_text: str, modified_ns: int) -> dict:
    del modified_ns
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


def esc(value) -> str:
    return html.escape(str(value or ""))


def badge(text: str, kind: str = "plain") -> str:
    return f'<span class="om-badge {kind}">{esc(text)}</span>'


def priority_kind(priority: str) -> str:
    return {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(priority, "plain")


def title_for(item: dict) -> str:
    return item.get("title") or Path(item.get("path", "")).name


def image_path(item: dict) -> Path:
    return DATASET / item.get("path", "")


def source_link(url: str, label: str = "Open Commons page") -> str:
    if not url:
        return '<span class="om-small">No Commons page recorded.</span>'
    return f'<a href="{esc(url)}" target="_blank" rel="noreferrer">{esc(label)}</a>'


def render_header(kicker: str, title: str, badges: list[str]) -> None:
    st.markdown(
        f'<div class="om-kicker">{esc(kicker)}</div>'
        f'<div class="om-title">{esc(title)}</div>'
        f'<div class="om-badges">{" ".join(badges)}</div>',
        unsafe_allow_html=True,
    )


def render_image(item: dict, caption: str) -> None:
    path = image_path(item)
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.error(f"Image file is missing: {path}")
        st.code(str(path))


def saved_caption(decision: dict | None) -> None:
    if not decision:
        st.markdown('<div class="om-small">Not reviewed yet.</div>', unsafe_allow_html=True)
        return
    note = f" · note: {decision.get('note')}" if decision.get("note") else ""
    st.markdown(
        '<div class="om-saved">'
        f"Saved: <strong>{esc(decision.get('decision'))}</strong>"
        f" · {esc(decision.get('reviewed_at'))}"
        f" · {esc(decision.get('reviewer') or 'unknown reviewer')}{esc(note)}"
        "</div>",
        unsafe_allow_html=True,
    )


def save_flash(message: str) -> None:
    st.session_state["flash"] = message


def show_flash() -> None:
    message = st.session_state.pop("flash", None)
    if message:
        st.success(message)


def matches_search(item: dict, search: str) -> bool:
    if not search:
        return True
    haystack = " ".join(
        str(item.get(key, "")) for key in (
            "title", "path", "class", "reason", "metadata_flags",
            "first_title", "second_title", "first_path", "second_path", "issue",
        )
    ).lower()
    return search.lower() in haystack


def label_queue_options(labels: list[dict]) -> dict[str, str]:
    def count(queue: str) -> int:
        return sum(1 for item in labels if queue in item["queues"])

    return {
        f"HIGH label problems ({count('high_label')})": "high_label",
        f"Model disagreements ({count('model_strong')})": "model_strong",
        f"MEDIUM label problems ({count('medium_label')})": "medium_label",
        f"All label review ({len(labels)})": "all",
        "Other / leakage / uncertain": "other",
    }


def in_queue(item: dict, queue_key: str) -> bool:
    if queue_key == "all":
        return True
    if queue_key == "other":
        return any(key in item["queues"] for key in ("other", "leakage", "model_uncertain"))
    return queue_key in item["queues"]


def filter_labels(labels: list[dict], queue_key: str, classes: list[str], splits: list[str],
                  unreviewed_only: bool, decisions: dict[str, dict], search: str) -> list[dict]:
    rows = []
    for item in labels:
        if not in_queue(item, queue_key):
            continue
        if classes and item["class"] not in classes:
            continue
        if splits and item["split"] not in splits:
            continue
        if unreviewed_only and item["item_id"] in decisions:
            continue
        if not matches_search(item, search):
            continue
        rows.append(item)
    return rows


def filter_pairs(pairs: list[dict], classes: list[str], splits: list[str],
                 unreviewed_only: bool, decisions: dict[str, dict], search: str) -> list[dict]:
    rows = []
    for pair in pairs:
        if classes and pair["class"] not in classes:
            continue
        if splits and not ({pair["first_split"], pair["second_split"]} & set(splits)):
            continue
        if unreviewed_only and pair["pair_id"] in decisions:
            continue
        if not matches_search(pair, search):
            continue
        rows.append(pair)
    return rows


def reset_index_on_filter_change(section: str, signature) -> None:
    key = f"{section}_filter_signature"
    index_key = f"{section}_index"
    if st.session_state.get(key) != signature:
        st.session_state[key] = signature
        st.session_state[index_key] = 0


def clamp_index(section: str, length: int) -> int:
    key = f"{section}_index"
    if length <= 0:
        st.session_state[key] = 0
        return 0
    current = int(st.session_state.get(key, 0))
    current = max(0, min(current, length - 1))
    st.session_state[key] = current
    return current


def nav_buttons(section: str, index: int, length: int) -> None:
    left, middle, right = st.columns([1, 2, 1])
    with left:
        if st.button("← Previous", disabled=index <= 0, key=f"{section}_prev"):
            st.session_state[f"{section}_index"] = index - 1
            st.rerun()
    with middle:
        st.markdown(
            f'<div class="om-small" style="text-align:center;padding-top:.45rem;">'
            f"Item {index + 1} of {length}</div>",
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Next →", disabled=index >= length - 1, key=f"{section}_next"):
            st.session_state[f"{section}_index"] = index + 1
            st.rerun()


if not PAYLOAD_PATH.exists():
    st.error(
        "The review payload is missing. Run "
        "`.venv\\Scripts\\python.exe scripts\\phase2_build_review_data.py` first."
    )
    st.stop()

payload = load_payload(str(PAYLOAD_PATH), PAYLOAD_PATH.stat().st_mtime_ns)
labels = payload["labels"]
pairs = payload["pairs"]
label_decisions = load_decisions(LABEL_DECISIONS)
pair_decisions = load_decisions(PAIR_DECISIONS, key="pair_id")

with st.sidebar:
    st.markdown("## Review desk")
    st.caption("Phase 2 · OfficeMate AI")
    reviewer = st.text_input("Reviewer name", value="student", key="reviewer_name")
    st.markdown('<div class="om-rule"></div>', unsafe_allow_html=True)

    high_ids = {item["item_id"] for item in labels if "high_label" in item["queues"]}
    high_done = sum(1 for item_id in high_ids if item_id in label_decisions)
    st.metric("HIGH label problems reviewed", f"{high_done} / {len(high_ids)}")
    st.progress(high_done / len(high_ids) if high_ids else 0.0)
    st.metric("All label items reviewed", f"{len(label_decisions)} / {len(labels)}")
    st.metric("Duplicate pairs reviewed", f"{len(pair_decisions)} / {len(pairs)}")

    st.markdown('<div class="om-rule"></div>', unsafe_allow_html=True)
    st.caption("Decisions are written only when you press Save.")
    st.markdown(
        '<div class="om-path">'
        f"{esc(LABEL_DECISIONS)}<br>{esc(PAIR_DECISIONS)}"
        "</div>",
        unsafe_allow_html=True,
    )
    if LABEL_DECISIONS.exists():
        st.download_button(
            "Download label decisions",
            LABEL_DECISIONS.read_bytes(),
            file_name="label_decisions.csv",
            mime="text/csv",
        )
    if PAIR_DECISIONS.exists():
        st.download_button(
            "Download duplicate decisions",
            PAIR_DECISIONS.read_bytes(),
            file_name="duplicate_decisions.csv",
            mime="text/csv",
        )
    if st.button("Reload review data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

st.markdown(
    f'<div class="om-kicker">Phase 2 · dataset cleaning</div>'
    f'<h1 style="margin:.1rem 0 .2rem 0;">Review the evidence before the dataset changes</h1>'
    f'<div class="om-small">'
    f'{len(labels)} label-review items · {len(pairs)} duplicate pairs · '
    f'original photographs are read-only'
    f"</div>",
    unsafe_allow_html=True,
)
show_flash()

label_tab, pair_tab, source_tab, help_tab = st.tabs(
    ["Label review", "Duplicate pairs", "Source follow-up", "How to use"]
)

with label_tab:
    queue_map = label_queue_options(labels)
    queue_choice = st.selectbox("Queue", list(queue_map.keys()), key="label_queue")
    queue_key = queue_map[queue_choice]

    f1, f2, f3, f4 = st.columns([1.3, 1, 1, 1.3])
    with f1:
        classes = st.multiselect("Class", list(CLASSES), default=list(CLASSES), key="label_classes")
    with f2:
        splits = st.multiselect("Split", SPLITS, default=SPLITS, key="label_splits")
    with f3:
        unreviewed_only = st.checkbox("Only unreviewed", value=True, key="label_unreviewed")
    with f4:
        search = st.text_input("Search title or path", key="label_search")

    filtered = filter_labels(
        labels, queue_key, classes, splits, unreviewed_only, label_decisions, search
    )
    reset_index_on_filter_change(
        "label", (queue_key, tuple(classes), tuple(splits), unreviewed_only, search)
    )

    if not filtered:
        st.info("No items match this filter. Change the queue or turn off “Only unreviewed”.")
    else:
        index = clamp_index("label", len(filtered))
        item = filtered[index]
        saved = label_decisions.get(item["item_id"])
        nav_buttons("label", index, len(filtered))

        left, right = st.columns([1.12, 0.88], gap="large")
        with left:
            render_image(item, title_for(item))
            st.markdown(
                f'<div class="om-path">{esc(item["path"])}</div>',
                unsafe_allow_html=True,
            )
        with right:
            badges = [
                badge(item["class"], "teal"),
                badge(item["split"], "plain"),
                badge(f'{item["priority"]} priority', priority_kind(item["priority"])),
            ]
            for queue in item["queues"]:
                badges.append(badge(queue.replace("_", " "), "plain"))
            render_header(f'Label review · {index + 1} of {len(filtered)}',
                          title_for(item), badges)

            st.markdown(
                f'<div class="om-callout"><strong>Why this is here</strong>'
                f'{esc(item["reason"])}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Evidence**")
            st.write(f"Current class: **{item['class']}**")
            st.write(f"Split: **{item['split']}**")
            if item.get("metadata_flags"):
                st.write(f"Metadata flags: {item['metadata_flags']}")
            if item.get("metadata_note"):
                st.caption(item["metadata_note"])
            if item.get("model_predicted"):
                conf = item.get("model_confidence")
                conf_text = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
                st.write(
                    f"Model triage: predicted **{item['model_predicted']}** "
                    f"(confidence {conf_text}); current label is **{item['class']}**."
                )
                st.caption("The model is a prompt to look, not proof of a label error.")
            if item.get("cluster_id") not in ("", None):
                st.write(
                    f"Near-duplicate cluster **{item['cluster_id']}** "
                    f"({item.get('cluster_size')} images; splits {item.get('cluster_splits')})."
                )
            if item.get("pair_ids"):
                st.write("Flagged in: " + ", ".join(item["pair_ids"]))
            if item.get("recommended_action"):
                st.caption(item["recommended_action"])

            st.markdown('<div class="om-rule"></div>', unsafe_allow_html=True)
            st.markdown("**Source and licence**")
            st.markdown(source_link(item.get("source_page", "")), unsafe_allow_html=True)
            st.write(f"Creator: {item.get('creator') or 'not recorded'}")
            st.write(f"Licence: {item.get('license') or 'not recorded'}")
            if item.get("license_url"):
                st.markdown(source_link(item["license_url"], "Licence URL"), unsafe_allow_html=True)
            st.caption(item.get("verification_status", ""))

            st.markdown('<div class="om-rule"></div>', unsafe_allow_html=True)
            saved_caption(saved)
            choice_key = f'label_choice::{item["item_id"]}'
            note_key = f'label_note::{item["item_id"]}'
            if choice_key not in st.session_state:
                st.session_state[choice_key] = (
                    saved.get("decision") if saved and saved.get("decision") in LABEL_OPTIONS else None
                )
            if note_key not in st.session_state:
                st.session_state[note_key] = saved.get("note", "") if saved else ""
            choice = st.radio(
                "Is the current label correct?",
                LABEL_OPTIONS,
                index=None,
                horizontal=True,
                key=choice_key,
            )
            note = st.text_area(
                "Note",
                key=note_key,
                placeholder="For Incorrect, say what the object actually is. For Uncertain, say what is unclear.",
                height=90,
            )
            if st.button(
                "Save decision",
                type="primary",
                disabled=choice is None,
                key=f'save_label::{item["item_id"]}',
                width="stretch",
            ):
                row = {
                    "item_id": item["item_id"],
                    "path": item["path"],
                    "class": item["class"],
                    "split": item["split"],
                    "priority": item["priority"],
                    "queues": "|".join(item["queues"]),
                    "title": title_for(item),
                    "reason": item["reason"],
                    "decision": choice,
                    "note": note.strip(),
                    "reviewer": (reviewer or "anonymous").strip(),
                    "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                upsert_decision(LABEL_DECISIONS, row, LABEL_DECISION_FIELDS, "item_id")
                save_flash(f"Saved “{title_for(item)}” as {choice}.")
                st.session_state["label_index"] = min(index + 1, len(filtered) - 1)
                st.rerun()

with pair_tab:
    p1, p2, p3, p4 = st.columns([1.3, 1, 1, 1.3])
    with p1:
        pair_classes = st.multiselect("Class", list(CLASSES), default=list(CLASSES), key="pair_classes")
    with p2:
        pair_splits = st.multiselect("Split", SPLITS, default=SPLITS, key="pair_splits")
    with p3:
        pair_unreviewed = st.checkbox("Only unreviewed", value=True, key="pair_unreviewed")
    with p4:
        pair_search = st.text_input("Search title or path", key="pair_search")

    filtered_pairs = filter_pairs(
        pairs, pair_classes, pair_splits, pair_unreviewed, pair_decisions, pair_search
    )
    reset_index_on_filter_change(
        "pair", (tuple(pair_classes), tuple(pair_splits), pair_unreviewed, pair_search)
    )

    if not filtered_pairs:
        st.info("No pairs match this filter. Change the filter or turn off “Only unreviewed”.")
    else:
        pair_index = clamp_index("pair", len(filtered_pairs))
        pair = filtered_pairs[pair_index]
        saved = pair_decisions.get(pair["pair_id"])
        nav_buttons("pair", pair_index, len(filtered_pairs))

        badges = [
            badge(pair["class"], "teal"),
            badge(f'{pair["first_split"]} ↔ {pair["second_split"]}', "plain"),
            badge(pair["machine_verdict"], "medium"),
        ]
        render_header(
            f'Duplicate pair {pair["pair_id"].split(":")[1]} · {pair_index + 1} of {len(filtered_pairs)}',
            f'{pair["class"]} pair: are these the same object or scene?',
            badges,
        )
        st.markdown(
            f'<div class="om-callout"><strong>Why this pair is here</strong>'
            f'{esc(pair["reason"])}</div>',
            unsafe_allow_html=True,
        )

        first_col, second_col = st.columns(2, gap="large")
        with first_col:
            first = {
                "path": pair["first_path"],
                "title": pair["first_title"],
                "class": pair["class"],
            }
            render_image(first, f'{pair["first_split"]} · {pair["first_title"]}')
            st.markdown(
                f'<div class="om-path">{esc(pair["first_path"])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                source_link(pair.get("first_source_page", ""), "Open first image on Commons"),
                unsafe_allow_html=True,
            )
        with second_col:
            second = {
                "path": pair["second_path"],
                "title": pair["second_title"],
                "class": pair["class"],
            }
            render_image(second, f'{pair["second_split"]} · {pair["second_title"]}')
            st.markdown(
                f'<div class="om-path">{esc(pair["second_path"])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                source_link(pair.get("second_source_page", ""), "Open second image on Commons"),
                unsafe_allow_html=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("dHash distance", pair["dhash_distance"])
        m2.metric("pHash distance", pair["phash_distance"])
        m3.metric("SSIM", f'{pair["ssim"]:.3f}' if isinstance(pair.get("ssim"), (int, float)) else "n/a")
        m4.metric("ORB inliers", pair["orb_inliers"])

        st.markdown('<div class="om-rule"></div>', unsafe_allow_html=True)
        saved_caption(saved)
        pair_choice_key = f'pair_choice::{pair["pair_id"]}'
        pair_note_key = f'pair_note::{pair["pair_id"]}'
        if pair_choice_key not in st.session_state:
            st.session_state[pair_choice_key] = (
                saved.get("decision") if saved and saved.get("decision") in PAIR_OPTIONS else None
            )
        if pair_note_key not in st.session_state:
            st.session_state[pair_note_key] = saved.get("note", "") if saved else ""
        pair_choice = st.radio(
            "What is the relationship?",
            PAIR_OPTIONS,
            index=None,
            horizontal=True,
            key=pair_choice_key,
        )
        pair_note = st.text_area(
            "Note",
            key=pair_note_key,
            placeholder="Example: same keyboard, different crop; keep in one split.",
            height=90,
        )
        if st.button(
            "Save decision",
            type="primary",
            disabled=pair_choice is None,
            key=f'save_pair::{pair["pair_id"]}',
            width="stretch",
        ):
            row = {
                "pair_id": pair["pair_id"],
                "class": pair["class"],
                "first_path": pair["first_path"],
                "second_path": pair["second_path"],
                "first_split": pair["first_split"],
                "second_split": pair["second_split"],
                "machine_verdict": pair["machine_verdict"],
                "reason": pair["reason"],
                "decision": pair_choice,
                "note": pair_note.strip(),
                "reviewer": (reviewer or "anonymous").strip(),
                "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            upsert_decision(PAIR_DECISIONS, row, PAIR_DECISION_FIELDS, "pair_id")
            save_flash(f"Saved {pair['pair_id']} as {pair_choice}.")
            st.session_state["pair_index"] = min(pair_index + 1, len(filtered_pairs) - 1)
            st.rerun()

with source_tab:
    followup = payload.get("source_followup", [])
    st.markdown("### Source and attribution follow-up")
    st.markdown(
        "These images are not label errors. They need a source check before the "
        "dataset is documented or published. The app does not save a decision here; "
        "verify the Commons page and record the result separately."
    )
    if not followup:
        st.success("No source follow-up items were found in the Phase 1 audit.")
    else:
        st.markdown(
            f"**{len(followup)} items** — "
            f"{sum(1 for item in followup if 'missing artist and credit' in item['issue'])} "
            "with no artist/credit, and "
            f"{sum(1 for item in followup if 'attribution licence has no URL' in item['issue'])} "
            "with a non-PD/CC0 licence but no licence URL (one image has both issues)."
        )
        table = [
            {
                "issue": item["issue"],
                "class": item["class"],
                "split": item["split"],
                "title": item["title"],
                "creator": item["creator"] or "not recorded",
                "licence": item["license"] or "not recorded",
                "source page": item["source_page"],
            }
            for item in followup
        ]
        st.dataframe(table, width="stretch", hide_index=True)

        selected = st.selectbox(
            "Inspect one source item",
            options=list(range(len(followup))),
            format_func=lambda i: (
                f"{followup[i]['issue']} · {followup[i]['class']} · "
                f"{followup[i]['title'][:70]}"
            ),
            key="source_selected",
        )
        item = followup[selected]
        left, right = st.columns([1, 1], gap="large")
        with left:
            render_image(item, title_for(item))
            st.markdown(
                f'<div class="om-path">{esc(item["path"])}</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(f"**Issue:** {esc(item['issue'])}")
            st.write(f"Class: **{item['class']}** · Split: **{item['split']}**")
            st.write(f"Creator: {item['creator'] or 'not recorded'}")
            st.write(f"Credit: {item['credit'] or 'not recorded'}")
            st.write(f"Licence: {item['license'] or 'not recorded'}")
            if item.get("license_url"):
                st.markdown(
                    source_link(item["license_url"], "Licence URL"),
                    unsafe_allow_html=True,
                )
            st.markdown(
                source_link(item.get("source_page", ""), "Open Commons page"),
                unsafe_allow_html=True,
            )
            st.caption(item.get("verification_status", ""))
            st.caption(
                "Do not invent missing attribution. Verify on the Commons page "
                "before publishing or redistributing the image."
            )

        if SOURCE_FOLLOWUP.exists():
            st.download_button(
                "Download source follow-up CSV",
                SOURCE_FOLLOWUP.read_bytes(),
                file_name="source_followup.csv",
                mime="text/csv",
            )

with help_tab:
    st.markdown(
        """
### What this app does

This is a **review desk**, not a cleaner. It shows the evidence found in Phase 1
and records your decisions. It does not delete images, change labels, rebuild
splits or touch Experiment 1.

### Label review

Start with the **49 HIGH label problems**. For each image:

- **Correct** — the current class is right.
- **Incorrect** — the photo is not the current class. Add a note saying what it is.
- **Uncertain** — you cannot tell, or it needs a second opinion.

The queue also includes MEDIUM/LOW items, model disagreements, and leakage
flags so you can inspect anything else that looks suspicious.

### Duplicate pairs

All 36 pairs are shown side by side.

- **Duplicate** — same photo or same scene; never split them.
- **Near-duplicate** — same object/session, different crop or angle; keep together.
- **Distinct** — different objects/scenes; they may be split.
- **Uncertain** — needs another look.

### Saving

A row is written only when you press **Save decision**. You can stop and resume;
the app loads saved decisions on the next start. The files are:

- `results/phase2/review/decisions/label_decisions.csv`
- `results/phase2/review/decisions/duplicate_decisions.csv`

### What happens next

After your review, Phase 2 will build a **new working dataset** from the original
photographs. Excluded images will be recorded with a reason. Confirmed duplicate
groups will stay in one split. The old test set will not be treated as a new
independent final test; a new test set must be collected separately.

No retraining will start until you approve the Phase 2 summary and Experiment 2
training configuration.
"""
    )
