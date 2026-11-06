from pathlib import Path

from officemate.review_store import (
    LABEL_DECISION_FIELDS,
    load_decisions,
    upsert_decision,
)
from phase2_build_review_data import build_review_payload


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "results" / "phase1"


def test_payload_has_36_pairs_and_the_49_high_label_items():
    payload = build_review_payload(PHASE1, ROOT / "data" / "office_dataset" / "manifest.csv", ROOT)

    assert len(payload["pairs"]) == 36
    assert sum(1 for item in payload["labels"] if "high_label" in item["queues"]) == 49
    assert sum(1 for item in payload["labels"] if "model_strong" in item["queues"]) == 10
    assert len(payload["labels"]) >= 49
    assert len(payload["labels"]) <= 400, "the review queue must stay focused, not all 1,981 images"
    assert all(item["queues"] for item in payload["labels"])
    assert len(payload["source_followup"]) == 13
    assert sum(1 for item in payload["source_followup"] if "missing artist and credit" in item["issue"]) == 11
    assert sum(1 for item in payload["source_followup"] if "attribution licence has no URL" in item["issue"]) == 3

    for item in payload["labels"]:
        assert item["item_id"]
        assert item["path"]
        assert item["class"]
        assert item["split"] in {"train", "val", "test"}
        assert item["reason"]
        if (ROOT / "data/office_dataset/train").is_dir():
            assert (ROOT / "data" / "office_dataset" / item["path"]).exists()

    for pair in payload["pairs"]:
        assert pair["pair_id"]
        assert pair["first_path"] and pair["second_path"]
        assert pair["class"]
        assert pair["machine_verdict"]
        assert pair["reason"]
        if (ROOT / "data/office_dataset/train").is_dir():
            assert (ROOT / "data" / "office_dataset" / pair["first_path"]).exists()
        if (ROOT / "data/office_dataset/train").is_dir():
            assert (ROOT / "data" / "office_dataset" / pair["second_path"]).exists()


def test_upsert_decision_updates_one_row_and_keeps_others(tmp_path):
    path = tmp_path / "label_decisions.csv"
    first = {field: "" for field in LABEL_DECISION_FIELDS}
    first.update({
        "item_id": "label:a", "path": "a.jpg", "class": "book", "split": "train",
        "decision": "correct", "note": "", "reviewed_at": "2026-01-01T10:00:00",
    })
    second = dict(first, item_id="label:b", path="b.jpg", decision="uncertain")

    upsert_decision(path, first, LABEL_DECISION_FIELDS, "item_id")
    upsert_decision(path, second, LABEL_DECISION_FIELDS, "item_id")
    assert set(load_decisions(path)) == {"label:a", "label:b"}

    changed = dict(first, decision="incorrect", note="not a book")
    upsert_decision(path, changed, LABEL_DECISION_FIELDS, "item_id")

    decisions = load_decisions(path)
    assert len(decisions) == 2
    assert decisions["label:a"]["decision"] == "incorrect"
    assert decisions["label:a"]["note"] == "not a book"
    assert decisions["label:b"]["decision"] == "uncertain"


def test_building_review_data_does_not_create_decisions(tmp_path):
    payload = build_review_payload(PHASE1, ROOT / "data" / "office_dataset" / "manifest.csv", ROOT)
    out = tmp_path / "review"
    out.mkdir()
    (out / "review_items.json").write_text("{}", encoding="utf-8")

    assert payload["labels"]
    assert not (out / "label_decisions.csv").exists()
    assert not (out / "duplicate_decisions.csv").exists()


def test_review_app_starts_without_exception():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "scripts" / "phase2_review_app.py"), default_timeout=30)
    app.run()

    assert not app.exception
    assert len(app.tabs) == 4
    assert len(app.radio) >= 1
    assert any("Review the evidence" in str(block.value) for block in app.markdown)


def test_review_app_saves_only_after_a_decision(tmp_path, monkeypatch):
    import shutil

    from streamlit.testing.v1 import AppTest

    review = tmp_path / "review"
    review.mkdir()
    shutil.copy(ROOT / "results" / "phase2" / "review" / "review_items.json", review / "review_items.json")
    monkeypatch.setenv("OFFICEMATE_REVIEW_DIR", str(review))

    app = AppTest.from_file(str(ROOT / "scripts" / "phase2_review_app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    decisions = review / "decisions" / "label_decisions.csv"
    assert not decisions.exists(), "loading the app must not mark anything reviewed"

    app.radio[0].set_value("Correct").run()
    assert not decisions.exists(), "choosing a value must not save until Save is pressed"

    save_buttons = [button for button in app.button if button.label == "Save decision"]
    assert save_buttons, "the label review needs a Save decision button"
    save_buttons[0].click().run()
    assert not app.exception

    rows = decisions.read_text(encoding="utf-8")
    assert "Correct" in rows
    assert len(rows.splitlines()) == 2


def test_review_app_saves_pair_decision_separately(tmp_path, monkeypatch):
    import shutil

    from streamlit.testing.v1 import AppTest

    review = tmp_path / "review"
    review.mkdir()
    shutil.copy(ROOT / "results" / "phase2" / "review" / "review_items.json", review / "review_items.json")
    monkeypatch.setenv("OFFICEMATE_REVIEW_DIR", str(review))

    app = AppTest.from_file(str(ROOT / "scripts" / "phase2_review_app.py"), default_timeout=30)
    app.run()
    assert not app.exception

    app.radio[1].set_value("Near-duplicate").run()
    save_buttons = [button for button in app.button if button.label == "Save decision"]
    assert len(save_buttons) >= 2
    save_buttons[1].click().run()
    assert not app.exception

    pair_decisions = review / "decisions" / "duplicate_decisions.csv"
    assert pair_decisions.exists()
    assert "Near-duplicate" in pair_decisions.read_text(encoding="utf-8")
    assert not (review / "decisions" / "label_decisions.csv").exists()



