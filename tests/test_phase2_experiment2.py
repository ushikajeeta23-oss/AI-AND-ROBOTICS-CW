from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from officemate.config import CLASSES
from phase2_build_experiment2 import (
    audit_review,
    build_plan,
    materialize_plan,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "results" / "phase2" / "review"
MANIFEST = ROOT / "data" / "office_dataset" / "manifest.csv"


def test_audit_review_reports_saved_and_missing_decisions():
    audit = audit_review(REVIEW, MANIFEST)

    assert audit["label_decisions"] == 49
    assert audit["pair_decisions"] == 0
    assert audit["label_counts"] == {"Correct": 3, "Incorrect": 42, "Uncertain": 4}
    assert audit["pair_counts"] == {}

    assert audit["high_total"] == 49
    assert audit["high_decided"] == 49
    assert audit["high_missing"] == []
    assert len(audit["label_missing"]) == 246
    assert len(audit["pair_missing"]) == 36
    assert audit["source_followup"] == 13


@pytest.mark.skipif(not (MANIFEST.parent / "train").is_dir(), reason="Requires local source photos")
def test_plan_is_conservative_and_keeps_near_duplicate_groups_together():
    plan, summary = build_plan(ROOT)

    assert summary["total"] == 1981
    assert summary["retained"] == 1416
    assert summary["excluded_incorrect"] == 42
    assert summary["quarantined"] == 523
    assert summary["status_counts"]["held_out_invalid"] == 6
    assert summary["status_counts"]["held_out_second_review"] == 3
    assert summary["held_out_uncertain"] == 4
    assert summary["held_out_unreviewed_label"] == 246
    assert summary["retained_reviewed_correct"] == 0
    assert summary["train"] == 1132
    assert summary["val"] == 284
    assert summary["review_conflicts"] == 3
    assert summary["manually_confirmed_groups"] == 0
    assert summary["suspected_groups_retained"] > 0
    assert any("SSIM" in edge["evidence"] for edge in summary["duplicate_edges"])

    assert set(summary["class_counts_train"]) == set(CLASSES)
    assert set(summary["class_counts_val"]) == set(CLASSES)
    assert all(summary["class_counts_train"][c] > 0 for c in CLASSES)
    assert all(summary["class_counts_val"][c] > 0 for c in CLASSES)

    retained = [row for row in plan if row["retained"]]
    assert all(row["source_split"] in {"train", "val"} for row in retained)
    assert all(row["new_split"] in {"train", "val"} for row in retained)
    assert all(row["new_path"].startswith((f"train\\", f"val\\")) for row in retained)

    held_out_or_excluded = [row for row in plan if not row["retained"]]
    assert len(held_out_or_excluded) == 565
    assert all(row["new_split"] == "" and row["new_path"] == "" for row in held_out_or_excluded)

    # No exact SHA-256 duplicate may appear in the retained set.
    retained_hashes = [row["sha256"] for row in retained]
    assert len(retained_hashes) == len(set(retained_hashes))

    # Every machine-detected near-duplicate cluster that has retained members
    # must stay wholly inside one new split.
    split_by_group = {}
    for row in retained:
        previous = split_by_group.setdefault(row["group_id"], row["new_split"])
        assert previous == row["new_split"], f"group crosses splits: {row['group_id']}"

    # The known cross-split clusters must no longer cross the new train/val boundary.
    for row in retained:
        if row["cluster_id"]:
            cluster_splits = {
                other["new_split"] for other in retained if other["cluster_id"] == row["cluster_id"]
            }
            assert len(cluster_splits) == 1, f"cluster {row['cluster_id']} crosses splits"

    # All ten classes are present.
    assert {row["class"] for row in retained} == set(CLASSES)


def test_materialize_copies_only_retained_images_and_preserves_originals(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "experiment2"
    colors = {
        "train-book.png": (255, 0, 0),
        "val-book.png": (0, 255, 0),
        "excluded-book.png": (0, 0, 255),
        "heldout-book.png": (255, 255, 0),
    }
    for name, color in colors.items():
        Image.new("RGB", (16, 16), color).save(source / name)
    original_bytes = {name: (source / name).read_bytes() for name in colors}
    hashes = {name: sha256(data).hexdigest() for name, data in original_bytes.items()}

    plan = [
        {
            "source_path": "train-book.png",
            "class": "book",
            "sha256": hashes["train-book.png"],
            "status": "retained_unflagged",
            "retained": True,
            "reason": "test",
            "new_split": "train",
            "new_path": "train\\book\\train-book.png",
        },
        {
            "source_path": "val-book.png",
            "class": "book",
            "sha256": hashes["val-book.png"],
            "status": "retained_unflagged",
            "retained": True,
            "reason": "test",
            "new_split": "val",
            "new_path": "val\\book\\val-book.png",
        },
        {
            "source_path": "excluded-book.png",
            "class": "book",
            "sha256": "c" * 64,
            "status": "excluded_incorrect",
            "retained": False,
            "reason": "test",
            "new_split": "",
            "new_path": "",
        },
        {
            "source_path": "heldout-book.png",
            "class": "book",
            "sha256": "d" * 64,
            "status": "held_out_unreviewed_label",
            "retained": False,
            "reason": "test",
            "new_split": "",
            "new_path": "",
        },
    ]

    materialize_plan(plan, source, output)

    assert (output / "train" / "book" / "train-book.png").is_file()
    assert (output / "val" / "book" / "val-book.png").is_file()
    assert not (output / "excluded-book.png").exists()
    assert not (output / "heldout-book.png").exists()
    assert (output / "manifest.csv").is_file()

    for name, expected in original_bytes.items():
        assert (source / name).read_bytes() == expected


def test_materialize_refuses_to_overwrite_a_nonempty_dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "experiment2"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError):
        materialize_plan([], source, output)

def test_cross_class_duplicate_evidence_quarantines_connected_candidates(tmp_path):
    from scripts.experiment2_integrity import group_candidates

    Image.new("RGB", (32, 32), (160, 50, 10)).save(tmp_path / "a.png")
    Image.new("RGB", (32, 32), (160, 50, 10)).save(tmp_path / "b.png")
    candidates = [
        {"source_path": name, "class": label, "retained": True, "cluster_id": "",
         "sha256": sha256((tmp_path / name).read_bytes()).hexdigest()}
        for name, label in (("a.png", "book"), ("b.png", "notebook"))
    ]
    edges = group_candidates(candidates, tmp_path, tmp_path, {"pairs": []}, {})
    assert any(edge["evidence"] == "sha256" for edge in edges)
    assert len({row["group_id"] for row in candidates}) == 1
    assert all(row["status"] == "held_out_cross_class" and not row["retained"] for row in candidates)
