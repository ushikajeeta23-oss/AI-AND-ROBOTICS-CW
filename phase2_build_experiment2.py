#!/usr/bin/env python
"""Build a separate, leakage-aware Experiment 2 dataset from saved Phase 2 decisions.

This script is deliberately conservative:

* The original ``data/office_dataset`` is read-only.
* The Experiment 1 model and results are never touched.
* A saved ``Incorrect`` label decision excludes an image from Experiment 2.
* A saved ``Uncertain`` label decision holds the image out until resolved.
* A Phase 1 label-review item with no saved decision is held out; it is not
  silently treated as correct.
* Images outside the Phase 1 label-review queue are retained as the unreviewed
  baseline. They are not marked as human-reviewed.
* Machine-detected near-duplicate clusters are kept in one new split.
* No new test split is created from old test images.

Usage:
    python phase2_build_experiment2.py --dry-run
    python phase2_build_experiment2.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent

from officemate.config import CLASSES, IMAGE_EXTENSIONS, SEED
from officemate.features import load_image


RETAINED_STATUSES = {"retained_unflagged", "retained_reviewed_correct"}
HELD_OUT_STATUSES = {
    "held_out_uncertain", "held_out_unreviewed_label", "held_out_second_review",
    "held_out_original_test", "held_out_cross_class", "held_out_invalid",
}
SECOND_REVIEW = {"gnome-dev-mouse-optical.png", "mouse spider scale.jpg", "mug shot.jpg"}
EXCLUDED_STATUSES = {"excluded_incorrect", "excluded_exact_duplicate"}

PLAN_FIELDS = [
    "image_id",
    "source_path",
    "source_split",
    "class",
    "sha256",
    "manifest_sha256",
    "status",
    "retained",
    "reason",
    "label_decision",
    "label_note",
    "label_review_status",
    "pair_ids",
    "pair_decision",
    "pair_review_status",
    "cluster_id",
    "cluster_size",
    "group_id",
    "group_size",
    "new_split",
    "new_path",
    "title",
    "source_page",
    "creator",
    "credit",
    "license",
    "license_url",
    "license_family",
    "missing_artist",
    "missing_credit",
    "missing_license_url",
    "missing_license_url_attribution",
    "nonstandard_license",
    "source_followup_issue",
    "review_conflict",
]

NONSTANDARD_LICENSE_FAMILIES = {"GPL", "GFDL", "other"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _manifest_path(value: str) -> Path:
    """Turn a manifest-relative Windows or POSIX path into a Path."""
    return Path(value.replace("\\", "/"))


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {field: row.get(field, "") for field in PLAN_FIELDS}
    out["retained"] = _as_bool(out["retained"])
    return out


def audit_review(review_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Report exactly which saved decisions exist and which reviews are missing."""
    payload_path = review_dir / "review_items.json"
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    labels = payload.get("labels", [])
    pairs = payload.get("pairs", [])
    source_followup = payload.get("source_followup", [])

    label_decisions_path = review_dir / "decisions" / "label_decisions.csv"
    pair_decisions_path = review_dir / "decisions" / "duplicate_decisions.csv"
    label_decisions = {row["item_id"]: row for row in _read_csv(label_decisions_path)}
    pair_decisions = {row["pair_id"]: row for row in _read_csv(pair_decisions_path)}

    label_counts = Counter(row.get("decision", "") for row in label_decisions.values())
    pair_counts = Counter(row.get("decision", "") for row in pair_decisions.values())

    high_items = [item for item in labels if "high_label" in item.get("queues", [])]
    high_missing = [item for item in high_items if item["item_id"] not in label_decisions]
    label_missing = [item for item in labels if item["item_id"] not in label_decisions]
    pair_missing = [pair for pair in pairs if pair["pair_id"] not in pair_decisions]

    manifest = _read_csv(manifest_path)
    known_paths = {row["path"] for row in manifest}
    missing_images = []
    for item in labels:
        if item["path"] not in known_paths:
            missing_images.append(item["path"])
    for pair in pairs:
        for key in ("first_path", "second_path"):
            if pair[key] not in known_paths:
                missing_images.append(pair[key])

    return {
        "label_decisions": len(label_decisions),
        "pair_decisions": len(pair_decisions),
        "label_counts": dict(label_counts),
        "pair_counts": dict(pair_counts),
        "high_total": len(high_items),
        "high_decided": len(high_items) - len(high_missing),
        "high_missing": high_missing,
        "label_missing": label_missing,
        "pair_missing": pair_missing,
        "source_followup": len(source_followup),
        "missing_images": sorted(set(missing_images)),
    }


def _load_clusters(phase1_dir: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    path_to_cluster: dict[str, str] = {}
    clusters: dict[str, dict[str, str]] = {}
    for row in _read_csv(phase1_dir / "duplicate_clusters.csv"):
        cluster_id = row["cluster_id"]
        clusters[cluster_id] = row
        for path in row["paths"].split("|"):
            if path:
                path_to_cluster[path] = cluster_id
    return path_to_cluster, clusters


def _load_licensing(phase1_dir: Path) -> dict[str, dict[str, str]]:
    return {row["path"]: row for row in _read_csv(phase1_dir / "licensing_audit.csv")}


def _load_source_followup(review_dir: Path) -> dict[str, str]:
    rows = _read_csv(review_dir / "source_followup.csv")
    return {row["path"]: row.get("issue", "") for row in rows}


def _label_review_status(decision: dict[str, str] | None, in_queue: bool) -> str:
    if decision:
        return {
            "Correct": "reviewed_correct",
            "Incorrect": "reviewed_incorrect",
            "Uncertain": "reviewed_uncertain",
        }.get(decision.get("decision", ""), "reviewed_other")
    return "unreviewed_flagged" if in_queue else "not_in_review_queue"


def _pair_review_status(decisions: list[str], in_pair_review: bool) -> str:
    if not in_pair_review:
        return "not_in_pair_review"
    if not decisions:
        return "unreviewed_pair"
    unique = set(decisions)
    if unique <= {"Duplicate", "Near-duplicate"}:
        return "reviewed_duplicate_or_near_duplicate"
    if unique == {"Distinct"}:
        return "reviewed_distinct"
    if "Uncertain" in unique:
        return "reviewed_uncertain"
    return "reviewed_mixed"


def _status_and_reason(
    path: str,
    label_decision: dict[str, str] | None,
    label_paths: set[str],
) -> tuple[str, str]:
    if label_decision:
        decision = label_decision.get("decision", "")
        if decision == "Incorrect":
            return (
                "excluded_incorrect",
                "User marked the current label Incorrect; excluded from Experiment 2.",
            )
        if decision == "Uncertain":
            return (
                "held_out_uncertain",
                "User marked the label Uncertain; held out until the label is resolved.",
            )
        if decision == "Correct":
            return "retained_reviewed_correct", "User marked the current label Correct."
        return (
            "held_out_uncertain",
            f"Unexpected saved label decision {decision!r}; held out pending review.",
        )
    if path in label_paths:
        return (
            "held_out_unreviewed_label",
            "Phase 1 flagged this image for label review; no saved decision was found.",
        )
    return (
        "retained_unflagged",
        "Not flagged in the Phase 1 label review; retained as unreviewed baseline.",
    )


def _assign_splits(retained: list[dict[str, Any]], seed: int) -> None:
    if not retained:
        return
    x = np.zeros(len(retained), dtype=np.int8)
    y = [row["class"] for row in retained]
    groups = [row["group_id"] for row in retained]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(x, y, groups))

    for indices, split in ((train_idx, "train"), (val_idx, "val")):
        for index in indices:
            row = retained[index]
            row["new_split"] = split
            row["new_path"] = f"{split}\\{row['class']}\\{Path(row['source_path']).name}"

    # Hard checks: every class must be represented, and no group may cross.
    for split in ("train", "val"):
        present = {row["class"] for row in retained if row["new_split"] == split}
        missing = set(CLASSES) - present
        if missing:
            raise ValueError(f"New {split} split is missing classes: {sorted(missing)}")
    split_by_group: dict[str, str] = {}
    for row in retained:
        previous = split_by_group.setdefault(row["group_id"], row["new_split"])
        if previous != row["new_split"]:
            raise ValueError(f"Group {row['group_id']} crosses train/val")


def build_plan(
    root: Path = ROOT,
    review_dir: Path | None = None,
    phase1_dir: Path | None = None,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    review_dir = Path(review_dir) if review_dir else root / "results" / "phase2" / "review"
    phase1_dir = Path(phase1_dir) if phase1_dir else root / "results" / "phase1"
    manifest_path = root / "data" / "office_dataset" / "manifest.csv"

    audit = audit_review(review_dir, manifest_path)
    manifest = _read_csv(manifest_path)
    payload = json.loads((review_dir / "review_items.json").read_text(encoding="utf-8"))
    label_items = {item["path"]: item for item in payload.get("labels", [])}
    label_paths = set(label_items)
    label_decisions = {
        row["item_id"]: row
        for row in _read_csv(review_dir / "decisions" / "label_decisions.csv")
    }
    pair_decisions = {
        row["pair_id"]: row
        for row in _read_csv(review_dir / "decisions" / "duplicate_decisions.csv")
    }

    pair_ids_by_path: dict[str, list[str]] = defaultdict(list)
    pair_decisions_by_path: dict[str, list[str]] = defaultdict(list)
    for pair in payload.get("pairs", []):
        pair_id = pair["pair_id"]
        decision = pair_decisions.get(pair_id, {}).get("decision", "")
        for key in ("first_path", "second_path"):
            path = pair[key]
            pair_ids_by_path[path].append(pair_id)
            if decision:
                pair_decisions_by_path[path].append(decision)

    path_to_cluster, clusters = _load_clusters(phase1_dir)
    licensing = _load_licensing(phase1_dir)
    source_followup = _load_source_followup(review_dir)

    plan: list[dict[str, Any]] = []
    for row in manifest:
        path = row["path"]
        label_decision = label_decisions.get(f"label:{path}")
        status, reason = _status_and_reason(path, label_decision, label_paths)
        title = (label_decision or {}).get("title", "").removeprefix("File:").casefold()
        if status == "retained_reviewed_correct" and title in SECOND_REVIEW:
            status = "held_out_second_review"
            reason = "Saved Correct decision preserved. Physical object relevance requires second review."
        if status in RETAINED_STATUSES and row["split"] == "test":
            status, reason = "held_out_original_test", "Former test image reserved outside Experiment 2."
        retained = status in RETAINED_STATUSES

        label_item = label_items.get(path)
        review_conflict = ""
        if label_decision and label_decision.get("decision") == "Correct" and label_item:
            review_conflict = f"Phase 1 evidence: {label_item.get('reason', '')}"

        pair_ids = pair_ids_by_path.get(path, [])
        pair_values = pair_decisions_by_path.get(path, [])
        pair_status = _pair_review_status(pair_values, bool(pair_ids))

        cluster_id = path_to_cluster.get(path, "")
        cluster_size = int(clusters[cluster_id]["n_images"]) if cluster_id else 0
        group_id = f"cluster:{cluster_id}" if cluster_id else f"image:{path}"

        source_file = root / "data" / "office_dataset" / _manifest_path(path)
        actual_sha256 = sha256(source_file.read_bytes()).hexdigest() if source_file.is_file() else ""
        parts = _manifest_path(path).parts
        if retained and (
            row["class"] not in CLASSES or parts[:2] != (row["split"], row["class"])
            or len(parts) != 3 or source_file.suffix.lower() not in IMAGE_EXTENSIONS
            or actual_sha256 != row["sha256"]
        ):
            status, reason, retained = "held_out_invalid", "Invalid label, path or source checksum.", False

        source = licensing.get(path, {})
        creator = source.get("api_artist", "") or ""
        credit = source.get("api_credit", "") or ""
        license_family = source.get("license_family", "") or ""
        missing_license_url = _as_bool(source.get("missing_license_url", "False"))
        missing_license_url_attribution = (
            missing_license_url and license_family != "public domain / CC0"
        )

        plan.append(
            {
                "image_id": path,
                "source_path": path,
                "source_split": row["split"],
                "class": row["class"],
                "sha256": actual_sha256,
                "manifest_sha256": row["sha256"],
                "status": status,
                "retained": retained,
                "reason": reason,
                "label_decision": label_decision.get("decision", "") if label_decision else "",
                "label_note": label_decision.get("note", "") if label_decision else "",
                "label_review_status": _label_review_status(label_decision, path in label_paths),
                "pair_ids": "|".join(pair_ids),
                "pair_decision": "|".join(pair_values),
                "pair_review_status": pair_status,
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "group_id": group_id,
                "group_size": 0,
                "new_split": "",
                "new_path": "",
                "title": source.get("title", "") or "",
                "source_page": source.get("source_page", "") or "",
                "creator": creator,
                "credit": credit,
                "license": source.get("api_license", "") or "",
                "license_url": source.get("api_license_url", "") or "",
                "license_family": license_family,
                "missing_artist": not bool(creator.strip()),
                "missing_credit": not bool(credit.strip()),
                "missing_license_url": missing_license_url,
                "missing_license_url_attribution": missing_license_url_attribution,
                "nonstandard_license": license_family in NONSTANDARD_LICENSE_FAMILIES,
                "source_followup_issue": source_followup.get(path, ""),
                "review_conflict": review_conflict,
            }
        )

    from scripts.experiment2_integrity import group_candidates

    edges = group_candidates(plan, root / "data" / "office_dataset", phase1_dir, payload, pair_decisions)
    retained_rows = [row for row in plan if row["retained"]]
    group_sizes = Counter(row["group_id"] for row in retained_rows)
    for row in plan:
        row["group_size"] = group_sizes.get(row["group_id"], 0)

    _assign_splits(retained_rows, seed)

    status_counts = Counter(row["status"] for row in plan)
    class_counts_retained = Counter(row["class"] for row in retained_rows)
    class_counts_train = Counter(row["class"] for row in retained_rows if row["new_split"] == "train")
    class_counts_val = Counter(row["class"] for row in retained_rows if row["new_split"] == "val")

    class_table: dict[str, dict[str, int]] = {}
    for label in CLASSES:
        class_table[label] = {
            "retained": class_counts_retained.get(label, 0),
            "train": class_counts_train.get(label, 0),
            "val": class_counts_val.get(label, 0),
            "excluded_incorrect": sum(
                1 for row in plan if row["class"] == label and row["status"] == "excluded_incorrect"
            ),
            "held_out_uncertain": sum(
                1 for row in plan if row["class"] == label and row["status"] == "held_out_uncertain"
            ),
            "held_out_unreviewed_label": sum(
                1 for row in plan
                if row["class"] == label and row["status"] == "held_out_unreviewed_label"
            ),
        }

    cluster_rows = []
    for cluster_id, cluster in clusters.items():
        members = [row for row in plan if row["cluster_id"] == cluster_id]
        retained_members = [row for row in members if row["retained"]]
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "n_images": int(cluster["n_images"]),
                "classes": cluster["classes"],
                "source_splits": cluster["splits"],
                "cross_split": _as_bool(cluster["cross_split"]),
                "retained": len(retained_members),
                "train": sum(1 for row in retained_members if row["new_split"] == "train"),
                "val": sum(1 for row in retained_members if row["new_split"] == "val"),
                "held_out": sum(1 for row in members if row["status"] in HELD_OUT_STATUSES),
                "excluded": sum(1 for row in members if row["status"] in EXCLUDED_STATUSES),
                "new_splits": "|".join(sorted({row["new_split"] for row in retained_members})),
            }
        )

    retained_hashes = [row["sha256"] for row in retained_rows]
    duplicate_hashes = sorted(
        digest for digest, count in Counter(retained_hashes).items() if count > 1
    )
    manifest_hash_mismatches = [
        row["source_path"] for row in plan if row["sha256"] != row["manifest_sha256"]
    ]
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for row in retained_rows:
        split_by_group[row["group_id"]].add(row["new_split"])
    groups_crossing = sorted(group for group, splits in split_by_group.items() if len(splits) > 1)

    retained_licensing = [row for row in retained_rows if row["missing_artist"] or row["missing_credit"] or row["missing_license_url_attribution"] or row["nonstandard_license"]]
    retained_followup = [row for row in retained_rows if row["source_followup_issue"]]
    review_conflicts = [row for row in plan if row["review_conflict"]]

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": len(plan),
        "status_counts": dict(status_counts),
        "quarantined": sum(status_counts[k] for k in HELD_OUT_STATUSES),
        "excluded": sum(status_counts[k] for k in EXCLUDED_STATUSES),
        "duplicate_edges": edges,
        "suspected_groups_retained": sum(n > 1 for n in group_sizes.values()),
        "manually_confirmed_groups": len({
            row["group_id"] for row in plan
            if set(row["pair_decision"].split("|")) & {"Duplicate", "Near-duplicate"}
        }),
        "retained": len(retained_rows),
        "excluded_incorrect": status_counts.get("excluded_incorrect", 0),
        "held_out_uncertain": status_counts.get("held_out_uncertain", 0),
        "held_out_unreviewed_label": status_counts.get("held_out_unreviewed_label", 0),
        "retained_reviewed_correct": status_counts.get("retained_reviewed_correct", 0),
        "retained_unflagged": status_counts.get("retained_unflagged", 0),
        "train": class_counts_train.total(),
        "val": class_counts_val.total(),
        "class_counts_retained": dict(class_counts_retained),
        "class_counts_train": dict(class_counts_train),
        "class_counts_val": dict(class_counts_val),
        "class_table": class_table,
        "review": {
            "label_decisions": audit["label_decisions"],
            "pair_decisions": audit["pair_decisions"],
            "label_counts": audit["label_counts"],
            "pair_counts": audit["pair_counts"],
            "high_total": audit["high_total"],
            "high_decided": audit["high_decided"],
            "high_missing": len(audit["high_missing"]),
            "label_missing": len(audit["label_missing"]),
            "pair_missing": len(audit["pair_missing"]),
            "source_followup": audit["source_followup"],
        },
        "clusters": {
            "n_clusters": len(cluster_rows),
            "n_images_in_clusters": sum(row["n_images"] for row in cluster_rows),
            "n_cross_split_clusters": sum(1 for row in cluster_rows if row["cross_split"]),
            "n_images_in_cross_split_clusters": sum(
                row["n_images"] for row in cluster_rows if row["cross_split"]
            ),
            "n_clusters_with_retained_members": sum(1 for row in cluster_rows if row["retained"]),
            "n_groups_crossing_new_splits": len(groups_crossing),
        },
        "duplicate_hashes_retained": duplicate_hashes,
        "manifest_hash_mismatches": len(manifest_hash_mismatches),
        "manifest_hash_mismatch_paths": manifest_hash_mismatches,
        "groups_crossing": groups_crossing,
        "all_classes_in_train_and_val": all(
            class_counts_train.get(label, 0) > 0 and class_counts_val.get(label, 0) > 0
            for label in CLASSES
        ),
        "source_followup": {
            "total": audit["source_followup"],
            "retained": len(retained_followup),
            "held_out_or_excluded": audit["source_followup"] - len(retained_followup),
        },
        "licensing_retained": {
            "missing_artist": sum(1 for row in retained_rows if row["missing_artist"]),
            "missing_credit": sum(1 for row in retained_rows if row["missing_credit"]),
            "missing_license_url": sum(1 for row in retained_rows if row["missing_license_url"]),
            "missing_license_url_attribution": sum(
                1 for row in retained_rows if row["missing_license_url_attribution"]
            ),
            "nonstandard_license": sum(1 for row in retained_rows if row["nonstandard_license"]),
            "rows_with_any_issue": len(retained_licensing),
        },
        "review_conflicts": len(review_conflicts),
        "review_conflict_rows": [
            {
                "path": row["source_path"],
                "class": row["class"],
                "title": row["title"],
                "reason": row["review_conflict"],
            }
            for row in review_conflicts
        ],
        "cluster_rows": cluster_rows,
        "source_followup_rows": [
            {
                "path": row["source_path"],
                "class": row["class"],
                "status": row["status"],
                "issue": row["source_followup_issue"],
            }
            for row in plan
            if row["source_followup_issue"]
        ],
        "licensing_issue_rows": [
            {
                "path": row["source_path"],
                "class": row["class"],
                "status": row["status"],
                "creator": row["creator"],
                "credit": row["credit"],
                "license": row["license"],
                "license_url": row["license_url"],
                "license_family": row["license_family"],
                "missing_artist": row["missing_artist"],
                "missing_credit": row["missing_credit"],
                "missing_license_url_attribution": row["missing_license_url_attribution"],
                "nonstandard_license": row["nonstandard_license"],
                "source_followup_issue": row["source_followup_issue"],
            }
            for row in retained_licensing
        ],
    }
    return plan, summary


def materialize_plan(plan: list[dict[str, Any]], source_root: Path, output_root: Path) -> int:
    """Copy retained images into the new dataset and write a full manifest."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists() and any(
        path for path in output_root.iterdir() if path.name != ".gitkeep"
    ):
        raise FileExistsError(
            f"Output {output_root} is not empty. Move it before rebuilding Experiment 2."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    normalised = [_normalise_row(row) for row in plan]
    copied = 0
    for row in normalised:
        if not row["retained"]:
            continue
        source = source_root / _manifest_path(row["source_path"])
        destination = output_root / _manifest_path(row["new_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        load_image(source)  # Verify the source image decodes before copying.
        shutil.copy2(source, destination)
        load_image(destination)  # Verify the copy decodes after copying.
        expected = str(row.get("sha256", "")).strip()
        if len(expected) == 64:
            actual = sha256(destination.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Copied image hash mismatch: {destination}")
        copied += 1

    _write_csv(output_root / "manifest.csv", normalised, PLAN_FIELDS)
    return copied


def write_reports(
    plan: list[dict[str, Any]],
    summary: dict[str, Any],
    output_root: Path,
    results_dir: Path,
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    class_rows = []
    for label in CLASSES:
        counts = summary["class_table"][label]
        class_rows.append(
            {
                "class": label,
                "retained": counts["retained"],
                "train": counts["train"],
                "val": counts["val"],
                "excluded_incorrect": counts["excluded_incorrect"],
                "held_out_uncertain": counts["held_out_uncertain"],
                "held_out_unreviewed_label": counts["held_out_unreviewed_label"],
            }
        )
    _write_csv(
        results_dir / "experiment2_class_counts.csv",
        class_rows,
        [
            "class",
            "retained",
            "train",
            "val",
            "excluded_incorrect",
            "held_out_uncertain",
            "held_out_unreviewed_label",
        ],
    )

    _write_csv(
        results_dir / "experiment2_excluded_images.csv",
        [row for row in plan if row["status"] in EXCLUDED_STATUSES],
        PLAN_FIELDS,
    )
    _write_csv(
        results_dir / "experiment2_held_out_images.csv",
        [row for row in plan if row["status"] in HELD_OUT_STATUSES],
        PLAN_FIELDS,
    )
    _write_csv(
        results_dir / "experiment2_duplicate_groups.csv",
        summary["cluster_rows"],
        [
            "cluster_id",
            "n_images",
            "classes",
            "source_splits",
            "cross_split",
            "retained",
            "train",
            "val",
            "held_out",
            "excluded",
            "new_splits",
        ],
    )
    _write_csv(
        results_dir / "experiment2_source_licensing.csv",
        summary["licensing_issue_rows"],
        [
            "path",
            "class",
            "status",
            "creator",
            "credit",
            "license",
            "license_url",
            "license_family",
            "missing_artist",
            "missing_credit",
            "missing_license_url_attribution",
            "nonstandard_license",
            "source_followup_issue",
        ],
    )
    _write_csv(
        results_dir / "experiment2_review_conflicts.csv",
        summary["review_conflict_rows"],
        ["path", "class", "title", "reason"],
    )
    _write_csv(
        results_dir / "experiment2_source_followup.csv",
        summary["source_followup_rows"],
        ["path", "class", "status", "issue"],
    )

    _write_csv(results_dir / "experiment2_duplicate_edges.csv", summary["duplicate_edges"],
               ["first", "second", "evidence"])
    from scripts.experiment2_integrity import check_dataset
    evidence = check_dataset(output_root, summary["duplicate_edges"])
    (results_dir / "integrity.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    write_dataset_readme(summary, output_root / "README.md")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_dataset_readme(summary: dict[str, Any], path: Path) -> None:
    class_rows = [
        [label, summary["class_table"][label]["train"], summary["class_table"][label]["val"]]
        for label in CLASSES
    ]
    text = f"""# OfficeMate AI — Experiment 2 working dataset

This folder is a separate working dataset built from saved Phase 2 review decisions.
The original dataset remains at `data/office_dataset` and was not modified.

- Train: {summary['train']} images
- Validation: {summary['val']} images
- Final test: not created here. A genuinely new independent test set must be collected separately.
- Retained images: {summary['retained']}
- Excluded as Incorrect: {summary['excluded_incorrect']}
- Held out as Uncertain: {summary['held_out_uncertain']}
- Held out because a label flag is unreviewed: {summary['held_out_unreviewed_label']}

{_markdown_table(['Class', 'Train', 'Val'], class_rows)}

The complete row-level record is in `manifest.csv`. Excluded and held-out images
remain in the original dataset and are listed under `results/experiment2/dataset/`.
Machine-detected near-duplicate clusters are kept in one split; see
`experiment2_duplicate_groups.csv`.

This split is for development. The former test photos remain outside it.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dataset. Default: <root>/data/experiment2")
    parser.add_argument("--results", type=Path, default=None,
                        help="Output report folder. Default: <root>/results/experiment2/dataset")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and print the summary without copying images.")
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out.resolve() if args.out else root / "data" / "experiment2"
    results = args.results.resolve() if args.results else root / "results" / "experiment2" / "dataset"

    plan, summary = build_plan(root, seed=args.seed)
    print(json.dumps({
        "total": summary["total"],
        "retained": summary["retained"],
        "excluded_incorrect": summary["excluded_incorrect"],
        "held_out_uncertain": summary["held_out_uncertain"],
        "held_out_unreviewed_label": summary["held_out_unreviewed_label"],
        "train": summary["train"],
        "val": summary["val"],
        "review_conflicts": summary["review_conflicts"],
        "clusters_crossing_new_splits": summary["clusters"]["n_groups_crossing_new_splits"],
    }, indent=2))

    if args.dry_run:
        print("Dry run only: no images copied, no reports written.")
        return

    if results.exists() and any(results.iterdir()):
        raise FileExistsError(f"Results directory is not empty: {results}")
    copied = materialize_plan(plan, root / "data" / "office_dataset", out)
    write_reports(plan, summary, out, results)
    print(f"Copied {copied} retained images to {out}")
    print(f"Wrote reports to {results}")


if __name__ == "__main__":
    main()
