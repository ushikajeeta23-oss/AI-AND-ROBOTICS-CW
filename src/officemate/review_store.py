"""Small, explicit decision store for the Phase 2 human-review app.

The review app is the only writer. A row is written only when a person
chooses a decision and presses Save; no image is ever marked reviewed by
default. Files are replaced atomically so an interrupted save cannot leave
a half-written CSV.
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path


LABEL_DECISION_FIELDS = [
    "item_id",
    "path",
    "class",
    "split",
    "priority",
    "queues",
    "title",
    "reason",
    "decision",
    "note",
    "reviewer",
    "reviewed_at",
]

PAIR_DECISION_FIELDS = [
    "pair_id",
    "class",
    "first_path",
    "second_path",
    "first_split",
    "second_split",
    "machine_verdict",
    "reason",
    "decision",
    "note",
    "reviewer",
    "reviewed_at",
]


def load_decisions(path: Path, key: str = "item_id") -> dict[str, dict]:
    """Return decisions keyed by ``key``; an absent file means no decisions."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[key]: row for row in csv.DictReader(handle) if row.get(key)}


def upsert_decision(path: Path, row: dict, fieldnames: list[str], key: str) -> None:
    """Insert or replace one decision and preserve every other row.

    The write goes to a temporary file in the same directory and is then
    atomically replaced, so a crash cannot corrupt the decision log.
    """
    path = Path(path)
    if not row.get(key):
        raise ValueError(f"decision row is missing {key!r}")
    clean = {field: str(row.get(field, "")) for field in fieldnames}
    decisions = load_decisions(path, key)
    decisions[clean[key]] = clean

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for decision_key in sorted(decisions):
            writer.writerow({field: decisions[decision_key].get(field, "") for field in fieldnames})
    os.replace(tmp, path)


def decision_counts(decisions: dict[str, dict]) -> Counter:
    return Counter(row.get("decision", "") for row in decisions.values() if row.get("decision"))
