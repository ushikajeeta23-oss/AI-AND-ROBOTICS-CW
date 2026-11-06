"""Build the Phase 2 human-review queue from the Phase 1 audit.

This script is read-only with respect to the dataset, model and Experiment 1.
It writes only the review payload and its instructions under
``results/phase2/review``. No image is marked reviewed here.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from officemate.config import CLASSES


ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / "results" / "phase1"
MANIFEST = ROOT / "data" / "office_dataset" / "manifest.csv"
OUT = ROOT / "results" / "phase2" / "review"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_dataset_path(path: str) -> str:
    """Use one path form even though Phase 1 CSVs mix prefixed and bare paths."""
    value = str(path or "").replace("/", "\\")
    prefix = "data\\office_dataset\\"
    if value.lower().startswith(prefix):
        value = value[len(prefix):]
    return value


def by_path(rows: list[dict]) -> dict[str, dict]:
    return {normalize_dataset_path(row.get("path", "")): row for row in rows if row.get("path")}


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def short(text: str, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _cluster_map(cluster_rows: list[dict]) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for cluster in cluster_rows:
        for path in (cluster.get("paths") or "").split("|"):
            if path:
                mapping[normalize_dataset_path(path)] = cluster
    return mapping


def _pair_map(pair_rows: list[dict]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for row in pair_rows:
        pair_id = f"pair:{as_int(row.get('pair')):02d}"
        for key in ("first_path", "second_path"):
            path = normalize_dataset_path(row.get(key, ""))
            if path:
                mapping[path].append(pair_id)
    return mapping


def _reason(meta: dict, model: dict, work: dict, cluster: dict | None,
            pair_ids: list[str]) -> str:
    parts: list[str] = []
    severity = (meta.get("metadata_severity") or "").strip()
    flags = (meta.get("metadata_flags") or "").strip()
    if flags:
        label = f"Label check ({severity or 'review'})" if severity in {"HIGH", "MEDIUM"} else "Label note"
        parts.append(f"{label}: {flags}")
    if as_bool(model.get("strong_disagreement")):
        parts.append(
            f"Model strongly predicted {model.get('predicted', '?')} "
            f"({as_float(model.get('confidence'), 0):.2f}) while the current label is "
            f"{model.get('class', '?')}. Treat this as a prompt to look, not proof."
        )
    elif as_bool(model.get("uncertain")):
        parts.append(
            f"Model was uncertain: {model.get('predicted', '?')} "
            f"({as_float(model.get('confidence'), 0):.2f})."
        )
    if cluster and as_bool(cluster.get("cross_split")):
        parts.append(
            f"Cross-split near-duplicate cluster {cluster.get('cluster_id')} "
            f"({cluster.get('n_images')} images; splits {cluster.get('splits')})."
        )
    if pair_ids:
        parts.append("Flagged in " + ", ".join(pair_ids) + ".")
    if not parts and (work.get("flags") or "").strip():
        parts.append(work["flags"])
    if not parts and (meta.get("metadata_note") or "").strip():
        parts.append(meta["metadata_note"])
    return " | ".join(parts) if parts else "Review prompt from the Phase 1 audit."


def _queues(meta: dict, model: dict, work: dict, cluster: dict | None,
            pair_ids: list[str]) -> list[str]:
    queues: list[str] = []
    severity = (meta.get("metadata_severity") or "").strip()
    if severity == "HIGH":
        queues.append("high_label")
    elif severity == "MEDIUM":
        queues.append("medium_label")
    if as_bool(model.get("strong_disagreement")):
        queues.append("model_strong")
    if as_bool(model.get("uncertain")):
        queues.append("model_uncertain")
    if (cluster and as_bool(cluster.get("cross_split"))) or pair_ids:
        queues.append("leakage")
    if not queues and work:
        queues.append("other")
    return queues


def build_source_followup(license_rows: list[dict], project_root: Path) -> list[dict]:
    """The 11 images with no artist/credit and the 3 non-PD/CC0 missing URLs."""
    items: list[dict] = []
    for row in license_rows:
        path = normalize_dataset_path(row.get("path", ""))
        if not path:
            continue
        issues: list[str] = []
        if not (row.get("api_artist") or row.get("csv_creator") or row.get("api_credit")):
            issues.append("missing artist and credit")
        family = (row.get("license_family") or "").lower()
        if (as_bool(row.get("missing_license_url"))
                and "public domain" not in family and "cc0" not in family):
            issues.append("attribution licence has no URL")
        if issues:
            items.append({
                "path": path,
                "image_exists": (project_root / "data" / "office_dataset" / path).exists(),
                "class": row.get("class", ""),
                "split": row.get("split", ""),
                "title": row.get("title", ""),
                "issue": "; ".join(issues),
                "creator": row.get("api_artist") or row.get("csv_creator") or row.get("api_credit", ""),
                "credit": row.get("api_credit", ""),
                "license": row.get("api_license") or row.get("csv_license", ""),
                "license_url": row.get("api_license_url") or row.get("csv_license_url", ""),
                "license_family": row.get("license_family", ""),
                "source_page": row.get("source_page", ""),
                "verification_status": row.get("verification_status", ""),
            })
    order = {"missing artist and credit": 0, "attribution licence has no URL": 1}
    items.sort(key=lambda item: (
        order.get(item["issue"].split("; ")[0], 9), item["class"], item["path"]
    ))
    return items



def build_review_payload(phase1_dir: Path = PHASE1, manifest_path: Path = MANIFEST,
                         project_root: Path = ROOT) -> dict:
    phase1_dir = Path(phase1_dir)
    project_root = Path(project_root)
    manifest_rows = read_csv(Path(manifest_path))
    manifest = by_path(manifest_rows)

    work_rows = read_csv(phase1_dir / "manual_review_priority.csv")
    meta_rows = read_csv(phase1_dir / "label_audit_metadata.csv")
    model_rows = read_csv(phase1_dir / "label_audit_model.csv")
    license_rows = read_csv(phase1_dir / "licensing_audit.csv")
    pair_rows = read_csv(phase1_dir / "duplicate_pairs.csv")
    cluster_rows = read_csv(phase1_dir / "duplicate_clusters.csv")

    work = by_path(work_rows)
    meta = by_path(meta_rows)
    model = by_path(model_rows)
    lic = by_path(license_rows)
    clusters = _cluster_map(cluster_rows)
    pairs_by_path = _pair_map(pair_rows)

    meta_candidates = {
        path for path, row in meta.items()
        if as_bool(row.get("metadata_suspicious"))
        or (row.get("metadata_severity") or "").strip() in {"HIGH", "MEDIUM"}
    }
    model_candidates = {
        path for path, row in model.items()
        if as_bool(row.get("strong_disagreement")) or as_bool(row.get("uncertain"))
    }
    # Keep the queue focused: the worklist, plus any HIGH/MEDIUM metadata row
    # or strong/uncertain model row that the worklist omitted.
    candidate_paths = (set(work) | meta_candidates | model_candidates) & set(manifest)

    labels: list[dict] = []
    for path in candidate_paths:
        rec = manifest[path]
        m = meta.get(path, {})
        mr = model.get(path, {})
        w = work.get(path, {})
        l = lic.get(path, {})
        cluster = clusters.get(path)
        pair_ids = pairs_by_path.get(path, [])
        queues = _queues(m, mr, w, cluster, pair_ids)
        severity = (m.get("metadata_severity") or "").strip()
        priority = "HIGH" if ("high_label" in queues or "model_strong" in queues) else (
            "MEDIUM" if ("medium_label" in queues or "model_uncertain" in queues) else "LOW"
        )
        image_path = project_root / "data" / "office_dataset" / path
        labels.append({
            "item_id": f"label:{path}",
            "path": path,
            "image_exists": image_path.exists(),
            "class": rec.get("class", w.get("class", "")),
            "split": rec.get("split", w.get("split", "")),
            "page_id": rec.get("group", w.get("page_id", "")),
            "title": l.get("title") or m.get("title") or w.get("title", ""),
            "description": short(m.get("description", "")),
            "priority": priority,
            "queues": queues,
            "metadata_severity": severity,
            "metadata_flags": m.get("metadata_flags", ""),
            "metadata_note": m.get("metadata_note", ""),
            "worklist_priority": w.get("priority", ""),
            "issue_types": w.get("issue_types", ""),
            "recommended_action": w.get("recommended_action", ""),
            "model_predicted": mr.get("predicted", ""),
            "model_confidence": as_float(mr.get("confidence")),
            "model_correct": as_bool(mr.get("correct")) if mr else None,
            "model_strong": as_bool(mr.get("strong_disagreement")),
            "model_uncertain": as_bool(mr.get("uncertain")),
            "cluster_id": cluster.get("cluster_id") if cluster else "",
            "cluster_size": as_int(cluster.get("n_images")) if cluster else "",
            "cluster_splits": cluster.get("splits", "") if cluster else "",
            "cluster_cross_split": as_bool(cluster.get("cross_split")) if cluster else False,
            "pair_ids": pair_ids,
            "reason": _reason(m, mr, w, cluster, pair_ids),
            "source_page": l.get("source_page", ""),
            "creator": l.get("api_artist") or l.get("csv_creator") or l.get("api_credit", ""),
            "credit": l.get("api_credit", ""),
            "license": l.get("api_license") or l.get("csv_license", ""),
            "license_url": l.get("api_license_url") or l.get("csv_license_url", ""),
            "license_family": l.get("license_family", ""),
            "verification_status": l.get("verification_status", ""),
        })

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    labels.sort(key=lambda item: (
        order.get(item["priority"], 9),
        0 if "high_label" in item["queues"] else 1,
        0 if "model_strong" in item["queues"] else 1,
        item["class"],
        item["split"],
        item["path"],
    ))

    pairs: list[dict] = []
    for row in pair_rows:
        first_path = normalize_dataset_path(row.get("first_path", ""))
        second_path = normalize_dataset_path(row.get("second_path", ""))
        first_lic = lic.get(first_path, {})
        second_lic = lic.get(second_path, {})
        first_cluster = clusters.get(first_path)
        second_cluster = clusters.get(second_path)
        pair_id = f"pair:{as_int(row.get('pair')):02d}"
        reason = (
            f"Machine verdict: {row.get('verdict', '')}. "
            f"dHash {row.get('dhash_distance', '?')}, pHash {row.get('phash_distance', '?')}, "
            f"SSIM {row.get('ssim', '?')}, ORB inliers {row.get('orb_inliers', '?')}. "
            "This is evidence for you to check, not a final decision."
        )
        if first_cluster and as_bool(first_cluster.get("cross_split")):
            reason += (
                f" Cluster {first_cluster.get('cluster_id')} spans "
                f"{first_cluster.get('splits')}."
            )
        pairs.append({
            "pair_id": pair_id,
            "class": row.get("class", ""),
            "first_path": first_path,
            "second_path": second_path,
            "first_exists": (project_root / "data" / "office_dataset" / first_path).exists(),
            "second_exists": (project_root / "data" / "office_dataset" / second_path).exists(),
            "first_split": row.get("first_split", ""),
            "second_split": row.get("second_split", ""),
            "dhash_distance": as_int(row.get("dhash_distance")),
            "phash_distance": as_int(row.get("phash_distance")),
            "ssim": as_float(row.get("ssim")),
            "orb_inliers": as_int(row.get("orb_inliers")),
            "same_commons_page": as_bool(row.get("same_commons_page")),
            "machine_verdict": row.get("verdict", ""),
            "first_title": first_lic.get("title") or row.get("first_title", ""),
            "second_title": second_lic.get("title") or row.get("second_title", ""),
            "first_source_page": first_lic.get("source_page", ""),
            "second_source_page": second_lic.get("source_page", ""),
            "first_creator": first_lic.get("api_artist") or row.get("first_creator", ""),
            "second_creator": second_lic.get("api_artist") or row.get("second_creator", ""),
            "first_license": first_lic.get("api_license") or row.get("first_license", ""),
            "second_license": second_lic.get("api_license") or row.get("second_license", ""),
            "first_cluster": first_cluster.get("cluster_id") if first_cluster else "",
            "second_cluster": second_cluster.get("cluster_id") if second_cluster else "",
            "reason": reason,
        })

    queue_counts = Counter()
    for item in labels:
        queue_counts.update(item["queues"])

    source_followup = build_source_followup(license_rows, project_root)

    return {
        "generated_from": {
            "phase1_dir": str(phase1_dir),
            "manifest": str(manifest_path),
            "dataset_images": len(manifest_rows),
            "phase1_pairs": len(pair_rows),
            "phase1_worklist": len(work_rows),
        },
        "summary": {
            "label_items": len(labels),
            "high_label_items": queue_counts.get("high_label", 0),
            "medium_label_items": queue_counts.get("medium_label", 0),
            "model_strong_items": queue_counts.get("model_strong", 0),
            "model_uncertain_items": queue_counts.get("model_uncertain", 0),
            "leakage_items": queue_counts.get("leakage", 0),
            "pair_items": len(pairs),
            "source_followup_items": len(source_followup),
            "missing_artist_credit_items": sum(
                1 for item in source_followup if "missing artist and credit" in item["issue"]
            ),
            "attribution_url_missing_items": sum(
                1 for item in source_followup if "attribution licence has no URL" in item["issue"]
            ),
            "classes": {cls: sum(1 for item in labels if item["class"] == cls) for cls in CLASSES},
        },
        "labels": labels,
        "pairs": pairs,
        "source_followup": source_followup,
    }


def write_review_payload(payload: dict, out_dir: Path = OUT) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review_items.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (out_dir / "review_items.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "item_id", "path", "class", "split", "priority", "queues", "title",
            "metadata_severity", "metadata_flags", "model_predicted",
            "model_confidence", "cluster_id", "cluster_splits", "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in payload["labels"]:
            writer.writerow({**item, "queues": "|".join(item["queues"])})

    with (out_dir / "source_followup.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "path", "class", "split", "title", "issue", "creator", "credit",
            "license", "license_url", "source_page", "verification_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload.get("source_followup", []))


def write_instructions(out_dir: Path = OUT) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README_REVIEW.md").write_text(
        "# Phase 2 visual review\n\n"
        "Start the review app from the project folder:\n\n"
        "```bat\nRUN_PHASE2_REVIEW.bat\n```\n\n"
        "Or run:\n\n"
        "```bat\n.venv\\Scripts\\python.exe -m streamlit run scripts\\phase2_review_app.py --server.port 8502\n```\n\n"
        "## What to review\n\n"
        "- **Label review:** the 49 HIGH label problems first. Mark each image "
        "Correct, Incorrect or Uncertain. Add a short note for Incorrect/Uncertain.\n"
        "- **Duplicate pairs:** all 36 pairs. Mark each Duplicate, Near-duplicate, "
        "Distinct or Uncertain.\n"
        "- The queue also contains MEDIUM/LOW items and model disagreements for context.\n"
        "- **Source follow-up:** `source_followup.csv` lists 13 unique images: "
        "11 with no artist/credit and 3 non-PD/CC0 images with no licence URL "
        "(one image has both issues). These need verification; do not invent "
        "missing attribution.\n\n"
        "## Where decisions go\n\n"
        "- `results/phase2/review/decisions/label_decisions.csv`\n"
        "- `results/phase2/review/decisions/duplicate_decisions.csv`\n\n"
        "A decision is written only when you press **Save decision**. The app never "
        "marks an image reviewed by default. You can stop and resume; saved rows are "
        "loaded again on the next start.\n\n"
        "No original photograph, dataset split, model or Experiment 1 result is changed "
        "by the review app.\n",
        encoding="utf-8",
    )


def main() -> None:
    payload = build_review_payload()
    write_review_payload(payload)
    write_instructions()
    print("Review payload written to", OUT)
    print(json.dumps(payload["summary"], indent=2))
    print("No image, dataset, model or Experiment 1 file was modified.")


if __name__ == "__main__":
    main()
