"""Additional duplicate screening and on-disk verification for Experiment 2."""
import csv
from collections import defaultdict
from hashlib import sha256

from PIL import Image
from skimage.metrics import structural_similarity

from officemate.config import CLASSES, IMAGE_EXTENSIONS
from .phase1_audit_dataset import dhash64, load_gray, phash64, pair_verdict


def group_candidates(plan, source_root, phase1_dir, payload, decisions):
    rows = {row["source_path"]: row for row in plan}
    parent = {path: path for path in rows}
    edges = []

    def find(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def link(first, second, evidence):
        first = first.replace("/", "\\").removeprefix("data\\office_dataset\\")
        second = second.replace("/", "\\").removeprefix("data\\office_dataset\\")
        parent[find(second)] = find(first)
        edges.append({"first": first, "second": second, "evidence": evidence})

    indexed = {}
    for path, row in rows.items():
        for key in ("cluster_id", "sha256"):
            value = row[key]
            if value:
                first = indexed.setdefault((key, value), path)
                if first != path:
                    link(first, path, key)

    for pair in payload["pairs"]:
        decision = decisions.get(pair["pair_id"], {}).get("decision")
        if decision != "Distinct":
            link(pair["first_path"], pair["second_path"], decision or "Phase 1 unresolved pair")

    audit = phase1_dir / "duplicate_pairs.csv"
    if audit.is_file():
        with audit.open(encoding="utf-8", newline="") as stream:
            for pair in csv.DictReader(stream):
                verdict = pair_verdict(
                    pair["same_commons_page"].lower() == "true", int(pair["dhash_distance"]),
                    int(pair["phash_distance"]), float(pair["ssim"]), int(pair["orb_inliers"]),
                )
                if verdict.startswith(("CONFIRMED", "LIKELY")):
                    link(pair["first_path"], pair["second_path"], f"Phase 1 machine evidence: {verdict}")

    fingerprints = []
    for path, row in rows.items():
        if not row["retained"]:
            continue
        image_path = source_root / path.replace("\\", "/")
        try:
            with Image.open(image_path) as image:
                image.verify()
            gray = load_gray(image_path, size=128)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            row.update(retained=False, status="held_out_invalid", reason=f"Image decode failed: {exc}")
            continue
        flipped = gray[:, ::-1].copy()
        fingerprints.append((path, gray, phash64(gray), dhash64(gray),
                             flipped, phash64(flipped), dhash64(flipped)))

    for i, (first, gray, p, d, _, _, _) in enumerate(fingerprints):
        for second, other, op, od, flipped, fp, fd in fingerprints[i + 1:]:
            for candidate, ph, dh, orientation in ((other, op, od, "direct"), (flipped, fp, fd, "mirror")):
                pd, dd = (p ^ ph).bit_count(), (d ^ dh).bit_count()
                if pd <= 6 or dd <= 2 or (pd <= 12 and dd <= 12):
                    score = structural_similarity(gray, candidate, data_range=255)
                    if (pd <= 6 and dd <= 6) or score >= 0.90:
                        link(first, second, f"{orientation}: pHash={pd} dHash={dd} SSIM={score:.4f}")
                        break

    classes = defaultdict(set)
    for path, row in rows.items():
        classes[find(path)].add(row["class"])
    seen_hashes = set()
    for path, row in rows.items():
        group = find(path)
        row["group_id"] = group
        if not row["retained"]:
            continue
        if len(classes[group]) > 1:
            row.update(retained=False, status="held_out_cross_class",
                       reason="Duplicate evidence connects conflicting classes. Manual review required.")
        elif row["sha256"] in seen_hashes:
            row.update(retained=False, status="excluded_exact_duplicate", reason="Exact retained copy exists.")
        else:
            seen_hashes.add(row["sha256"])
    return edges


def check_dataset(dataset, edges):
    """Check actual files, labels, hashes and recorded evidence after copying."""
    with (dataset / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        retained = [row for row in csv.DictReader(stream) if row["retained"] == "True"]
    expected = set()
    splits = defaultdict(set)
    labels = defaultdict(set)
    hashes = set()
    by_source = {}
    for row in retained:
        split, label = row["new_split"], row["class"]
        path = dataset / row["new_path"].replace("\\", "/")
        digest = sha256(path.read_bytes()).hexdigest()
        if path.relative_to(dataset).parts[:2] != (split, label) or digest != row["sha256"] or digest in hashes:
            raise ValueError(f"Unexpected path or checksum: {path}")
        hashes.add(digest)
        expected.add(path)
        by_source[row["source_path"]] = split
        splits[row["group_id"]].add(split)
        labels[split].add(label)
    actual = {p for p in dataset.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    if actual != expected or labels != {"train": set(CLASSES), "val": set(CLASSES)}:
        raise ValueError("Dataset files or class coverage differ from the manifest.")
    for edge in edges:
        first, second = by_source.get(edge["first"]), by_source.get(edge["second"])
        if first and second and first != second:
            raise ValueError(f"Duplicate edge crosses splits: {edge}")
    if any(len(values) != 1 for values in splits.values()):
        raise ValueError("Duplicate group crosses splits.")
    return {"images_verified": len(expected), "exact_duplicates": 0, "cross_split_groups": 0,
            "cross_split_edges": 0, "evidence_edges_checked": len(edges)}
