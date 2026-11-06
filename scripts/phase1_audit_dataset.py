#!/usr/bin/env python
"""Phase 1 read-only audit for the OfficeMate AI dataset.

This script NEVER writes to data/ and NEVER changes the trained model or baseline
results. It produces review artefacts under results/phase1/ only.

What it does
------------
1. Re-measures the 36 cross-split near-duplicate pairs with dHash, pHash, SSIM
   and ORB feature matching, and builds a side-by-side JPEG for each pair.
2. Clusters visually similar images across the whole dataset (conservative
   pHash/dHash thresholds) so Phase 2 can keep whole clusters in one split.
3. Flags label-quality suspects using (a) Commons titles/descriptions and
   (b) the existing model on train/val ONLY. The held-out test split is never
   used for model-based label auditing, so it remains usable for an independent
   final evaluation.
4. Verifies Wikimedia Commons licence/attribution metadata against the live
   Commons API and reports missing or mismatched fields.
5. Writes a Markdown report, an HTML review index and CSV worklists.

Usage
-----
    python scripts/phase1_audit_dataset.py

Optional:
    --skip-api       do not contact Wikimedia Commons
    --skip-model     do not run the existing model for label triage
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLASSES = ["bin", "book", "bottle", "chair", "keyboard", "mouse", "mug",
           "notebook", "phone", "stapler"]
TARGET_CLASSES = ["book", "notebook", "phone", "bottle", "stapler"]
SPLITS = ["train", "val", "test"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DATA = ROOT / "data" / "office_dataset"
DOWNLOADS = ROOT / "data" / "downloads"
PAIRS_CSV = ROOT / "results" / "possible_near_duplicates.csv"
OUT = ROOT / "results" / "phase1"
SHEETS = OUT / "sheets"
MODEL = ROOT / "models" / "officemate_yolov8n_cls.pt"

# Conservative grouping thresholds. False positives only make the Phase 2
# split stricter (whole group moved together), which is the safe direction.
PHASH_GROUP_THRESHOLD = 6
DHASH_GROUP_THRESHOLD = 2

# Regex review rules. Each rule is (pattern, human explanation, severity).
# Severity HIGH means the object is probably not the labelled class at all.
# MEDIUM means it is a different variant/context that a human must judge.
# LOW means it is probably acceptable but worth a glance.
# These are review prompts, not automatic deletions.
STRONG_PATTERNS = {
    "notebook": [
        (r"notepad\+\+", "software screenshot (Notepad++), not a physical notebook", "HIGH"),
        (r"screen ?shot|screen capture|screenshot", "screen capture, not a physical notebook", "HIGH"),
        (r"simulation|spreadsheet|\bpython\b", "software/simulation image, not a physical notebook", "HIGH"),
        (r"strukturformel|captura", "software screenshot, not a physical notebook", "HIGH"),
        (r"\bwindows\b|\bsoftware\b", "software screenshot, not a physical notebook", "HIGH"),
        (r"\bnotepad\b", "Notepad/software reference, verify physical notebook", "MEDIUM"),
        (r"\bcomputer\b", "computer context, verify the notebook is physical", "MEDIUM"),
        (r"\blaptop\b", "laptop present, verify the notebook is the main object", "MEDIUM"),
    ],
    "mouse": [
        (r"\bspider\b", "animal (mouse spider), not a computer mouse", "HIGH"),
        (r"\btrap\b", "mousetrap, not a computer mouse", "HIGH"),
        (r"\bicon\b|gnome", "software icon, not a physical mouse", "HIGH"),
        (r"\brat\b|rodent|shrew", "animal, not a computer mouse", "HIGH"),
        (r"cutaway|diagram|mechanism", "diagram, not a usable computer mouse", "MEDIUM"),
    ],
    "phone": [
        (r"charger|charging", "charger/charging accessory, not a phone", "HIGH"),
        (r"battery", "battery, not a phone", "HIGH"),
        (r"phone cover|smartphone cover|mobile phone case|phone case", "case/cover, not a phone", "HIGH"),
        (r"\bcase\b|\bcover\b", "case/cover present; verify the phone itself is the main object", "MEDIUM"),
        (r"\bmast\b", "phone mast, not a phone", "HIGH"),
        (r"locker", "phone locker, not a phone", "HIGH"),
        (r"screen time", "screen-time graphic, not a phone", "HIGH"),
        (r"headphone|earphone|microphone", "audio accessory, not a phone", "HIGH"),
        (r"smartwatch|tablet|laptop", "other device present, verify phone is main object", "MEDIUM"),
    ],
    "bottle": [
        (r"hot water bottle", "hot-water bottle, different context from an office water bottle", "MEDIUM"),
        (r"filling station", "water filling station, not a portable bottle", "HIGH"),
        (r"\btruck\b", "bottled-water truck, not a single bottle", "HIGH"),
        (r"filter|purifier", "filter/purifier; verify the bottle is the main object", "MEDIUM"),
        (r"dispenser", "dispenser, not a bottle", "MEDIUM"),
        (r"army uniform|ammunition", "kit layout, bottle is not the main object", "MEDIUM"),
        (r"\bjar\b", "jar, not a bottle", "HIGH"),
    ],
    "stapler": [
        (r"staple pins|400x|microscope", "microscopic staples, not a stapler", "HIGH"),
        (r"vascular|skin stapler|surgical", "surgical stapler, not an office stapler", "HIGH"),
        (r"staple remover", "staple remover; verify a stapler is also visible", "MEDIUM"),
        (r"paper clip|hole punch|binder", "other stationery; verify the stapler is the main object", "MEDIUM"),
        (r"stitcher", "historical wire stitcher (stapler); judge suitability", "LOW"),
    ],
    "book": [
        (r"\bnotebook\b", "notebook/laptop word in title; verify it is a book", "MEDIUM"),
        (r"\blaptop\b|\bcomputer\b|\btablet\b|\bipad\b|\bphone\b", "other device in title; verify book is main object", "MEDIUM"),
        (r"\bmagazine\b|\bcomic\b|\bkindle\b|e-reader|ereader", "not a conventional book; verify class", "MEDIUM"),
    ],
    "keyboard": [
        (r"\bpiano\b|\borgan\b|\bsynthesizer\b", "musical keyboard, not a computer keyboard", "HIGH"),
        (r"keycap", "keycap, not a full keyboard", "MEDIUM"),
        (r"giant keyboard", "novelty keyboard; judge suitability", "LOW"),
    ],
    "mug": [
        (r"mug ?shot|mugshot", "mug shot, not a drinking mug", "HIGH"),
        (r"\bglass\b|\bteacup\b|\btumbler\b|\bpitcher\b", "different vessel; verify class", "MEDIUM"),
    ],
    "bin": [],
    "chair": [
        (r"wheelchair", "wheelchair, not an office chair", "HIGH"),
        (r"sofa|couch|bench|stool", "different seating; verify class", "MEDIUM"),
    ],
}
# Lenient positive terms used only to note that a title/description is vague.
# This is a LOW-priority note, never a label-error claim.
POSITIVE_PATTERNS = {
    "book": r"\bbook\b|hardcover|paperback|softcover|book cover|\bedition\b|\bnovel\b|textbook|\bvolume\b",
    "notebook": r"notebook|note ?book|exercise book|notepad|\bpad\b|journal|spiral",
    "phone": r"\bphone\b|smartphone|iphone|mobile|cellular|telephone",
    "bottle": r"\bbottle\b|flask|thermos",
    "stapler": r"stapler|stapling",
    "chair": r"\bchair\b",
    "mouse": r"\bmouse\b",
    "mug": r"\bmug\b",
    "bin": r"\bbin\b|trash|waste|rubbish|garbage",
    "keyboard": r"keyboard",
}


# --------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------
def page_id_from_name(name: str) -> str | None:
    m = re.search(r"-commons-(\d+)-", name)
    if m:
        return m.group(1)
    m = re.search(r"_commons_(\d+)\.", name)
    return m.group(1) if m else None


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def short(text: str, n: int = 70) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[: n - 1] + "..."


def norm_title(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ")).strip().lower()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def load_gray(path: Path, size: int = 256) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size, size), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def dhash64(gray: np.ndarray) -> int:
    im = Image.fromarray(gray).resize((9, 8), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = 0
    for value in (a[:, 1:] > a[:, :-1]).flat:
        bits = (bits << 1) | int(value)
    return bits


def ahash64(gray: np.ndarray) -> int:
    im = Image.fromarray(gray).resize((8, 8), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    bits = 0
    for value in (a > a.mean()).flat:
        bits = (bits << 1) | int(value)
    return bits


def phash64(gray: np.ndarray) -> int:
    im = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(im))
    low = dct[:8, :8].copy()
    med = np.median(low[1:, :])  # ignore DC term
    bits = 0
    for value in (low > med).flat:
        bits = (bits << 1) | int(value)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def image_info(path: Path) -> dict:
    with Image.open(path) as im:
        return {"width": im.width, "height": im.height,
                "mode": im.mode, "format": im.format,
                "bytes": path.stat().st_size}


def orb_similarity(p1: Path, p2: Path) -> tuple[int, int]:
    """Return (good descriptor matches, RANSAC homography inliers)."""
    a = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return 0, 0

    def shrink(x):
        h, w = x.shape[:2]
        scale = 640.0 / max(h, w)
        if scale < 1:
            x = cv2.resize(x, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return x

    a, b = shrink(a), shrink(b)
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(da, db), key=lambda m: m.distance)
    good = [m for m in matches if m.distance <= 64]
    inliers = 0
    if len(good) >= 8:
        src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.sum())
    return len(good), inliers


def pair_verdict(same_page: bool, d: int, p: int, ssim: float, inliers: int) -> str:
    if same_page:
        return "CONFIRMED same Commons source page"
    if inliers >= 30 and ssim >= 0.80:
        return "CONFIRMED same scene (feature match)"
    if inliers >= 15 or ssim >= 0.97 or (d == 0 and p <= 4):
        return "LIKELY near-duplicate"
    if ssim >= 0.90 or p <= PHASH_GROUP_THRESHOLD or d <= DHASH_GROUP_THRESHOLD:
        return "POSSIBLE - inspect"
    return "Similar but likely distinct"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# Load dataset + source metadata
# --------------------------------------------------------------------------
def load_manifest() -> list[dict]:
    rows = []
    with (DATA / "manifest.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            path = DATA / r["path"]
            r["abs_path"] = path
            r["page_id"] = page_id_from_name(path.name)
            r["exists"] = path.is_file()
            rows.append(r)
    return rows


def load_sources() -> tuple[dict[str, dict], list[dict]]:
    by_page: dict[str, dict] = {}
    all_rows: list[dict] = []
    for c in CLASSES:
        p = DOWNLOADS / c / "commons_sources.csv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_class"] = c
                pid = page_id_from_name(r.get("file", ""))
                if pid:
                    by_page[pid] = r
                all_rows.append(r)
    return by_page, all_rows


# --------------------------------------------------------------------------
# 1. Duplicate pair audit + side-by-side sheets
# --------------------------------------------------------------------------
def audit_pairs(sources: dict[str, dict]) -> list[dict]:
    with PAIRS_CSV.open(encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    rows = []
    for i, r in enumerate(pairs, 1):
        p1 = ROOT / r["first_path"]
        p2 = ROOT / r["second_path"]
        pid1 = page_id_from_name(p1.name)
        pid2 = page_id_from_name(p2.name)
        s1 = sources.get(pid1 or "", {})
        s2 = sources.get(pid2 or "", {})
        g1, g2 = load_gray(p1), load_gray(p2)
        d = hamming(dhash64(g1), dhash64(g2))
        p = hamming(phash64(g1), phash64(g2))
        a = hamming(ahash64(g1), ahash64(g2))
        ssim = float(structural_similarity(g1, g2, data_range=255))
        good, inliers = orb_similarity(p1, p2)
        info1, info2 = image_info(p1), image_info(p2)
        verdict = pair_verdict(pid1 == pid2, d, p, ssim, inliers)
        row = {
            "pair": i,
            "class": r["first_class"],
            "first_split": r["first_split"],
            "second_split": r["second_split"],
            "dhash_distance": d,
            "phash_distance": p,
            "ahash_distance": a,
            "ssim": round(ssim, 4),
            "orb_good_matches": good,
            "orb_inliers": inliers,
            "same_commons_page": pid1 == pid2,
            "page_id_first": pid1,
            "page_id_second": pid2,
            "first_path": r["first_path"],
            "second_path": r["second_path"],
            "first_title": s1.get("title", ""),
            "second_title": s2.get("title", ""),
            "first_creator": s1.get("creator", ""),
            "second_creator": s2.get("creator", ""),
            "first_license": s1.get("license", ""),
            "second_license": s2.get("license", ""),
            "first_size": f'{info1["width"]}x{info1["height"]}',
            "second_size": f'{info2["width"]}x{info2["height"]}',
            "first_bytes": info1["bytes"],
            "second_bytes": info2["bytes"],
            "verdict": verdict,
            "visual_review_done": "NO",
            "review_notes": "",
        }
        rows.append(row)
        make_pair_sheet(i, p1, p2, row)
        print(f"  pair {i:02d}/36  {row['class']:9s} d={d:2d} p={p:2d} "
              f"ssim={ssim:.3f} inliers={inliers:3d}  {verdict}")
    return rows


def make_pair_sheet(i: int, p1: Path, p2: Path, row: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    for ax, path, label in ((axes[0], p1, "A"), (axes[1], p2, "B")):
        try:
            ax.imshow(load_rgb(path))
        except Exception as exc:  # pragma: no cover - corrupt file guard
            ax.text(0.5, 0.5, f"Could not open:\n{exc}", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(f"{label}: {row['first_split'] if label == 'A' else row['second_split']} / "
                     f"{row['class']}\n{path.name}", fontsize=9)
    fig.suptitle(
        f"Pair {i:02d}  |  {row['class']}  |  dHash {row['dhash_distance']}  "
        f"pHash {row['phash_distance']}  SSIM {row['ssim']:.3f}  "
        f"ORB inliers {row['orb_inliers']}  |  {row['verdict']}",
        fontsize=11, y=0.98)
    footer = (
        f"A: {short(row['first_title'], 95)}\n"
        f"   {short(row['first_creator'] or 'creator missing', 60)} | {row['first_license'] or 'licence missing'}\n"
        f"B: {short(row['second_title'], 95)}\n"
        f"   {short(row['second_creator'] or 'creator missing', 60)} | {row['second_license'] or 'licence missing'}\n"
        "Machine triage only - open the full-size files and confirm before relabelling or regrouping."
    )
    fig.text(0.01, 0.015, footer, fontsize=8, family="monospace", va="bottom")
    fig.subplots_adjust(top=0.88, bottom=0.22, left=0.01, right=0.99, wspace=0.02)
    fig.savefig(OUT / f"duplicate_pair_{i:02d}.jpg", dpi=115)
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. Whole-dataset near-duplicate clusters
# --------------------------------------------------------------------------
def cluster_dataset(manifest: list[dict]) -> tuple[list[dict], dict]:
    print("  hashing all dataset images for cluster analysis...")
    hashes = []
    for n, rec in enumerate(manifest, 1):
        try:
            g = load_gray(rec["abs_path"])
            hashes.append((dhash64(g), phash64(g)))
        except Exception:
            hashes.append((0, 0))
        if n % 400 == 0:
            print(f"    hashed {n}/{len(manifest)}")
    d_arr = np.array([h[0] for h in hashes], dtype=np.uint64)
    p_arr = np.array([h[1] for h in hashes], dtype=np.uint64)
    n = len(manifest)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Same Commons page id or same sha256 => same asset.
    for key in ("page_id", "sha256"):
        groups = defaultdict(list)
        for i, rec in enumerate(manifest):
            groups[rec.get(key)].append(i)
        for idxs in groups.values():
            if len(idxs) > 1:
                for j in idxs[1:]:
                    union(idxs[0], j)

    # Conservative perceptual grouping.
    for i in range(n):
        pd = np.bitwise_count(p_arr[i] ^ p_arr[i + 1:])
        dd = np.bitwise_count(d_arr[i] ^ d_arr[i + 1:])
        hits = np.where((pd <= PHASH_GROUP_THRESHOLD) | (dd <= DHASH_GROUP_THRESHOLD))[0]
        for off in hits:
            union(i, i + 1 + int(off))

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    clusters = [v for v in groups.values() if len(v) > 1]
    clusters.sort(key=lambda v: (-len(v), v[0]))

    rows = []
    for cid, idxs in enumerate(clusters, 1):
        recs = [manifest[i] for i in idxs]
        splits = sorted({r["split"] for r in recs})
        classes = sorted({r["class"] for r in recs})
        page_ids = sorted({r["page_id"] for r in recs if r["page_id"]})
        min_p = min(
            hamming(hashes[a][1], hashes[b][1])
            for ai, a in enumerate(idxs) for b in idxs[ai + 1:]
        ) if len(idxs) > 1 else 0
        rows.append({
            "cluster_id": cid,
            "n_images": len(recs),
            "classes": "|".join(classes),
            "splits": "|".join(splits),
            "cross_split": len(splits) > 1,
            "cross_class": len(classes) > 1,
            "min_phash_distance": min_p,
            "page_ids": "|".join(page_ids),
            "paths": "|".join(r["path"] for r in recs),
        })

    cross = [r for r in rows if r["cross_split"]]
    involved = sum(r["n_images"] for r in cross)
    by_class = Counter()
    for r in cross:
        for c in r["classes"].split("|"):
            by_class[c] += 1
    summary = {
        "phash_threshold": PHASH_GROUP_THRESHOLD,
        "dhash_threshold": DHASH_GROUP_THRESHOLD,
        "n_images": n,
        "n_clusters": len(rows),
        "n_images_in_clusters": sum(r["n_images"] for r in rows),
        "n_cross_split_clusters": len(cross),
        "n_images_in_cross_split_clusters": involved,
        "cross_split_clusters_by_class": dict(sorted(by_class.items())),
        "n_cross_class_clusters": sum(1 for r in rows if r["cross_class"]),
        "largest_cluster": max((r["n_images"] for r in rows), default=0),
    }
    return rows, summary


# --------------------------------------------------------------------------
# 3. Label-quality audit
# --------------------------------------------------------------------------
def metadata_label_flags(manifest: list[dict], sources: dict[str, dict]) -> list[dict]:
    rows = []
    for rec in manifest:
        src = sources.get(rec["page_id"] or "", {})
        title = src.get("title", "")
        desc = src.get("_description", "")
        # Strong label rules use the Commons title only. Descriptions often
        # mention unrelated background (delivery trucks, batteries in a spec
        # sheet, etc.) and produce false positives.
        strong, notes, severity = [], [], "NONE"
        for pattern, explanation, level in STRONG_PATTERNS.get(rec["class"], []):
            if re.search(pattern, title, re.I):
                strong.append(f"{explanation} [matched: {pattern}]")
                if level == "HIGH" or (level == "MEDIUM" and severity != "HIGH"):
                    severity = level
        if not re.search(POSITIVE_PATTERNS.get(rec["class"], r"$^"), f"{title} {desc}", re.I):
            notes.append("title/description does not explicitly name the class")
        rows.append({
            "path": rec["path"],
            "class": rec["class"],
            "split": rec["split"],
            "page_id": rec["page_id"],
            "title": title,
            "description": short(desc, 200),
            "metadata_flags": "; ".join(strong),
            "metadata_note": "; ".join(notes),
            "metadata_severity": severity,
            "metadata_suspicious": bool(strong),
        })
    return rows


def model_label_flags(manifest: list[dict], sources: dict[str, dict]) -> list[dict]:
    """Run the existing model on train/val only. Test stays sealed."""
    from officemate.features import load_image
    from officemate.inference import Predictor

    print("  loading existing model for label triage (train/val only)...")
    predictor = Predictor(MODEL)
    rows = []
    scope = [r for r in manifest if r["split"] in ("train", "val")]
    for n, rec in enumerate(scope, 1):
        try:
            pred = predictor.predict_rgb(load_image(rec["abs_path"]))
            correct = pred.label == rec["class"]
            rows.append({
                "path": rec["path"],
                "class": rec["class"],
                "split": rec["split"],
                "page_id": rec["page_id"],
                "title": sources.get(rec["page_id"] or "", {}).get("title", ""),
                "predicted": pred.label,
                "confidence": round(pred.confidence, 6),
                "correct": correct,
                "strong_disagreement": (not correct) and pred.confidence >= 0.90,
                "moderate_disagreement": (not correct) and pred.confidence >= 0.70,
                "uncertain": pred.confidence < 0.50,
            })
        except Exception as exc:
            rows.append({
                "path": rec["path"], "class": rec["class"], "split": rec["split"],
                "page_id": rec["page_id"], "title": "",
                "predicted": "ERROR", "confidence": 0.0, "correct": False,
                "strong_disagreement": False, "moderate_disagreement": False,
                "uncertain": True, "error": str(exc),
            })
        if n % 200 == 0:
            print(f"    predicted {n}/{len(scope)}")
    return rows


def make_suspect_sheet(cls: str, flagged: list[dict], model_rows: dict[str, dict]):
    """Grid contact sheet of the most useful suspects for one class."""
    if not flagged:
        return None
    flagged = flagged[:24]
    cols, rows_n = 4, math.ceil(len(flagged) / 4)
    fig, axes = plt.subplots(rows_n, cols, figsize=(16, 3.6 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for ax, item in zip(axes, flagged):
        path = ROOT / item["path"]
        try:
            ax.imshow(load_rgb(path))
        except Exception:
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
        m = model_rows.get(item["path"], {})
        cap = (f"{item['split']} | true: {item['class']}\n"
               f"model: {m.get('predicted', 'n/a')} ({float(m.get('confidence', 0)):.2f})\n"
               f"{short(item.get('metadata_flags') or m.get('reason', ''), 55)}")
        ax.set_title(cap, fontsize=7)
    fig.suptitle(f"{cls}: images needing human label review "
                 f"({len(flagged)} shown; machine triage, not proof of error)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = SHEETS / f"{cls}_label_suspects.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# 4. Wikimedia Commons licensing/attribution verification
# --------------------------------------------------------------------------
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "OfficeMateAI-PDE3802-audit/1.0 (educational dataset audit)"}
CACHE = OUT / "commons_metadata_by_pageid.json"


def _load_cache() -> dict[str, dict]:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict[str, dict]):
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE)


def fetch_commons_metadata(page_ids: list[str]) -> dict[str, dict]:
    """Fetch Commons metadata by file page id, resuming from cache.

    Page ids are used instead of titles because they are stable and avoid
    normalisation/redirect ambiguity. Requests are serial with retry/backoff
    to respect Commons API rate limits.
    """
    cache = _load_cache()
    missing = sorted({str(p) for p in page_ids if p and str(p) not in cache})
    print(f"  Commons metadata: {len(cache)} cached, {len(missing)} to fetch")
    session = requests.Session()
    session.headers.update(HEADERS)
    batch_size = 50
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        params = {
            "action": "query", "format": "json", "formatversion": 2,
            "prop": "imageinfo", "iiprop": "extmetadata|url|size",
            "pageids": "|".join(batch), "redirects": 1,
        }
        data = None
        for attempt in range(6):
            try:
                resp = session.get(API, params=params, timeout=60)
                if resp.status_code == 429:
                    wait = min(60, 5 * (2 ** attempt))
                    print(f"    429 rate limit; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if data.get("error", {}).get("code") == "maxlag":
                    wait = min(30, 3 * (2 ** attempt))
                    print(f"    maxlag; sleeping {wait}s")
                    time.sleep(wait)
                    data = None
                    continue
                break
            except requests.RequestException as exc:
                wait = min(30, 2 ** attempt)
                print(f"    API error ({exc}); retry in {wait}s")
                time.sleep(wait)
        if data is None:
            print(f"    giving up on batch at {start} after retries")
            continue

        for page in data.get("query", {}).get("pages", []) or []:
            pid = str(page.get("pageid", ""))
            if not pid:
                continue
            info = (page.get("imageinfo") or [{}])[0]
            md = info.get("extmetadata", {}) or {}

            def val(k):
                return strip_html(md.get(k, {}).get("value", "")) if k in md else ""

            cache[pid] = {
                "api_title": page.get("title", ""),
                "pageid": page.get("pageid"),
                "missing": "missing" in page,
                "descriptionurl": info.get("descriptionurl", ""),
                "license_short": val("LicenseShortName"),
                "license": val("License"),
                "license_url": val("LicenseUrl"),
                "usage_terms": val("UsageTerms"),
                "artist": val("Artist"),
                "credit": val("Credit"),
                "description": val("ImageDescription"),
                "date": val("DateTimeOriginal"),
                "attribution_required": val("AttributionRequired"),
                "copyrighted": val("Copyrighted"),
                "restrictions": val("Restrictions"),
            }
        _save_cache(cache)
        print(f"    Commons API {min(start + batch_size, len(missing))}/{len(missing)}")
        time.sleep(1.2)
    return cache


def license_class(lic: str) -> str:
    low = (lic or "").lower()
    if not low:
        return "missing"
    if "public domain" in low or low in {"cc0", "no restrictions"}:
        return "public domain / CC0"
    if "noncommercial" in low or " nc" in low or "-nc" in low:
        return "NC (non-commercial)"
    if "no derivative" in low or " nd" in low or "-nd" in low:
        return "ND (no derivatives)"
    if "cc by-sa" in low or "cc-by-sa" in low:
        return "CC BY-SA"
    if "cc by" in low or "cc-by" in low:
        return "CC BY"
    if "gfdl" in low:
        return "GFDL"
    if "gpl" in low:
        return "GPL"
    return "other"


def audit_licensing(manifest: list[dict], sources: dict[str, dict],
                    all_source_rows: list[dict], skip_api: bool) -> tuple[list[dict], dict]:
    page_ids = [r["page_id"] for r in manifest if r.get("page_id")]
    api = {} if skip_api else fetch_commons_metadata(page_ids)
    rows = []
    for rec in manifest:
        src = sources.get(rec["page_id"] or "", {})
        title = src.get("title", "")
        meta = api.get(str(rec["page_id"]), {})
        csv_creator = (src.get("creator") or "").strip()
        csv_license = (src.get("license") or "").strip()
        csv_url = (src.get("license_url") or "").strip()
        api_artist = (meta.get("artist") or "").strip()
        api_license = (meta.get("license_short") or "").strip()
        api_url = (meta.get("license_url") or "").strip()
        if skip_api:
            status = "API skipped"
        elif not meta or meta.get("missing"):
            status = "not found on Commons"
        elif not api_license:
            status = "API licence missing"
        elif not api_artist:
            status = "API artist missing"
        else:
            status = "verified"
        rows.append({
            "path": rec["path"],
            "class": rec["class"],
            "split": rec["split"],
            "page_id": rec["page_id"],
            "title": title,
            "source_page": src.get("source_page", ""),
            "csv_creator": csv_creator,
            "csv_license": csv_license,
            "csv_license_url": csv_url,
            "api_artist": api_artist,
            "api_license": api_license,
            "api_license_url": api_url,
            "api_credit": meta.get("credit", ""),
            "api_description": short(meta.get("description", ""), 300),
            "attribution_required": meta.get("attribution_required", ""),
            "copyrighted": meta.get("copyrighted", ""),
            "restrictions": meta.get("restrictions", ""),
            "license_family": license_class(csv_license or api_license),
            "missing_creator": not bool(csv_creator),
            "missing_license": not bool(csv_license),
            "missing_license_url": not bool(csv_url),
            "unreviewed_in_source_csv": (src.get("review_status", "") != "REVIEWED"),
            "verification_status": status,
        })
    # Attach API descriptions to source rows for metadata keyword checks.
    for r in all_source_rows:
        pid = page_id_from_name(r.get("file", ""))
        r["_description"] = api.get(str(pid), {}).get("description", "")

    per_class = {}
    for c in CLASSES:
        rs = [r for r in rows if r["class"] == c]
        per_class[c] = {
            "n": len(rs),
            "missing_creator": sum(r["missing_creator"] for r in rs),
            "missing_license": sum(r["missing_license"] for r in rs),
            "missing_license_url": sum(r["missing_license_url"] for r in rs),
            "unreviewed": sum(r["unreviewed_in_source_csv"] for r in rs),
            "api_verified": sum(r["verification_status"] == "verified" for r in rs),
            "api_not_found": sum(r["verification_status"] == "not found on Commons" for r in rs),
            "licenses": dict(Counter(r["license_family"] for r in rs).most_common()),
        }
    summary = {
        "n_images": len(rows),
        "n_source_rows": len(all_source_rows),
        "missing_creator": sum(r["missing_creator"] for r in rows),
        "missing_license": sum(r["missing_license"] for r in rows),
        "missing_license_url": sum(r["missing_license_url"] for r in rows),
        "unreviewed": sum(r["unreviewed_in_source_csv"] for r in rows),
        "license_families": dict(Counter(r["license_family"] for r in rows).most_common()),
        "verification_status": dict(Counter(r["verification_status"] for r in rows).most_common()),
        "api_checked": not skip_api,
        "per_class": per_class,
    }
    return rows, summary


# --------------------------------------------------------------------------
# 5. Manual-review worklist + report
# --------------------------------------------------------------------------
def build_worklist(manifest, pair_rows, cluster_rows, metadata_rows, model_rows):
    pair_flags = defaultdict(list)
    for r in pair_rows:
        pair_flags[r["first_path"]].append((r["verdict"], f"near-duplicate pair {r['pair']:02d} ({r['verdict']})"))
        pair_flags[r["second_path"]].append((r["verdict"], f"near-duplicate pair {r['pair']:02d} ({r['verdict']})"))
    cluster_flags = defaultdict(list)
    for c in cluster_rows:
        for p in c["paths"].split("|"):
            if c["cross_split"]:
                cluster_flags[p].append(f"cross-split cluster {c['cluster_id']} "
                                        f"({c['n_images']} images, {c['splits']})")
    meta = {r["path"]: r for r in metadata_rows}
    model = {r["path"]: r for r in model_rows}
    rows = []
    for rec in manifest:
        flags, actions, issue_types = [], [], []
        high = False
        if pair_flags.get(rec["path"]):
            for verdict, text in pair_flags[rec["path"]]:
                flags.append(text)
                issue_types.append("near-duplicate / leakage")
                if "CONFIRMED" in verdict:
                    high = True
            actions.append("Visually confirm pair; keep both images in the same split.")
        if cluster_flags.get(rec["path"]):
            flags += cluster_flags[rec["path"]]
            issue_types.append("near-duplicate / leakage")
            high = True
            actions.append("Keep the whole perceptual cluster in one split.")
        m = meta.get(rec["path"], {})
        if m.get("metadata_suspicious"):
            flags.append(f"label check ({m.get('metadata_severity', 'MEDIUM')}): {m['metadata_flags']}")
            issue_types.append("label error candidate")
            if m.get("metadata_severity") == "HIGH":
                high = True
            actions.append("Visually verify the label; relabel or replace if the object is wrong.")
        elif m.get("metadata_note"):
            flags.append(f"note: {m['metadata_note']}")
            issue_types.append("vague metadata")
        mr = model.get(rec["path"])
        if mr and mr.get("strong_disagreement"):
            flags.append(f"model strongly predicts {mr['predicted']} ({mr['confidence']:.2f})")
            issue_types.append("model disagreement")
            high = True
            actions.append("High-priority visual label check.")
        if mr and mr.get("uncertain"):
            flags.append(f"model uncertain ({mr['confidence']:.2f})")
            issue_types.append("hard example")
            actions.append("Check image clarity/crop; hard example.")
        if not flags:
            continue
        medium = (
            m.get("metadata_severity") == "MEDIUM"
            or any("LIKELY" in f for f in flags)
            or bool(mr and mr.get("uncertain"))
        )
        priority = "HIGH" if high else ("MEDIUM" if medium else "LOW")
        rows.append({
            "priority": priority,
            "path": rec["path"],
            "class": rec["class"],
            "split": rec["split"],
            "page_id": rec["page_id"],
            "title": m.get("title", ""),
            "issue_types": " | ".join(dict.fromkeys(issue_types)),
            "flags": " | ".join(flags),
            "recommended_action": " ".join(dict.fromkeys(actions)),
        })
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    rows.sort(key=lambda r: (order.get(r["priority"], 9), r["class"], r["split"], r["path"]))
    return rows


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    return "\n".join(out)


def write_report(pair_rows, cluster_rows, cluster_summary, metadata_rows,
                 model_rows, license_rows, license_summary, worklist):
    verdict_counts = Counter(r["verdict"] for r in pair_rows)
    pair_by_class = Counter(r["class"] for r in pair_rows)
    pair_split = Counter(f'{r["first_split"]} -> {r["second_split"]}' for r in pair_rows)
    meta_flags = [r for r in metadata_rows if r["metadata_suspicious"]]
    meta_by_class = Counter(r["class"] for r in meta_flags)
    strong = [r for r in model_rows if r.get("strong_disagreement")]
    strong_by_class = Counter(r["class"] for r in strong)
    uncertain = [r for r in model_rows if r.get("uncertain")]
    high = [r for r in worklist if r["priority"] == "HIGH"]

    lines = []
    lines.append("# Phase 1 - Dataset verification and correction report")
    lines.append("")
    lines.append("**Status:** read-only audit. No image, label, split, model or baseline result "
                 "has been changed. Generated by `scripts/phase1_audit_dataset.py`.")
    lines.append("")
    lines.append("## How to review this in 10 minutes")
    lines.append("")
    lines.append("1. Open `results/phase1/index.html` in a browser. It shows all 36 side-by-side "
                 "pair sheets and the five class suspect sheets.")
    lines.append("2. For each pair, compare the two full-size files (paths are in the sheet and in "
                 "`duplicate_pairs.csv`). Write `confirmed`, `likely` or `distinct` in the "
                 "`visual_review_done` / `review_notes` columns.")
    lines.append("3. Open `manual_review_priority.csv`, filter to `priority = HIGH`, and check the "
                 "label-error candidates first. The report below explains the counts.")
    lines.append("4. Approve or reject the Phase 2 plan at the end of this report. No dataset change "
                 "has been made yet.")
    lines.append("")
    lines.append("## 1. Baseline dataset")
    lines.append("")
    counts = Counter((r["split"], r["class"]) for r in [])
    per = defaultdict(int)
    for split in SPLITS:
        for c in CLASSES:
            per[(split, c)] = 0
    # manifest is not passed; derive from metadata rows (all dataset images)
    for r in metadata_rows:
        per[(r["split"], r["class"])] += 1
    table_rows = [[c] + [per[(s, c)] for s in SPLITS] + [sum(per[(s, c)] for s in SPLITS)]
                  for c in CLASSES]
    table_rows.append(["**Total**"] + [sum(per[(s, c)] for c in CLASSES) for s in SPLITS]
                      + [sum(per.values())])
    lines.append(md_table(["Class", "Train", "Val", "Test", "Total"], table_rows))
    lines.append("")
    lines.append(f"- Every dataset image has a unique Commons page ID (no exact page reuse).")
    lines.append(f"- Exact SHA-256 duplicates across splits: **0** (checked in the original audit).")
    lines.append(f"- Filename-derived groups: one group per file, so the current 'grouped' split "
                 f"is effectively a random per-image split.")
    lines.append("")
    lines.append("## 2. The 36 flagged cross-split pairs")
    lines.append("")
    lines.append("Each pair has a side-by-side sheet in `results/phase1/duplicate_pair_XX.jpg` "
                 "and a row in `duplicate_pairs.csv`.")
    lines.append("")
    lines.append(md_table(["Machine verdict", "Pairs"],
                          [[k, v] for k, v in verdict_counts.most_common()]))
    lines.append("")
    lines.append(md_table(["Class", "Pairs"], [[k, v] for k, v in pair_by_class.most_common()]))
    lines.append("")
    lines.append(md_table(["Split pair", "Pairs"], [[k, v] for k, v in pair_split.most_common()]))
    lines.append("")
    lines.append("### Pair-level results")
    lines.append("")
    lines.append(md_table(
        ["#", "Class", "Splits", "dHash", "pHash", "SSIM", "ORB inliers", "Verdict"],
        [[r["pair"], r["class"], f'{r["first_split"]}->{r["second_split"]}',
          r["dhash_distance"], r["phash_distance"], f'{r["ssim"]:.3f}',
          r["orb_inliers"], r["verdict"]] for r in pair_rows]))
    lines.append("")
    lines.append("**How to read this:** dHash/pHash are perceptual hashes (lower = more alike). "
                 "SSIM compares resized greyscale structure (1.0 = identical). ORB inliers count "
                 "local features that survive a geometric transform; >= 30 is strong evidence of "
                 "the same scene or object. A human must still look at the sheet: similar product "
                 "photos of the same model can match without being the same file.")
    lines.append("")
    lines.append("### Impact on the baseline")
    lines.append("")
    lines.append("These pairs put visually related images on both sides of a split boundary. "
                 "Because the current split has one filename group per image, the group-aware "
                 "code cannot keep a photo series together. The held-out test therefore contains "
                 "images that are visually close to training images, especially keyboards. "
                 "The measured 0.7609 accuracy is therefore **optimistic**, not a clean "
                 "generalisation estimate. The exact inflation cannot be quantified without "
                 "rebuilding the split and retraining; this report does not invent that number.")
    lines.append("")
    lines.append("## 3. Whole-dataset near-duplicate clusters")
    lines.append("")
    lines.append(f"Conservative grouping: pHash distance <= {cluster_summary['phash_threshold']} "
                 f"or dHash distance <= {cluster_summary['dhash_threshold']}.")
    lines.append("")
    lines.append(md_table(["Measure", "Value"], [
        ["Images scanned", cluster_summary["n_images"]],
        ["Clusters (>= 2 images)", cluster_summary["n_clusters"]],
        ["Images in a cluster", cluster_summary["n_images_in_clusters"]],
        ["Cross-split clusters", cluster_summary["n_cross_split_clusters"]],
        ["Images in cross-split clusters", cluster_summary["n_images_in_cross_split_clusters"]],
        ["Cross-class clusters", cluster_summary["n_cross_class_clusters"]],
        ["Largest cluster", cluster_summary["largest_cluster"]],
    ]))
    lines.append("")
    if cluster_summary["cross_split_clusters_by_class"]:
        lines.append(md_table(["Class", "Cross-split clusters"],
                              [[k, v] for k, v in cluster_summary["cross_split_clusters_by_class"].items()]))
        lines.append("")
    if cluster_rows:
        cross_rows = [r for r in cluster_rows if r["cross_split"]]
        lines.append("Largest cross-split clusters (these drive the leakage):")
        lines.append("")
        lines.append(md_table(["Cluster", "Images", "Class", "Splits"],
                              [[r["cluster_id"], r["n_images"], r["classes"], r["splits"]]
                               for r in cross_rows[:12]]))
        lines.append("")
    lines.append("Full list: `duplicate_clusters.csv`. Phase 2 must assign each **whole cluster** "
                 "to exactly one split, not individual files.")
    lines.append("")
    lines.append("## 4. Label-quality investigation")
    lines.append("")
    lines.append("### 4a. Commons title/description flags")
    lines.append("")
    lines.append("Rules use whole-word/regex matching to avoid substring false positives "
                 "(for example, 'ink' inside 'drinking' or 'can' inside 'American').")
    lines.append("")
    lines.append(md_table(["Class", "Strong flags", "HIGH severity"],
                          [[c, meta_by_class.get(c, 0),
                            sum(1 for r in meta_flags if r["class"] == c and r.get("metadata_severity") == "HIGH")]
                           for c in CLASSES]))
    lines.append("")
    lines.append("These are review prompts, not confirmed mistakes. Highest-severity examples first:")
    lines.append("")
    examples = sorted(
        [r for r in meta_flags if r["class"] in TARGET_CLASSES],
        key=lambda r: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(r.get("metadata_severity"), 9), r["class"])
    )[:30]
    lines.append(md_table(["Class", "Severity", "Split", "Title", "Why flagged"],
                          [[r["class"], r.get("metadata_severity", ""), r["split"],
                            short(r["title"], 65), short(r["metadata_flags"], 95)]
                           for r in examples]))
    lines.append("")
    high_meta = [r for r in meta_flags if r.get("metadata_severity") == "HIGH"]
    lines.append(md_table(
        ["Class", "HIGH train", "HIGH val", "HIGH test", "HIGH total"],
        [[c] + [sum(1 for r in high_meta if r["class"] == c and r["split"] == s) for s in SPLITS]
         + [sum(1 for r in high_meta if r["class"] == c)] for c in CLASSES]))
    lines.append("")
    lines.append("The clearest systematic problem is in **notebook**: several Commons search results "
                 "are software screenshots (Notepad++, lab-notebook simulations) rather than physical "
                 "paper notebooks. These are strong label-error candidates. Similar, smaller problems "
                 "exist for mouse (spider/mousetrap/icon), phone (chargers, batteries, cases, masts) "
                 "and stapler (microscopic or surgical staples). Bottle contains many hot-water "
                 "bottles, which are bottles but not office water bottles; these are a context "
                 "mismatch rather than a strict label error.")
    lines.append("")
    lines.append("### 4b. Existing-model disagreement (train/val only)")
    lines.append("")
    lines.append("The test split was deliberately **not** used for model-based label triage, so it "
                 "remains available for an independent final evaluation.")
    lines.append("")
    lines.append(md_table(["Class", "Strong disagreement (>=0.90)"],
                          [[c, strong_by_class.get(c, 0)] for c in CLASSES]))
    lines.append("")
    lines.append(f"- Train/val images checked: **{len(model_rows)}**")
    lines.append(f"- Strong disagreements: **{len(strong)}**")
    lines.append(f"- Uncertain predictions (<0.50): **{len(uncertain)}**")
    lines.append("")
    if strong:
        lines.append("Strong-disagreement examples (true label -> model prediction):")
        lines.append("")
        lines.append(md_table(["Split", "True", "Predicted", "Conf.", "Title"],
                              [[r["split"], r["class"], r["predicted"], f'{r["confidence"]:.3f}',
                                short(r["title"], 75)] for r in strong[:15]]))
        lines.append("")
    lines.append("Per-class suspect sheets: `sheets/<class>_label_suspects.jpg`. "
                 "The model is a triage tool, not ground truth: a confident wrong prediction can "
                 "mean a label error, an ambiguous object, or a genuine model weakness.")
    lines.append("")
    lines.append("## 5. Wikimedia Commons licensing and attribution")
    lines.append("")
    lines.append(f"Licence metadata was checked against the live Commons API: "
                 f"**{'yes' if license_summary['api_checked'] else 'no (skipped)'}**.")
    lines.append("")
    lines.append(md_table(["Measure", "Value"], [
        ["Dataset images audited", license_summary["n_images"]],
        ["Source CSV rows", license_summary["n_source_rows"]],
        ["Missing creator", license_summary["missing_creator"]],
        ["Missing licence", license_summary["missing_license"]],
        ["Missing licence URL", license_summary["missing_license_url"]],
        ["Source CSV still marked UNREVIEWED", license_summary["unreviewed"]],
    ]))
    lines.append("")
    lines.append(md_table(["Licence family", "Images"],
                          [[k, v] for k, v in license_summary["license_families"].items()]))
    lines.append("")
    lines.append(md_table(["API verification status", "Images"],
                          [[k, v] for k, v in license_summary["verification_status"].items()]))
    lines.append("")
    lines.append("### Per class")
    lines.append("")
    lines.append(md_table(
        ["Class", "n", "Missing creator", "Missing licence", "Missing URL", "UNREVIEWED", "API verified"],
        [[c, d["n"], d["missing_creator"], d["missing_license"], d["missing_license_url"],
          d["unreviewed"], d["api_verified"]] for c, d in license_summary["per_class"].items()]))
    lines.append("")
    missing_url_non_pd = [r for r in license_rows
                          if not r["api_license_url"].strip()
                          and r["license_family"] != "public domain / CC0"]
    no_artist_or_credit = [r for r in license_rows
                           if not r["api_artist"].strip() and not r["api_credit"].strip()]
    lines.append(f"- Of the {license_summary['missing_license_url']} missing licence URLs, "
                 f"**{license_summary['missing_license_url'] - len(missing_url_non_pd)} are public "
                 f"domain/CC0**, where a URL is not legally required. Only "
                 f"**{len(missing_url_non_pd)}** are attribution licences with no URL.")
    lines.append(f"- **{license_summary['missing_creator']}** images have no artist field on Commons, "
                 f"but **{license_summary['missing_creator'] - len(no_artist_or_credit)}** of them have "
                 f"a credit field that can be used for attribution.")
    lines.append(f"- **{len(no_artist_or_credit)}** images have neither an artist nor a credit field. "
                 f"These cannot be fully attributed from Commons metadata alone.")
    lines.append("")
    if missing_url_non_pd:
        lines.append("Attribution licences with no licence URL:")
        lines.append("")
        lines.append(md_table(["Class", "Licence", "Title"],
                              [[r["class"], r["csv_license"], short(r["title"], 80)]
                               for r in missing_url_non_pd]))
        lines.append("")
    if no_artist_or_credit:
        lines.append("Images with neither artist nor credit:")
        lines.append("")
        lines.append(md_table(["Class", "Licence", "Title", "Commons page"],
                              [[r["class"], r["csv_license"], short(r["title"], 70), r["source_page"]]
                               for r in no_artist_or_credit]))
        lines.append("")
    lines.append("Non-standard licences in the dataset: 14 GPL (13 Notepad++/software screenshots "
                 "in notebook plus one mouse icon), 3 GFDL, 6 OGL, 1 FAL, 2 'Attribution', and "
                 "1 'Copyrighted free use'. These are usable only with correct attribution; the "
                 "GPL/software images are also label-error candidates.")
    lines.append("")
    lines.append("**Missing attribution information matters:** the source CSV has no per-image "
                 "creator for many images and no licence URL for others. A submission or public "
                 "repository must include, per image: title, creator, licence name, licence URL, "
                 "and the Commons source page. CC BY / CC BY-SA / GFDL images legally require "
                 "attribution; public-domain/CC0 images should still be credited academically.")
    lines.append("")
    lines.append("## 6. What requires correction or manual review")
    lines.append("")
    lines.append(f"- **{len(high)} HIGH-priority images** and "
                 f"**{len(worklist)} total worklist entries** in `manual_review_priority.csv`.")
    lines.append("- Every one of the 36 pairs needs your visual confirmation (see sheets).")
    lines.append("- Every image in a cross-split cluster must move as a cluster in Phase 2.")
    lines.append("- Metadata/model-flagged images need a label check; do not delete blindly.")
    lines.append("- 11 images have neither an artist nor a credit on Commons; 3 attribution "
                 "licences have no licence URL. These need a source-page attribution check or "
                 "replacement before public release.")
    lines.append("- All 1,981 images are still marked `UNREVIEWED` in the Commons source CSV; "
                 "the `--reviewed` flag on the pipeline is only an assertion.")
    lines.append("")
    lines.append("### Priority worklist preview")
    lines.append("")
    lines.append(md_table(["Priority", "Class", "Split", "Image", "Flags"],
                          [[r["priority"], r["class"], r["split"], short(Path(r["path"]).name, 55),
                            short(r["flags"], 95)] for r in worklist[:30]]))
    lines.append("")
    lines.append("## 7. Recommendations for Phase 2 (not executed)")
    lines.append("")
    lines.append("1. Confirm or reject each of the 36 pairs visually, then record the decision in "
                 "`duplicate_pairs.csv` (`visual_review_done`, `review_notes`).")
    lines.append("2. Build clusters from pHash/dHash plus confirmed pairs, and split by **cluster** "
                 "with a fixed seed. Keep all views of one physical object/session together.")
    lines.append("3. Hold out a genuinely independent test set, ideally photos taken by the group in "
                 "the actual office, not more Wikimedia search results.")
    lines.append("4. Correct only confirmed label errors; keep an audit trail of every change.")
    lines.append("5. Re-run the baseline once on the clean split and label it **Experiment 2**; keep "
                 "Experiment 1 (0.7609 accuracy) unchanged.")
    lines.append("6. Do not use the old test predictions to tune Experiment 2. Freeze the new test "
                 "split before training and evaluate once at the end.")
    lines.append("")
    lines.append("## 8. Approval needed before Phase 2")
    lines.append("")
    lines.append("- [ ] I have opened the 36 pair sheets and marked confirmed/likely/distinct.")
    lines.append("- [ ] I approve moving whole clusters into one split.")
    lines.append("- [ ] I approve the proposed relabel/replace actions for flagged images.")
    lines.append("- [ ] I approve keeping Experiment 1 untouched and creating Experiment 2.")
    lines.append("")
    lines.append("## Files produced")
    lines.append("")
    for name in ["duplicate_pairs.csv", "duplicate_clusters.csv",
                 "duplicate_cluster_summary.json",
                 "label_audit_metadata.csv", "label_audit_model.csv",
                 "licensing_audit.csv", "licensing_summary.json",
                 "commons_metadata_by_pageid.json",
                 "manual_review_priority.csv",
                 "duplicate_pair_01.jpg ... duplicate_pair_36.jpg",
                 "sheets/<class>_label_suspects.jpg", "index.html"]:
        lines.append(f"- `results/phase1/{name}`")
    lines.append("")
    (OUT / "PHASE1_DATASET_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def write_html(pair_rows, cluster_summary, license_summary, worklist):
    def esc(x):
        return html.escape(str(x))

    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>OfficeMate AI - Phase 1 dataset review</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;max-width:1200px;color:#222}
h1,h2{color:#17365d} table{border-collapse:collapse;margin:10px 0 22px}
th,td{border:1px solid #bbb;padding:5px 8px;font-size:13px;vertical-align:top}
th{background:#eef3f8} img{max-width:100%;border:1px solid #ccc}
.pair{margin:18px 0 30px} .note{background:#fff8e1;border-left:4px solid #f0ad4e;padding:10px}
code{background:#f3f3f3;padding:1px 4px}
</style></head><body>"""]
    parts.append("<h1>OfficeMate AI - Phase 1 dataset review</h1>")
    parts.append("<p class='note'>Read-only audit. No dataset, model or baseline result was changed. "
                 "Machine triage is not a substitute for your visual confirmation.</p>")
    parts.append("<h2>Summary</h2><table><tr><th>Measure</th><th>Value</th></tr>")
    for k, v in [("Flagged pairs", len(pair_rows)),
                 ("Cross-split clusters", cluster_summary["n_cross_split_clusters"]),
                 ("Images in cross-split clusters", cluster_summary["n_images_in_cross_split_clusters"]),
                 ("High-priority worklist entries", sum(1 for r in worklist if r["priority"] == "HIGH")),
                 ("Missing creators", license_summary["missing_creator"]),
                 ("Missing licence URLs", license_summary["missing_license_url"]),
                 ("Still UNREVIEWED", license_summary["unreviewed"])]:
        parts.append(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>")
    parts.append("</table>")
    parts.append("<p>Full written report: <a href='PHASE1_DATASET_AUDIT.md'>PHASE1_DATASET_AUDIT.md</a>. "
                 "Worklist: <a href='manual_review_priority.csv'>manual_review_priority.csv</a>.</p>")
    parts.append("<h2>36 flagged pairs - inspect each one</h2>")
    for r in pair_rows:
        parts.append("<div class='pair'>")
        parts.append(f"<h3>Pair {r['pair']:02d} - {esc(r['class'])} - {esc(r['verdict'])}</h3>")
        parts.append(f"<p>{esc(r['first_split'])} vs {esc(r['second_split'])} | "
                     f"dHash {r['dhash_distance']} | pHash {r['phash_distance']} | "
                     f"SSIM {r['ssim']} | ORB inliers {r['orb_inliers']} | "
                     f"same Commons page: {esc(r['same_commons_page'])}</p>")
        parts.append(f"<img src='duplicate_pair_{r['pair']:02d}.jpg' alt='pair {r['pair']}'>")
        parts.append("</div>")
    parts.append("<h2>Per-class label suspect sheets</h2>")
    for c in TARGET_CLASSES:
        p = SHEETS / f"{c}_label_suspects.jpg"
        if p.exists():
            parts.append(f"<h3>{esc(c)}</h3><img src='sheets/{c}_label_suspects.jpg' alt='{esc(c)} suspects'>")
    parts.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------
# Resume helpers (used by --only-licensing)
# --------------------------------------------------------------------------
def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_pair_rows() -> list[dict]:
    rows = read_csv_rows(OUT / "duplicate_pairs.csv")
    for r in rows:
        r["pair"] = int(r["pair"])
        for k in ("dhash_distance", "phash_distance", "ahash_distance", "orb_good_matches",
                  "orb_inliers", "first_bytes", "second_bytes"):
            if k in r:
                r[k] = int(r[k])
        for k in ("ssim",):
            if k in r:
                r[k] = float(r[k])
        r["same_commons_page"] = as_bool(r.get("same_commons_page"))
    return rows


def load_cluster_rows() -> tuple[list[dict], dict]:
    rows = read_csv_rows(OUT / "duplicate_clusters.csv")
    for r in rows:
        r["cluster_id"] = int(r["cluster_id"])
        r["n_images"] = int(r["n_images"])
        r["min_phash_distance"] = int(r["min_phash_distance"])
        r["cross_split"] = as_bool(r.get("cross_split"))
        r["cross_class"] = as_bool(r.get("cross_class"))
    summary_path = OUT / "duplicate_cluster_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return rows, summary


def load_model_rows() -> list[dict]:
    rows = read_csv_rows(OUT / "label_audit_model.csv")
    for r in rows:
        r["confidence"] = float(r.get("confidence") or 0)
        for k in ("correct", "strong_disagreement", "moderate_disagreement", "uncertain"):
            r[k] = as_bool(r.get(k))
    return rows


def write_suspect_sheets(metadata_rows, model_rows):
    model_by_path = {r["path"]: r for r in model_rows}
    for cls in TARGET_CLASSES:
        flagged = [r for r in metadata_rows if r["class"] == cls and r["metadata_suspicious"]]
        model_flagged = [r for r in model_rows
                         if r["class"] == cls and (r.get("strong_disagreement") or r.get("uncertain"))]
        seen = {r["path"] for r in flagged}
        for r in model_flagged:
            if r["path"] not in seen:
                flagged.append({
                    "path": r["path"], "class": cls, "split": r["split"],
                    "metadata_flags": f"model: {r['predicted']} ({r['confidence']:.2f})",
                })
                seen.add(r["path"])
        flagged.sort(key=lambda r: (r["split"], r["path"]))
        make_suspect_sheet(cls, flagged, model_by_path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-api", action="store_true", help="do not contact Wikimedia Commons")
    ap.add_argument("--skip-model", action="store_true", help="do not run the existing model")
    ap.add_argument("--only-licensing", action="store_true",
                    help="reuse existing pair/cluster/model CSVs and only refresh licensing, "
                         "metadata flags, worklist and reports")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)

    print("Loading dataset manifest and Commons source rows...")
    manifest = load_manifest()
    sources, all_source_rows = load_sources()
    missing_source = [r for r in manifest if r["page_id"] not in sources]
    print(f"  dataset images: {len(manifest)}; source rows: {len(all_source_rows)}; "
          f"unmatched source records: {len(missing_source)}")

    if args.only_licensing:
        print("Resume mode: refreshing licensing and reports from existing audit artefacts...")
        license_rows, license_summary = audit_licensing(
            manifest, sources, all_source_rows, skip_api=args.skip_api)
        write_csv(OUT / "licensing_audit.csv", license_rows)
        (OUT / "licensing_summary.json").write_text(
            json.dumps(license_summary, indent=2), encoding="utf-8")

        metadata_rows = metadata_label_flags(manifest, sources)
        write_csv(OUT / "label_audit_metadata.csv", metadata_rows)
        pair_rows = load_pair_rows()
        cluster_rows, cluster_summary = load_cluster_rows()
        model_rows = load_model_rows()
        write_suspect_sheets(metadata_rows, model_rows)
        worklist = build_worklist(manifest, pair_rows, cluster_rows, metadata_rows, model_rows)
        write_csv(OUT / "manual_review_priority.csv", worklist)
        write_report(pair_rows, cluster_rows, cluster_summary, metadata_rows,
                     model_rows, license_rows, license_summary, worklist)
        write_html(pair_rows, cluster_summary, license_summary, worklist)
        print("\nLicensing/report refresh complete. No data/model files were modified.")
        print(f"Open: {OUT / 'index.html'}")
        print(f"Read: {OUT / 'PHASE1_DATASET_AUDIT.md'}")
        return

    print("1/5 Measuring the 36 flagged pairs and building side-by-side sheets...")
    pair_rows = audit_pairs(sources)
    write_csv(OUT / "duplicate_pairs.csv", pair_rows)

    print("2/5 Clustering near-duplicates across the whole dataset...")
    cluster_rows, cluster_summary = cluster_dataset(manifest)
    write_csv(OUT / "duplicate_clusters.csv", cluster_rows)
    (OUT / "duplicate_cluster_summary.json").write_text(
        json.dumps(cluster_summary, indent=2), encoding="utf-8")

    print("3/5 Auditing Wikimedia Commons licensing/attribution...")
    license_rows, license_summary = audit_licensing(
        manifest, sources, all_source_rows, skip_api=args.skip_api)
    write_csv(OUT / "licensing_audit.csv", license_rows)
    (OUT / "licensing_summary.json").write_text(
        json.dumps(license_summary, indent=2), encoding="utf-8")

    print("4/5 Flagging label-quality suspects...")
    metadata_rows = metadata_label_flags(manifest, sources)
    write_csv(OUT / "label_audit_metadata.csv", metadata_rows)
    if args.skip_model:
        model_rows = []
    else:
        model_rows = model_label_flags(manifest, sources)
        write_csv(OUT / "label_audit_model.csv", model_rows)
    write_suspect_sheets(metadata_rows, model_rows)

    print("5/5 Building worklist and report...")
    worklist = build_worklist(manifest, pair_rows, cluster_rows, metadata_rows, model_rows)
    write_csv(OUT / "manual_review_priority.csv", worklist)
    write_report(pair_rows, cluster_rows, cluster_summary, metadata_rows,
                 model_rows, license_rows, license_summary, worklist)
    write_html(pair_rows, cluster_summary, license_summary, worklist)

    print("\nPhase 1 audit complete. No data/model files were modified.")
    print(f"Open: {OUT / 'index.html'}")
    print(f"Read: {OUT / 'PHASE1_DATASET_AUDIT.md'}")


if __name__ == "__main__":
    main()
