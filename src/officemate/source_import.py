"""Convert user-authorized exports into a ten-class *classification* dataset.

Roboflow projects may provide object DETECTION or instance SEGMENTATION boxes.
The original export is never modified. We save full originals for already-single-
item images; when annotations exist, we create derivative object-only crops.
Every result is recorded in a provenance manifest for human review.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from hashlib import sha256, sha1
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import re
import shutil
import zipfile

from PIL import Image, ImageOps

from .config import CLASSES, IMAGE_EXTENSIONS


MANIFEST_COLUMNS = ['class', 'path', 'source_url', 'source_license', 'author',
                    'source_path', 'group', 'source_split', 'annotation',
                    'sha256', 'needs_human_review']


def canonical(text: object) -> str:
    """Simplify source label names without silently translating classes."""
    return re.sub(r'[^a-z0-9]+', ' ', str(text).strip().lower()).strip()


def source_group(stem: str, origin: str) -> str:
    # Roboflow augmented file names often add .rf.<hex> after the source stem.
    clean_stem = re.sub(r'\.rf\.[0-9a-f]{8,}$', '', stem, flags=re.I)
    readable = re.sub(r'[^a-zA-Z0-9]+', '-', clean_stem).strip('-')[:28] or 'image'
    identity = sha1((origin + ':' + clean_stem).encode('utf-8')).hexdigest()[:12]
    return f'{readable}-{identity}'


def _load_names(root: Path) -> dict[int, str]:
    # `data.yaml` can be present at the dataset root or above a split directory.
    for filename in ('data.yaml', 'dataset.yaml'):
        p = root / filename
        if p.exists():
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError('Install requirements-sources.txt to read YOLO data.yaml') from exc
            data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
            names = data.get('names', {})
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
            if isinstance(names, dict):
                return {int(i): str(n) for i, n in names.items()}
    p = root / 'classes.txt'
    if p.exists():
        return {i: n.strip() for i, n in enumerate(p.read_text(encoding='utf-8').splitlines())}
    return {}


def _clip_box(box: tuple[float, float, float, float], w: int, h: int):
    x0, y0, x1, y1 = box
    coords = (max(0, min(w, round(x0))), max(0, min(h, round(y0))),
              max(0, min(w, round(x1))), max(0, min(h, round(y1))))
    if coords[2] - coords[0] < 18 or coords[3] - coords[1] < 18:
        return None
    return coords


def _yolo_boxes(image: Path, split_root: Path, names: dict[int, str], aliases: set[str]):
    # Standard `train/images/a.jpg` -> `train/labels/a.txt`.
    if image.parent.name.lower() == 'images':
        label = image.parent.parent / 'labels' / (image.stem + '.txt')
    else:
        label = image.with_suffix('.txt')
    if not label.exists():
        return None
    with Image.open(image) as im:
        w, h = im.size
    boxes = []
    for line in label.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            points = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        # Unknown labels are NOT treated as our target category.
        if names:
            if canonical(names.get(cid, f'unknown {cid}')) not in aliases:
                continue
        elif cid != 0:
            continue  # Single-class export with no names file: only ID 0.
        if len(points) == 4:
            xc, yc, bw, bh = points
            box = ((xc - bw / 2) * w, (yc - bh / 2) * h,
                   (xc + bw / 2) * w, (yc + bh / 2) * h)
        elif len(points) >= 6 and len(points) % 2 == 0:
            xs = points[::2]
            ys = points[1::2]
            box = (min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h)
        else:
            continue
        box = _clip_box(box, w, h)
        if box:
            boxes.append(box)
    return boxes


def _coco_boxes(root: Path, aliases: set[str]):
    """Map COCO original relative filename to boxes. Supports COCO split folders."""
    output = defaultdict(list)
    seen_files = set()
    for annotation in root.rglob('*.json'):
        if 'coco' not in annotation.name.lower():
            continue
        try:
            data = json.loads(annotation.read_text(encoding='utf-8'))
            categories = {int(c['id']): canonical(c['name']) for c in data['categories']}
            originals = {int(i['id']): i['file_name'] for i in data['images']}
            records = data['annotations']
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        for item in records:
            if categories.get(int(item['category_id'])) not in aliases:
                continue
            file_name = originals.get(int(item['image_id']))
            if not file_name:
                continue
            x, y, w, h = item['bbox']
            candidate = annotation.parent / file_name
            if not candidate.exists():
                candidate = annotation.parent / 'images' / Path(file_name).name
            if not candidate.exists():
                continue
            output[candidate.resolve()].append((x, y, x + w, y + h))
            seen_files.add(candidate.resolve())
    return output, bool(seen_files)


def _images(root: Path):
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


@contextmanager
def _export_roots(source_dir: Path):
    """Support downloaded/extracted Roboflow datasets and authorized ZIP exports.

    Archives are extracted into a temporary folder without permitting ZIP traversal.
    Archives remain untouched in downloads/ so originals can always be recovered.
    """
    with TemporaryDirectory(prefix='officemate_import_') as tmp:
        tmp_root = Path(tmp)
        roots = []
        zips = sorted(source_dir.rglob('*.zip'))
        for i, archive in enumerate(zips):
            out = tmp_root / f'export_{i}'
            out.mkdir(parents=True)
            with zipfile.ZipFile(archive) as zf:
                info = zf.infolist()
                if sum(x.file_size for x in info) > 5 * 1024 ** 3:
                    raise ValueError(f'Archive expands to >5GB: {archive.name}')
                for entry in info:
                    p = out / entry.filename
                    if not p.resolve().is_relative_to(out.resolve()):
                        raise ValueError(f'Unsafe ZIP archive path: {entry.filename}')
                    if entry.is_dir():
                        p.mkdir(parents=True, exist_ok=True)
                    else:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(entry) as src, p.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
            roots.append((out, f'zip:{archive.name}'))
        # Standalone folder exports as well as JPGs placed directly in downloads/<class>.
        if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in source_dir.rglob('*') if p.is_file()):
            roots.append((source_dir, f'folder:{source_dir.name}'))
        yield roots


def import_downloads(downloads: Path, out: Path, source_cfg: dict,
                     max_per_class: int = 0, use_crops: bool = True):
    """Import author-provided exports; refuse overwrites and never guess a label.

    `use_crops=True` isolates target objects from detection datasets. Source full
    images stay unchanged in downloads/, and a manifest links each derived photo.
    If there are no labels, the full image is copied and needs manual review.
    """
    if out.exists() and any(p for p in out.iterdir() if p.name != '.gitkeep'):
        raise FileExistsError(f'{out} already contains data. Use a new output directory.')
    rows = []
    digests = set()
    summaries = {}
    for cls in CLASSES:
        entry = dict(source_cfg['sources'][cls])
        base = downloads / cls
        # Optional per-class source attribution override for self-captured or
        # replacement datasets. Never silently attribute them to another author.
        override = base / 'source_metadata.json'
        if override.is_file():
            metadata = json.loads(override.read_text(encoding='utf-8'))
            for field in ('url', 'license', 'author', 'project', 'aliases'):
                if field in metadata:
                    entry[field] = metadata[field]
        if not base.exists():
            summaries[cls] = {'added': 0, 'note': f'Missing directory: {base}'}
            continue
        allowed = {canonical(x) for x in entry['aliases']}
        allowed.add(canonical(cls))
        count = 0
        skipped = 0
        with _export_roots(base) as roots:
            for root, desc in roots:
                if max_per_class and count >= max_per_class:
                    break
                # metadata can be in a nested exporter folder; look for a single root data.yaml.
                meta_root = next((p.parent for p in root.rglob('data.yaml')), root)
                names = _load_names(meta_root)
                coco_boxes, has_coco = _coco_boxes(root, allowed)
                coco_exists = any('coco' in p.name.lower() for p in root.rglob('*.json'))
                for photo in _images(root):
                    if max_per_class and count >= max_per_class:
                        break
                    try:
                        with Image.open(photo) as im:
                            im.verify()
                        with Image.open(photo) as im:
                            w, h = im.size
                    except Exception:
                        skipped += 1
                        continue
                    if min(w, h) < 24:
                        skipped += 1
                        continue
                    yolo = _yolo_boxes(photo, root, names, allowed)
                    # Annotation-bearing detection images with zero valid target boxes are skipped.
                    if use_crops and yolo is not None:
                        boxes = yolo
                        ann = 'yolo_box_or_polygon'
                    elif use_crops and coco_exists:
                        boxes = coco_boxes.get(photo.resolve(), [])
                        ann = 'coco_box'
                    else:
                        boxes = [None]
                        ann = 'full_image_needs_review'
                    for index, box in enumerate(boxes):
                        if max_per_class and count >= max_per_class:
                            break
                        if box is not None:
                            box = _clip_box(box, w, h)
                            if box is None:
                                skipped += 1
                                continue
                            with Image.open(photo) as im:
                                rgb = ImageOps.exif_transpose(im).convert('RGB')
                                # A small context margin keeps useful shape/background cues.
                                x0, y0, x1, y1 = box
                                margin = round(max(x1 - x0, y1 - y0) * .06)
                                crop = rgb.crop((max(0, x0 - margin), max(0, y0 - margin),
                                                 min(w, x1 + margin), min(h, y1 + margin)))
                            buffer_path = None
                            # Encode only derived crop, never modify the downloaded original.
                            from io import BytesIO
                            io = BytesIO()
                            crop.save(io, format='JPEG', quality=94, subsampling=0)
                            contents = io.getvalue()
                            extension = '.jpg'
                        else:
                            contents = photo.read_bytes()  # retain original file exactly
                            extension = photo.suffix.lower()
                        digest = sha256(contents).hexdigest()
                        if digest in digests:
                            skipped += 1
                            continue
                        digests.add(digest)
                        group = source_group(photo.stem, entry['project'])
                        name = f'{group}__{index:02d}-{digest[:8]}{extension}'
                        dest = out / cls / name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(contents)
                        parts = set(photo.relative_to(root).parts)
                        split = next((x for x in ('train', 'valid', 'val', 'test') if x in parts), 'unknown')
                        rows.append({
                            'class': cls, 'path': str(dest.relative_to(out)),
                            'source_url': entry['url'], 'source_license': entry['license'],
                            'author': entry['author'],
                            'source_path': f'{desc}/{photo.relative_to(root).as_posix()}',
                            'group': group, 'source_split': split, 'annotation': ann,
                            'sha256': digest, 'needs_human_review': 'YES'
                        })
                        count += 1
        summaries[cls] = {'added': count, 'skipped_duplicate_or_invalid': skipped,
                          'source': entry['url']}
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'source_manifest.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out / 'import_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    return summaries
