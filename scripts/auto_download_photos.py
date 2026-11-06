"""Download real, source-traceable office photos ON THE STUDENT'S COMPUTER.

Source order: authorized Roboflow dataset exports already in data/downloads;
Roboflow API with your own credentials (when available); Open Images V7 for
nine supported categories; Wikimedia Commons for paper notebooks.

Keeps ORIGINAL photo bytes under data/source_originals/ and puts *separate*
annotated object crops under data/downloads/ for model training. NEVER invents
images or labels. All photos still require HUMAN REVIEW before final use.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
import os
import re
import shutil
import time
from pathlib import Path

from PIL import Image, ImageOps
from officemate.config import CLASSES, IMAGE_EXTENSIONS

OI_LABELS = {
    'bin': 'Waste container', 'book': 'Book', 'bottle': 'Bottle',
    'chair': 'Chair', 'keyboard': 'Computer keyboard',
    'mouse': 'Computer mouse', 'mug': 'Mug',
    'phone': 'Mobile phone', 'stapler': 'Stapler',
}
NOTEBOOK_SEARCH = ('paper notebook stationery', 'spiral notebook stationery',
                   'exercise book pages', 'notebook writing paper')


def has_images(path: Path) -> bool:
    if not path.exists():
        return False
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
               for p in path.rglob('*')) or any(path.rglob('*.zip'))


def write_metadata(destination: Path, source: dict):
    destination.mkdir(parents=True, exist_ok=True)
    meta = destination / 'source_metadata.json'
    if not meta.exists():
        meta.write_text(json.dumps(source, indent=2), encoding='utf-8')


def get_openimages(target: str, n: int, downloads: Path,
                   originals: Path, *, seed: int = 42) -> int:
    try:
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError('FiftyOne is required for Open Images: pip install -r requirements-download.txt') from exc
    label = OI_LABELS[target]
    class_folder = downloads / target / 'open_images_v7'
    raw_folder = originals / target / 'open_images_v7'
    class_folder.mkdir(parents=True, exist_ok=True)
    raw_folder.mkdir(parents=True, exist_ok=True)
    write_metadata(downloads / target, {
        'url': 'https://storage.googleapis.com/openimages/web/index.html',
        'license': 'Open Images V7 (CC BY 2.0); check per-photo original attribution',
        'author': 'Open Images V7 image authors', 'project': 'openimages_v7',
        'aliases': [target],
    })
    print(f'[{target}] Open Images V7: requesting up to {n} photos of {label}.', flush=True)
    ds = foz.load_zoo_dataset(
        'open-images-v7', split='train', label_types=['detections'],
        classes=[label], max_samples=n * 2, only_matching=True,
        shuffle=True, seed=seed,
    )
    records = []
    written = 0
    for sample in ds:
        if written >= n:
            break
        detections = sample.get('detections')
        matches = [] if detections is None else [x for x in detections.detections
                                                  if x.label == label]
        if not matches:
            continue
        # Largest target box; one training image per source photo.
        box = max(matches, key=lambda d: d.bounding_box[2] * d.bounding_box[3]).bounding_box
        source = Path(sample.filepath)
        if not source.is_file():
            continue
        source_id = source.stem
        try:
            with Image.open(source) as picture:
                raw = ImageOps.exif_transpose(picture).convert('RGB')
                w, h = raw.size
                x, y, bw, bh = box
                margin = 0.06
                x0 = max(0, int((x - bw * margin) * w))
                y0 = max(0, int((y - bh * margin) * h))
                x1 = min(w, int((x + bw * (1 + margin)) * w))
                y1 = min(h, int((y + bh * (1 + margin)) * h))
                if x1 - x0 < 50 or y1 - y0 < 50:
                    continue
                region = raw.crop((x0, y0, x1, y1))
                if min(region.size) < 48:
                    continue
                original_path = raw_folder / f'{source_id}{source.suffix.lower()}'
                if not original_path.exists():
                    shutil.copy2(source, original_path)
                destination = class_folder / f'{target}_{source_id}.jpg'
                if destination.exists():
                    continue
                region.save(destination, quality=94, subsampling=0)
        except (OSError, ValueError):
            continue
        records.append({'class': target, 'photo_id': source_id,
                        'source_url': f'https://storage.googleapis.com/openimages/web/index.html#{source_id}',
                        'original_file': str(original_path),
                        'training_crop': str(destination),
                        'human_review': 'REQUIRED'})
        written += 1
    with (class_folder / 'source_index.csv').open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['class', 'photo_id', 'source_url',
                                          'original_file', 'training_crop', 'human_review'])
        wr.writeheader(); wr.writerows(records)
    print(f'[{target}] Saved {written} original photographs AND {written} separate crops.', flush=True)
    return written


def _strip_html(value):
    return re.sub(r'<[^>]+>', '', html.unescape(str(value or '')))


def get_commons_notebooks(n: int, downloads: Path, originals: Path) -> int:
    import requests
    output = downloads / 'notebook' / 'wikimedia_commons'
    raw = originals / 'notebook' / 'wikimedia_commons'
    output.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    write_metadata(downloads / 'notebook', {
        'url': 'https://commons.wikimedia.org/', 'license': 'Varies by file; see notebook_sources.csv',
        'author': 'Individual Wikimedia Commons contributors',
        'project': 'wikimedia_commons_notebook', 'aliases': ['notebook'],
    })
    agent = 'OfficeMate-AI-academic-photo-collection/1.0 (please respect Wikimedia Commons attribution)'
    session = requests.Session()
    session.headers['User-Agent'] = agent
    rows, seen = [], set()
    for query in NOTEBOOK_SEARCH:
        if len(rows) >= n:
            break
        offset = 0
        while len(rows) < n and offset < 400:
            params = {'action': 'query', 'generator': 'search', 'gsrsearch': query,
                      'gsrnamespace': 6, 'gsrlimit': 50, 'gsroffset': offset,
                      'prop': 'imageinfo', 'iiprop': 'url|size|mime|extmetadata',
                      'format': 'json', 'formatversion': 2}
            try:
                reply = session.get('https://commons.wikimedia.org/w/api.php',
                                    params=params, timeout=35)
                reply.raise_for_status()
                info = reply.json()
            except (requests.RequestException, ValueError) as exc:
                print(f'[notebook] Commons API problem: {exc}', flush=True)
                break
            pages = (info.get('query') or {}).get('pages') or []
            if not pages:
                break
            for page in pages:
                if len(rows) >= n:
                    break
                title = page.get('title', '')
                imageinfo = page.get('imageinfo') or []
                if not imageinfo or page.get('pageid') in seen:
                    continue
                record = imageinfo[0]
                if record.get('mime') not in ('image/jpeg', 'image/png', 'image/webp'):
                    continue
                if record.get('size', 0) > 10_000_000 or record.get('size', 0) < 20000:
                    continue
                if not any(word in title.lower() for word in ('notebook', 'exercise book', 'spiral')):
                    continue
                if any(word in title.lower() for word in ('computer', 'laptop', 'software', 'screenshot')):
                    continue
                url = record.get('url')
                if not url:
                    continue
                extension = { 'image/jpeg': '.jpg', 'image/png': '.png',
                              'image/webp': '.webp'}[record['mime']]
                source_id = f"commons_{page['pageid']}"
                source_file = raw / f'{source_id}{extension}'
                target = output / f'notebook_{source_id}{extension}'
                if target.exists():
                    continue
                try:
                    response = session.get(url, timeout=45)
                    response.raise_for_status()
                    if len(response.content) > 10_000_000:
                        continue
                    source_file.write_bytes(response.content)
                    with Image.open(source_file) as test:
                        test.verify()
                    shutil.copy2(source_file, target)
                except (requests.RequestException, OSError):
                    source_file.unlink(missing_ok=True)
                    continue
                extra = record.get('extmetadata') or {}
                rows.append({'class': 'notebook', 'file': str(target),
                             'source_title': title, 'source_page': record.get('descriptionurl', ''),
                             'author': _strip_html((extra.get('Artist') or {}).get('value')),
                             'license': _strip_html((extra.get('LicenseShortName') or {}).get('value')),
                             'license_url': (extra.get('LicenseUrl') or {}).get('value', ''),
                             'review': 'REQUIRED'})
                seen.add(page['pageid'])
                time.sleep(0.15)
            offset += 50
    with (output / 'notebook_sources.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['class','file','source_title','source_page',
                                         'author','license','license_url','review'])
        w.writeheader(); w.writerows(rows)
    print(f'[notebook] Downloaded {len(rows)} Wikimedia notebook photo(s).', flush=True)
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--per-class', type=int, default=200,
                   help='200/class matches report; use 80/class for the earlier 800-photo aim.')
    p.add_argument('--source', choices=['auto', 'openimages', 'roboflow'], default='auto')
    p.add_argument('--downloads', type=Path, default=Path('data/downloads'))
    p.add_argument('--originals', type=Path, default=Path('data/source_originals'))
    p.add_argument('--class-name', default='all')
    a = p.parse_args()
    if a.class_name != 'all' and a.class_name not in CLASSES:
        p.error('Expected class: ' + ', '.join(CLASSES))
    if a.per_class < 3:
        p.error('At least 3 independent original photographs are required per class.')
    targets = [a.class_name] if a.class_name != 'all' else CLASSES
    summary = {}
    for target in targets:
        dst = a.downloads / target
        dst.mkdir(parents=True, exist_ok=True)
        if has_images(dst):
            summary[target] = 'Pre-existing authorized dataset found; left unchanged.'
            print(f'[{target}] Found your existing photos or ZIP; not replacing them.')
            continue
        try:
            # Roboflow requires the user's own account and some projects lack versions.
            if a.source in ('auto', 'roboflow') and os.getenv('ROBOFLOW_API_KEY'):
                from roboflow import Roboflow
                info = json.loads(Path('sources.json').read_text(encoding='utf-8'))['sources'][target]
                if info.get('version_hint'):
                    folder = dst / f"{info['project']}_v{info['version_hint']}"
                    project = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY']).workspace(
                        info['workspace']).project(info['project'])
                    project.version(info['version_hint']).download('yolov8', location=str(folder))
                    summary[target] = f'Authorized Roboflow version {info["version_hint"]} downloaded.'
                    continue
            if a.source == 'roboflow':
                summary[target] = 'No accessible published dataset version or no API key; use fork/export.'
                continue
            if target == 'notebook':
                got = get_commons_notebooks(a.per_class, a.downloads, a.originals)
            else:
                got = get_openimages(target, a.per_class, a.downloads, a.originals)
            summary[target] = f'{got} photographic files collected; human review required.'
        except Exception as exc:
            summary[target] = f'FAILED: {type(exc).__name__}: {exc}'
            print(f'[{target}] ERROR: {exc}', flush=True)
    a.downloads.mkdir(parents=True, exist_ok=True)
    (a.downloads / 'download_status.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n=== DOWNLOAD REPORT ===')
    for k, val in summary.items():
        print(f'{k}: {val}')
    missing = [c for c in targets if not has_images(a.downloads / c)]
    if missing:
        raise SystemExit('Missing photo sources for: ' + ', '.join(missing) +
                         '. Check download_status.json or export a dataset manually.')
    print('All classes have source photos. NEXT: import + review them; no accuracy measured yet.')

if __name__ == '__main__':
    main()
