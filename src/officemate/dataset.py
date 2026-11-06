"""Dataset indexing, file hash checking and group-aware train/val/test splitting."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import csv
import random
import shutil
from .config import CLASSES, IMAGE_EXTENSIONS, SEED
from .features import load_image


def images(folder: Path):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def indexed_split(dataset: Path, split: str):
    for label in CLASSES:
        for path in images(dataset / split / label):
            yield path, label


def audit(dataset: Path):
    counts = {}
    hashes = {}
    overlaps = []
    for split in ('train', 'val', 'test'):
        counts[split] = Counter()
        for path, label in indexed_split(dataset, split):
            load_image(path)  # Decode to detect corrupt images before training/evaluation.
            counts[split][label] += 1
            digest = sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                overlaps.append((str(hashes[digest]), str(path)))
            hashes[digest] = path
    return counts, overlaps


def split_raw(raw: Path, destination: Path, seed: int = SEED):
    """Group by the prefix before '__' (one physical object/capture session per group).

    Filename example: notebook_session03__view001.jpg. Requires >= 3 groups
    per class; groups cannot leak between train, val and test. Refuses overwrite.
    """
    if destination.exists() and any(p for p in destination.iterdir() if p.name != '.gitkeep'):
        raise FileExistsError(f'Output {destination} is not empty. Move it before rebuilding.')
    rng = random.Random(seed)
    manifest = []
    seen_hashes = set()
    for label in CLASSES:
        source_dir = raw / label
        paths = images(source_dir) if source_dir.is_dir() else []
        if not paths:
            raise ValueError(f'No images provided for {label}. Expected {source_dir}')
        groups = {}
        for p in paths:
            if '__' not in p.stem:
                raise ValueError(f'Filename {p.name} needs a group prefix, e.g. item01__photo001.jpg')
            key = p.stem.split('__', 1)[0]
            groups.setdefault(key, []).append(p)
        keys = sorted(groups)
        if len(keys) < 3:
            raise ValueError(f'{label}: need >=3 independent groups; found {len(keys)}.')
        rng.shuffle(keys)
        n = len(keys)
        # Rough 70/15/15 group proportions. At least one group in val and test.
        n_test = max(1, round(n * .15))
        n_val = max(1, round(n * .15))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f'{label}: not enough groups for all splits.')
        allocated = {'train': keys[:n_train], 'val': keys[n_train:n_train + n_val],
                     'test': keys[n_train + n_val:]}
        for split, group_keys in allocated.items():
            for group in group_keys:
                for image in groups[group]:
                    digest = sha256(image.read_bytes()).hexdigest()
                    if digest in seen_hashes:
                        raise ValueError(f'An exact duplicate image was detected: {image}')
                    seen_hashes.add(digest)
                    output = destination / split / label / image.name
                    # Catch two source extensions or sessions with identical destination names.
                    if output.exists():
                        raise FileExistsError(output)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image, output)
                    manifest.append({'split': split, 'class': label, 'group': group,
                                     'path': str(output.relative_to(destination)), 'sha256': digest})
    with (destination / 'manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'class', 'group', 'path', 'sha256'])
        writer.writeheader()
        writer.writerows(manifest)
    return manifest
