"""Report visually similar images across TRAIN / VAL / TEST for human review.

A difference-hash is a warning heuristic, not proof that two photos are duplicates.
For tens of thousands of images use a dedicated nearest-neighbour index.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageOps
from officemate.config import CLASSES, IMAGE_EXTENSIONS


def difference_hash(path: Path) -> int:
    with Image.open(path) as file:
        im = ImageOps.exif_transpose(file).convert('L').resize((9, 8))
        pixels = np.asarray(im, dtype=np.int16)
    diffs = pixels[:, 1:] > pixels[:, :-1]
    bits = 0
    for value in diffs.flat:
        bits = (bits << 1) | int(value)
    return bits


def find_candidates(dataset: Path, threshold: int = 3):
    records = []
    for split in ('train', 'val', 'test'):
        for cls in CLASSES:
            for path in (dataset / split / cls).rglob('*'):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append((split, cls, path, difference_hash(path)))
    matches = []
    for i, (split1, cls1, p1, hash1) in enumerate(records):
        for split2, cls2, p2, hash2 in records[i+1:]:
            if split1 == split2:
                continue
            distance = (hash1 ^ hash2).bit_count()
            if distance <= threshold:
                matches.append({'first_split': split1, 'first_class': cls1,
                                'first_path': str(p1), 'second_split': split2,
                                'second_class': cls2, 'second_path': str(p2),
                                'distance': distance})
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/office_dataset'))
    parser.add_argument('--out', type=Path, default=Path('results/possible_near_duplicates.csv'))
    parser.add_argument('--threshold', type=int, default=3)
    a = parser.parse_args()
    if not 0 <= a.threshold <= 64:
        parser.error('Threshold should be between 0 and 64.')
    warnings = find_candidates(a.data, a.threshold)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    columns = ['first_split', 'first_class', 'first_path', 'second_split', 'second_class',
               'second_path', 'distance']
    with a.out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(warnings)
    print(f'Possible cross-split near-duplicates: {len(warnings)}. Saved {a.out}')
    print('Inspect flagged pairs visually. Similar backgrounds/shapes can cause false alarms.')
    if warnings:
        print('For an academically valid test, resolve actual duplicate frames and rebuild the split.')


if __name__ == '__main__':
    main()
