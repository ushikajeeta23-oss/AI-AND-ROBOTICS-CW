"""Verify the dataset is decodable, split by class and free of exact duplicates."""
import argparse
from pathlib import Path
from officemate.dataset import audit
from officemate.config import CLASSES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/office_dataset'))
    a = p.parse_args()
    counts, overlaps = audit(a.data)
    for split, counter in counts.items():
        print(f'{split.upper()}: total={sum(counter.values())}')
        for label in CLASSES:
            print(f'  {label:12s} {counter[label]:4d}')
    if overlaps:
        raise SystemExit(f'FAIL: {len(overlaps)} exact duplicates, including {overlaps[0]}')
    if any(counts[split][c] == 0 for split in counts for c in CLASSES):
        raise SystemExit('FAIL: a class is absent from train/val/test.')
    print('PASS: class coverage and exact-hash leakage checks. Visual near-duplicates still need manual review.')

if __name__ == '__main__':
    main()
