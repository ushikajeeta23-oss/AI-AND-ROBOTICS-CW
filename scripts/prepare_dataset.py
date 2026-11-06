"""Create leakage-controlled YOLO-compatible class-folder splits."""
import argparse
from pathlib import Path
from officemate.dataset import split_raw, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('data/raw'))
    p.add_argument('--out', type=Path, default=Path('data/office_dataset'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    manifest = split_raw(args.raw, args.out, args.seed)
    counts, duplicates = audit(args.out)
    print(f'Wrote {len(manifest)} images to {args.out}')
    print({s: dict(v) for s, v in counts.items()})
    if duplicates:
        raise RuntimeError('Unexpected exact duplicate across splits')

if __name__ == '__main__':
    main()
