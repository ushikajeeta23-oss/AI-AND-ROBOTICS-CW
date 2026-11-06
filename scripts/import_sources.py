"""Import authorized Roboflow zip/folder exports; preserve downloaded originals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from officemate.source_import import import_downloads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--downloads', type=Path, default=Path('data/downloads'))
    p.add_argument('--out', type=Path, default=Path('data/curated_raw'))
    p.add_argument('--limit-per-class', type=int, default=0, help='0 keeps ALL available photos')
    p.add_argument('--full-photos', action='store_true',
                   help='Do not crop annotation boxes. Requires manual one-item visual check.')
    args = p.parse_args()
    if args.limit_per_class < 0:
        p.error('--limit-per-class must be zero or positive.')
    source_config = json.loads(Path('sources.json').read_text(encoding='utf-8'))
    report = import_downloads(args.downloads, args.out, source_config,
                              max_per_class=args.limit_per_class,
                              use_crops=not args.full_photos)
    for cls, meta in report.items():
        print(f'{cls:10s}: {meta.get("added", 0):5d} imported')
    print('Source originals remain unchanged. Curated data:', args.out.resolve())
    print('Provenance:', args.out / 'source_manifest.csv')
    print('IMPORTANT: visually inspect imported classes; remove wrong pictures before training.')


if __name__ == '__main__':
    main()
