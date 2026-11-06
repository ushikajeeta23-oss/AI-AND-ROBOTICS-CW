"""Fetch legitimate Roboflow dataset VERSION exports using your own account key.

Dataset overview pages cannot be downloaded as if they were ZIP file URLs.
Some projects have no public version; use their browser page's Fork/Dataset
interface to create an authorized export, or add photos manually.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class-name', dest='target', default='all',
                        help='one of ten classes or all (default: all)')
    parser.add_argument('--version', type=int, default=None,
                        help='Override version when known for ONE class')
    parser.add_argument('--root', type=Path, default=Path('data/downloads'))
    parser.add_argument('--format', default='yolov8',
                        choices=['yolov8', 'coco'], help='Roboflow source export format')
    args = parser.parse_args()
    sources = json.loads(Path('sources.json').read_text(encoding='utf-8'))['sources']
    if args.target != 'all' and args.target not in sources:
        parser.error(f'Unknown class: {args.target}; choices are {", ".join(sources)}')
    if args.version and args.target == 'all':
        parser.error('--version requires a specific --class-name')
    api_key = os.environ.get('ROBOFLOW_API_KEY', '').strip()
    if not api_key or api_key == 'put_your_private_key_here':
        parser.error('Set ROBOFLOW_API_KEY as a PRIVATE environment variable, or use manual dataset exports.')
    try:
        from roboflow import Roboflow
    except ImportError:
        parser.error('Missing SDK. Run: python -m pip install -r requirements-sources.txt')
    rf = Roboflow(api_key=api_key)
    targets = [args.target] if args.target != 'all' else list(sources)
    for cls in targets:
        info = sources[cls]
        v = args.version or info.get('version_hint')
        if not v:
            print(f'[{cls}] SKIPPED: this project has no confirmed export version. '
                  f'Open {info["url"]} and create/download an authorized version or save photos manually.')
            continue
        dest = args.root / cls / f'{info["project"]}_v{v}'
        if dest.exists() and any(dest.iterdir()):
            print(f'[{cls}] Already downloaded at {dest}; no overwrite.')
            continue
        print(f'[{cls}] Downloading {info["url"]}, version {v}, format {args.format}')
        try:
            project = rf.workspace(info['workspace']).project(info['project'])
            dataset = project.version(v).download(args.format, location=str(dest))
            print(f'[{cls}] Saved to {dataset.location}')
        except Exception as exc:
            print(f'[{cls}] COULD NOT FETCH: {type(exc).__name__}: {exc}\n'
                  'Use the Roboflow web Download/Fork workflow for this project '
                  'or place an authorized ZIP in data/downloads/<class>/.')
    print('Finished source download attempts; imports have NOT run yet. Review licenses and photos.')


if __name__ == '__main__':
    main()
