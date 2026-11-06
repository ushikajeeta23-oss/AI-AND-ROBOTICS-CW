"""Finish real-data training and evaluation using the report-specified YOLOv8n-cls.

Requires HUMAN-CHECKED photographs. This script refuses to claim benchmark
results if the real dataset is missing. Call after import and source review.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from officemate.config import CLASSES, DEFAULT_MODEL
from officemate.dataset import split_raw, audit
from train_yolo import train
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviewed', action='store_true', help='Group has reviewed ALL labels, photos and licenses.')
    p.add_argument('--raw', type=Path, default=Path('data/curated_raw'))
    p.add_argument('--data', type=Path, default=Path('data/office_dataset'))
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--results', type=Path, default=Path('results/REAL_YOLOV8N_TEST'))
    p.add_argument('--epochs', type=int, default=45)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if not args.reviewed:
        p.error('Photos must be reviewed first. Read results/review sheets, then add --reviewed.')
    missing = [x for x in CLASSES if not (args.raw / x).exists() or
               not any((args.raw / x).glob('*.*'))]
    if missing:
        p.error('Real photos are missing for: ' + ', '.join(missing))
    if args.data.exists() and any(p for p in args.data.iterdir() if p.name != '.gitkeep'):
        print(f'Existing dataset found: {args.data}; using it as-is without changing test set.')
    else:
        print('Preparing 70/15/15 grouped train/validation/test splits...')
        split_raw(args.raw, args.data)
    counts, duplicates = audit(args.data)
    if duplicates:
        raise RuntimeError(f'Same image in several splits: {duplicates[0]}')
    for split in ('train','val','test'):
        print(split, {k: counts[split][k] for k in CLASSES})
    if not args.model.exists():
        print('Training YOLOv8n-cls as specified in the report...')
        train(args.data, args.model, epochs=args.epochs, device=args.device)
    else:
        print(f'Reusing existing model: {args.model} (will not overwrite).')
    print('Evaluating the historical test split...')
    scores = evaluate(args.data, args.model, args.results)
    print('PROVISIONAL HISTORICAL TEST RESULT:', json.dumps({
        'accuracy': scores['accuracy'], 'macro_f1': scores['macro_f1'],
        'review_rate': scores['review_rate'], 'accepted_accuracy': scores['accepted_accuracy'],
        'confusion_matrix': str(args.results / 'confusion_matrix.png'),
    }, indent=2))

if __name__ == '__main__':
    main()
