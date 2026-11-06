"""Train report-specified YOLOv8n-cls with transfer learning (ten classes).

Validation data are used for training decisions, not the held-out test data.
The exported BEST model is copied to models/officemate_yolov8n_cls.pt.
"""
import argparse
import shutil
from pathlib import Path
from officemate.dataset import audit
from officemate.config import CLASSES, DEFAULT_MODEL, SEED


def train(data: Path, output: Path = DEFAULT_MODEL, *, epochs: int = 45,
          imgsz: int = 224, batch: int = 16, device: str = 'cpu', project='runs/officemate'):
    counts, duplicates = audit(data)
    if duplicates:
        raise ValueError(f'Exact images appear across data splits: {duplicates[:3]}')
    missing = [f'{split}/{c}' for split in ('train', 'val', 'test')
               for c in CLASSES if counts[split][c] < 1]
    if missing:
        raise ValueError('Need labelled images for every class in every split: ' + ', '.join(missing))
    if output.exists():
        raise FileExistsError(f'Output model already exists: {output}. Keep the previous experiment unchanged.')
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install the Ultralytics requirements first.') from exc
    pretrained = YOLO('yolov8n-cls.pt')  # Exact pretrained architecture stated in the report.
    history = pretrained.train(
        data=str(data.resolve()), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, seed=SEED, deterministic=True, workers=0,
        patience=12, plots=True, project=project, name='yolov8n_cls',
        exist_ok=False, verbose=True, augment=True,
    )
    run_dir = Path(pretrained.trainer.save_dir)
    best = run_dir / 'weights' / 'best.pt'
    if not best.is_file():
        raise RuntimeError(f'Expected best weights at {best}, but training did not produce them.')
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    print(f'Best YOLOv8n-cls model: {output}\nTraining charts: {run_dir}')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/office_dataset'))
    p.add_argument('--out', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--epochs', type=int, default=45)
    p.add_argument('--imgsz', type=int, default=224)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='cpu', help='cpu, 0 for first NVIDIA GPU, or mps')
    args = p.parse_args()
    train(args.data, args.out, epochs=args.epochs, imgsz=args.imgsz,
          batch=args.batch, device=args.device)

if __name__ == '__main__':
    main()
