"""Generate a ten-class contact sheet and review report (NO automatic relabelling)."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from officemate.config import CLASSES, IMAGE_EXTENSIONS


def make_sheets(root: Path, out: Path, limit: int = 64) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    counts = {}
    for cls in CLASSES:
        folder = root / cls
        images = sorted(x for x in folder.rglob('*') if x.is_file()
                        and x.suffix.lower() in IMAGE_EXTENSIONS) if folder.exists() else []
        counts[cls] = len(images)
        pics = images[:limit]
        columns = 8
        cell_w, cell_h = 154, 148
        rows = (len(pics) + columns - 1) // columns
        sheet = Image.new('RGB', (columns * cell_w, max(1, rows) * cell_h), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(pics):
            x, y = (i % columns) * cell_w, (i // columns) * cell_h
            try:
                with Image.open(path) as im:
                    rgb = ImageOps.exif_transpose(im).convert('RGB')
                    sizes.setdefault(cls, []).append(rgb.size)
                    rgb.thumbnail((cell_w - 10, cell_h - 37))
                sheet.paste(rgb, (x + (cell_w - rgb.width) // 2, y + 4))
            except Exception:
                draw.text((x + 3, y + 6), 'INVALID IMAGE', fill='red')
            draw.text((x + 4, y + cell_h - 29), path.name[:22], fill='black')
            draw.text((x + 4, y + cell_h - 14), f'{i+1}/{len(images)}', fill='black')
        sheet.save(out / f'{cls}_review_sheet.jpg', quality=90)
    (out / 'image_counts.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    with (out / 'READ_ME_FIRST.txt').open('w', encoding='utf-8') as f:
        f.write('Human review is REQUIRED. For each class, inspect the original images, not only these thumbnails.\n')
        f.write('Remove drawings, duplicates, wrong categories, tiny or obscured items, sensitive data.\n')
        f.write('Confirm copyright permission and attribution; do not add test images from training.\n\n')
        f.write('\n'.join(f'{cls}: {n} imported' for cls, n in counts.items()))
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('data/curated_raw'))
    p.add_argument('--out', type=Path, default=Path('results/review'))
    p.add_argument('--limit', type=int, default=64)
    args = p.parse_args()
    counts = make_sheets(args.raw, args.out, limit=args.limit)
    print('Review sheets saved to', args.out)
    for k, n in counts.items():
        print(f'{k:12s}  {n:5d} images')
    print('Review each class manually before train/val/test splitting.')


if __name__ == '__main__':
    main()
