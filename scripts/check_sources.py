"""Print the configured class-specific dataset sources and local availability."""
import json
from pathlib import Path
from officemate.config import CLASSES, IMAGE_EXTENSIONS


def main():
    obj = json.loads(Path('sources.json').read_text(encoding='utf-8'))
    for cls in CLASSES:
        src = obj['sources'][cls]
        folder = Path('data/downloads') / cls
        count = sum(1 for p in folder.rglob('*') if p.is_file() and
                    (p.suffix.lower() in IMAGE_EXTENSIONS or p.suffix.lower() == '.zip')) if folder.exists() else 0
        print(f'[{cls:9}] Files/archives: {count} | Version: {src.get("version_hint") or "manual"}')
        print('  ' + src['url'])
        if src.get('note'):
            print('  NOTE:', src['note'])
    print('\nThe Kaggle bottle link is SYNTHETIC and excluded from the real-photograph dataset.')


if __name__ == '__main__':
    main()
