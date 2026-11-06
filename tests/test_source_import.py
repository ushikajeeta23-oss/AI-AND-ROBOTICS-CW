"""Tests use temporary artificial patterns strictly for code verification."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import csv
import json
import numpy as np
from PIL import Image

from officemate.source_import import canonical, source_group, import_downloads
from officemate.config import CLASSES
from officemate.dataset import split_raw, audit


def _source_config():
    return {'sources': {c: {'project': f'{c}-data', 'url': f'https://example.org/{c}',
                            'license': 'sample', 'author': 'fixture', 'aliases': [c]}
                        for c in CLASSES}}


def _make_png(color):
    image = Image.new('RGB', (100, 100), color=color)
    b = BytesIO()
    image.save(b, format='PNG')
    return b.getvalue()


def test_source_label_normalization():
    assert canonical('Office-Bin') == 'office bin'
    assert source_group('sample_png.rf.abcdef1234567890', 'mug') == source_group('sample_png', 'mug')


def test_import_yolo_zip_keeps_zip_unmodified_and_crops(tmp_path):
    downloads = tmp_path / 'downloads'
    d = downloads / 'mug'
    d.mkdir(parents=True)
    archive = d / 'mug_export.zip'
    with ZipFile(archive, 'w') as z:
        z.writestr('data.yaml', 'names: ["Mug", "Noise"]\n')
        z.writestr('train/images/mugA.jpg', _make_png((140, 40, 80)))
        z.writestr('train/labels/mugA.txt', '0 0.5 0.5 0.7 0.7\n1 0.5 0.5 0.8 0.8\n')
    original = archive.read_bytes()
    imported = tmp_path / 'curated'
    summary = import_downloads(downloads, imported, _source_config(), use_crops=True)
    assert summary['mug']['added'] == 1
    assert archive.read_bytes() == original
    outputs = list((imported / 'mug').glob('*.jpg'))
    assert len(outputs) == 1
    with Image.open(outputs[0]) as im:
        assert im.width < 100
        assert im.height < 100
    manifest = list(csv.DictReader((imported / 'source_manifest.csv').open(encoding='utf-8')))
    assert manifest[0]['annotation'] == 'yolo_box_or_polygon'
    assert manifest[0]['needs_human_review'] == 'YES'


def test_copy_full_original_without_annotations(tmp_path):
    root = tmp_path / 'downloads'
    for i, cls in enumerate(CLASSES):
        f = root / cls / 'photos'
        f.mkdir(parents=True)
        (f / f'{cls}_01.png').write_bytes(_make_png((i * 22, 60, 80)))
    out = tmp_path / 'curated'
    result = import_downloads(root, out, _source_config(), use_crops=True)
    assert all(result[c]['added'] == 1 for c in CLASSES)
    for cls in CLASSES:
        path = next((out / cls).glob('*.png'))
        assert path.read_bytes() == (root / cls / 'photos' / f'{cls}_01.png').read_bytes()
    with __import__('pytest').raises(FileExistsError):
        import_downloads(root, out, _source_config())


def test_import_coco_bbox(tmp_path):
    downloads = tmp_path / 'downloads'
    f = downloads / 'mug' / 'folder' / 'train'
    f.mkdir(parents=True)
    (f / 'a.jpg').write_bytes(_make_png((110, 30, 20)))
    coco = {'categories':[{'id':3, 'name':'mug'}],
            'images':[{'id': 1, 'file_name':'a.jpg'}],
            'annotations':[{'image_id':1, 'category_id':3, 'bbox':[10,10,60,70]}]}
    (f / '_annotations.coco.json').write_text(json.dumps(coco))
    out = tmp_path / 'curated'
    summary = import_downloads(downloads, out, _source_config())
    assert summary['mug']['added'] == 1
    assert list((out/'mug').glob('*.jpg'))
