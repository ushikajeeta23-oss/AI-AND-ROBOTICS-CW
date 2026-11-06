#!/usr/bin/env python3
"""Collect public REAL photographs for ten office-object classes in one batch.

Primary automatic source: Wikimedia Commons MediaWiki API (no login required).
Optional: Authorized Roboflow exports for projects with confirmed dataset versions,
using a key supplied as ROBOFLOW_API_KEY; unavailable projects fall back to Commons.

No Open Images full-train metadata. No invented pictures, labels or measured scores.
Every downloaded photo remains UNREVIEWED. See source CSV for photographer/rights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CLASSES = ('bin', 'book', 'bottle', 'chair', 'keyboard', 'mouse', 'mug', 'notebook', 'phone', 'stapler')
QUERIES = {
    'bin': ['office waste bin', 'wastebasket', 'trash can office', 'rubbish bin', 'waste paper basket'],
    'book': ['hardcover book', 'paperback book', 'textbook', 'closed books', 'book on table'],
    'bottle': ['water bottle', 'plastic drinking bottle', 'reusable bottle', 'water drinking bottle', 'empty bottle'],
    'chair': ['office chair', 'swivel desk chair', 'computer chair', 'ergonomic chair', 'task chair'],
    'keyboard': ['computer keyboard', 'desktop keyboard', 'wireless keyboard', 'keyboard computer desk'],
    'mouse': ['computer mouse', 'optical mouse computer', 'wireless mouse computer', 'USB mouse'],
    'mug': ['ceramic coffee mug', 'coffee mug', 'tea mug', 'mug cup', 'mug on desk'],
    'notebook': ['paper notebook', 'spiral notebook stationery', 'exercise notebook paper', 'writing notebook'],
    'phone': ['smartphone', 'mobile phone', 'cell phone', 'smartphone device', 'mobile phone on table'],
    'stapler': ['office stapler', 'desk stapler', 'stapler stationery', 'hand stapler'],
}
TITLE_HINTS = {
    'bin': ('bin','trash','waste','rubbish','basket'),
    'book': ('book','textbook','paperback','hardcover'),
    'bottle': ('bottle','flask','canteen'),
    'chair': ('chair','office seating','swivel'),
    'keyboard': ('keyboard',),
    'mouse': ('mouse','mice'),
    'mug': ('mug','cup'),
    'notebook': ('notebook','exercise book','spiral bound','notepad'),
    'phone': ('phone','smartphone','mobile','iphone','android handset'),
    'stapler': ('stapler','stapling'),
}
EXCLUDE = ('svg', 'logo', 'clipart', 'icon', 'cartoon', 'diagram', 'vector', 'screenshot',
           'screen shot','ai generated', 'illustration', 'drawing', 'painting', 'map ',
           'template', 'wikipedia screenshot', 'phone wallpaper')
CSV_FIELDS = ['class','file','source_page','download_url','creator','license','license_url','title',
              'sha256','source','review_status']
SUPPORTED = {'image/jpeg':'.jpg', 'image/png':'.png', 'image/webp':'.webp'}
API_URL = 'https://commons.wikimedia.org/w/api.php'


def safe_text(value):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', html.unescape(str(value or '')))).strip()


def seems_relevant(cls, title):
    name = title.lower().removeprefix('file:')
    if any(x in name for x in EXCLUDE): return False
    if not any(x in name for x in TITLE_HINTS[cls]): return False
    if cls == 'mouse' and any(x in name for x in ('mouse animal','field mouse','mickey mouse','mouse lemur','mouse deer','mouse bird', 'mouse-eared', 'mouse moth')): return False
    if cls == 'phone' and any(x in name for x in ('phone number','payphone','telephone booth','telephone exchange')): return False
    if cls == 'notebook' and any(x in name for x in ('computer notebook','notebook computer','laptop notebook','laptop')): return False
    if cls == 'book' and any(x in name for x in ('book cover illustration', 'comic book art', 'ebook')): return False
    return True


def http_session(contact):
    sess = requests.Session()
    # Wikimedia APIs require an identifiable user-agent. Operators may add
    # contact via WIKIMEDIA_CONTACT (email or full URL) on their own machine.
    if not contact:
        raise ValueError('Set WIKIMEDIA_CONTACT to YOUR contact email or public contact URL before using Wikimedia Commons API.')
    tag = contact
    sess.headers['User-Agent'] = f'OfficeMateAIStudentPhotoCollector/1.1 ({tag}) python-requests/{requests.__version__}'
    sess.headers['Accept'] = 'application/json,image/*;q=0.8,*/*;q=0.5'
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429,500,502,503,504],
                  allowed_methods=['GET'], respect_retry_after_header=True)
    sess.mount('https://', HTTPAdapter(max_retries=retry))
    return sess


def search_api(session, query, max_pages=5):
    offset = 0
    for _ in range(max_pages):
        params = {'action':'query', 'generator':'search', 'gsrsearch':query + ' filetype:bitmap',
                  'gsrnamespace':6, 'gsrlimit':50, 'gsroffset':offset,
                  'prop':'imageinfo', 'iiprop':'url|size|mime|extmetadata',
                  'iiurlwidth':1400,
                  'iiextmetadatafilter':'LicenseShortName|LicenseUrl|Artist',
                  'format':'json', 'formatversion':'2'}
        response = session.get(API_URL, params=params, timeout=50)
        response.raise_for_status()
        payload = response.json()
        if 'error' in payload: raise RuntimeError(str(payload['error']))
        pages = (payload.get('query') or {}).get('pages') or []
        if not pages: return
        for page in pages:
            info = page.get('imageinfo') or []
            if info: yield page, info[0]
        cont = payload.get('continue') or {}
        if 'gsroffset' not in cont: return
        offset = cont['gsroffset']
        time.sleep(0.45)


def stream_photo(session, url, path, cap_bytes):
    temp = path.with_name(path.name + '.part')
    total = 0
    try:
        with session.get(url, stream=True, timeout=(12,65)) as res:
            res.raise_for_status()
            if int(res.headers.get('Content-Length',0)) > cap_bytes: return None
            with temp.open('wb') as dst:
                for chunk in res.iter_content(chunk_size=262144):
                    if not chunk: continue
                    total += len(chunk)
                    if total > cap_bytes: return None
                    dst.write(chunk)
        with Image.open(temp) as picture:
            picture.verify()
        with Image.open(temp) as picture:
            w,h=picture.size
            if min(w,h) < 180 or max(w,h)>20000: return None
        digest = hashlib.sha256(temp.read_bytes()).hexdigest()
        temp.replace(path)
        return digest
    except (OSError, ValueError, requests.RequestException):
        return None
    finally:
        temp.unlink(missing_ok=True)


def count_photos(folder):
    return len([p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp')]) if folder.exists() else 0


def existing_row_data(path):
    if not path.is_file(): return [], set(), set()
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows, {x.get('source_page','') for x in rows}, {x.get('sha256','') for x in rows}


def commons_class(sess, cls, goal, downloads, max_mb, pause):
    base = downloads / cls
    out = base / 'commons'
    out.mkdir(parents=True, exist_ok=True)
    index = base / 'commons_sources.csv'
    old, prior_pages, prior_hashes = existing_row_data(index)
    # Do not overwrite existing Roboflow or manual exports, or mix source rights.
    prior_count = count_photos(base)
    found = 0
    remaining = max(0, goal-prior_count)
    if remaining == 0:
        print(f'[{cls}] Already has at least {goal} photos; skipping.', flush=True)
        return prior_count
    rows = list(old)
    print(f'[{cls}] Looking for {remaining} additional real photographs on Wikimedia Commons.', flush=True)
    for phrase in QUERIES[cls]:
        if found >= remaining: break
        try:
            for page, meta in search_api(sess, phrase, max_pages=6):
                if found >= remaining: break
                title = str(page.get('title',''))
                if not seems_relevant(cls,title): continue
                mime = meta.get('mime')
                if mime not in SUPPORTED: continue
                if int(meta.get('size',0)) < 20000: continue
                page_url = meta.get('descriptionurl') or ''
                if not page_url or page_url in prior_pages: continue
                if meta.get('filehidden'): continue
                url = meta.get('thumburl') or meta.get('url')
                if not url: continue
                ext = SUPPORTED[mime]
                name = f'{cls}_commons_{page.get("pageid")}{ext}'
                target = out / name
                if target.exists(): continue
                digest = stream_photo(sess,url,target,cap_bytes=max_mb*1024*1024)
                if not digest: continue
                if digest in prior_hashes:
                    target.unlink(missing_ok=True); continue
                extra=meta.get('extmetadata') or {}
                rows.append({'class':cls, 'file':str(target.as_posix()),
                             'source_page':page_url,'download_url':url,
                             'creator':safe_text((extra.get('Artist') or {}).get('value')),
                             'license':safe_text((extra.get('LicenseShortName') or {}).get('value')),
                             'license_url':safe_text((extra.get('LicenseUrl') or {}).get('value')),
                             'title':title,'sha256':digest,'source':'Wikimedia Commons',
                             'review_status':'UNREVIEWED'})
                found += 1
                prior_pages.add(page_url); prior_hashes.add(digest)
                if found % 10 == 0: print(f'[{cls}] {prior_count+found}/{goal} downloaded...',flush=True)
                time.sleep(pause)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            print(f"[{cls}] Search '{phrase}' temporarily failed: {str(exc)[:220]}", flush=True)
            time.sleep(2)
            continue
    with index.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS); writer.writeheader(); writer.writerows(rows)
    if found:
        (base/'source_metadata.json').write_text(json.dumps({
             'url':'https://commons.wikimedia.org/',
             'license':'Per-image licenses differ. See commons_sources.csv',
             'author':'Individual Commons photographers; see commons_sources.csv',
             'aliases':[cls], 'project':'wikimedia_commons_photographs'
        },indent=2),encoding='utf-8')
    print(f'[{cls}] +{found} photos this run; now {count_photos(base)}/{goal}. Manual review required.',flush=True)
    return count_photos(base)


def roboflow_class(cls, downloads, rf, sources):
    info=sources.get(cls, {})
    v=info.get('version_hint')
    if not v or not info.get('workspace') or not info.get('project'): return False
    base=downloads / cls
    if any(base.rglob('*.zip')) or count_photos(base)>0: return False
    dest=base/f'{info["project"]}_v{v}'
    if dest.exists() and any(dest.iterdir()): return False
    try:
        print(f'[{cls}] Trying permitted Roboflow export, version {v}...', flush=True)
        rf.workspace(info['workspace']).project(info['project']).version(v).download('yolov8', location=str(dest))
        print(f'[{cls}] Roboflow export saved to {dest}',flush=True)
        return True
    except Exception as exc:
        print(f'[{cls}] Roboflow unavailable ({type(exc).__name__}); trying Commons.',flush=True)
        return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--per-class',type=int,default=200,help='200 per class matches report target; use 80 for an earlier 800-image pilot')
    p.add_argument('--downloads',type=Path,default=Path('data/downloads'))
    p.add_argument('--max-mb',type=int,default=8)
    p.add_argument('--pause',type=float,default=0.5,help='Delay between image transfers to respect host resources')
    p.add_argument('--classes',nargs='+',choices=CLASSES,default=list(CLASSES))
    p.add_argument('--no-roboflow',action='store_true')
    a=p.parse_args()
    if not (3 <= a.per_class <= 500): p.error('--per-class must be 3 to 500')
    a.downloads.mkdir(parents=True,exist_ok=True)
    sess=http_session(os.getenv('WIKIMEDIA_CONTACT','').strip())
    sources_path=Path('sources.json')
    sources=(json.loads(sources_path.read_text(encoding='utf-8'))['sources'] if sources_path.exists() else {})
    rf=None
    key=os.getenv('ROBOFLOW_API_KEY','').strip()
    if key and not a.no_roboflow:
        try:
            from roboflow import Roboflow
            rf=Roboflow(api_key=key)
            print('Roboflow authentication available; using permitted versions when possible.')
        except Exception as exc:
            print(f'Roboflow could not be enabled: {type(exc).__name__}')
    summary={}
    for cls in a.classes:
        base=a.downloads / cls; base.mkdir(parents=True,exist_ok=True)
        if rf and roboflow_class(cls, a.downloads, rf, sources):
            summary[cls]={'image_files':count_photos(base),
                          'status':'authorized Roboflow export obtained; importer can read its files'}
            # Keep Roboflow provenance separate; do not overwrite with a Commons label.
            continue
        # Existing Roboflow zips cannot be counted without importing, avoid replacing them.
        if any(base.rglob('*.zip')):
            summary[cls]={'image_files': count_photos(base), 'status':'existing ZIP export found; importer can read it'}
            print(f'[{cls}] Found source ZIP; keeping unchanged.')
            continue
        try:
            n=commons_class(sess,cls,a.per_class,a.downloads,a.max_mb,a.pause)
            summary[cls]={'image_files':n, 'status':'needs human visual review' if n else 'NO IMAGES FOUND'}
        except Exception as exc:
            summary[cls]={'image_files':count_photos(base), 'status':f'ERROR: {type(exc).__name__}: {exc}'}
            print(f'[{cls}] ERROR {exc}',flush=True)
        time.sleep(0.8)
    output=a.downloads/'batch_download_status.json'
    output.write_text(json.dumps(summary, indent=2),encoding='utf-8')
    print('\n==== ALL 10 CLASSES: CURRENT DOWNLOAD STATUS ====')
    for cls,info in summary.items(): print(f'{cls:10s} {info["image_files"]:4d} files  | {info["status"]}')
    print(f'\nSaved source information in per-class commons_sources.csv files. Status: {output}')
    print('Review every image! Remove photos that are not of the requested object.')
    absent=[x for x in a.classes if summary[x]['image_files']==0 and 'ZIP export' not in summary[x]['status']]
    if absent:
        print('NOT FOUND / ACCESS FAILED:', ', '.join(absent))
        print('You can rerun this script to try other search results, or use authorized Roboflow exports.')
        return 2
    return 0

if __name__=='__main__':
    sys.exit(main())
