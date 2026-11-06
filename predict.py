"""Predict from a single image, folder of images or standard webcam.

Saves labelled preview images, predictions.csv and (optionally) camera screenshots.
"""
import argparse
import csv
from pathlib import Path
import cv2
from PIL import Image
from officemate.config import IMAGE_EXTENSIONS, DEFAULT_THRESHOLD, DEFAULT_MODEL
from officemate.features import load_image
from officemate.inference import Predictor
from officemate.policy import decide


def overlay(bgr, prediction, decision):
    canvas = bgr.copy()
    message = f'{prediction.label} | {prediction.confidence:.1%} | {decision.action}'
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 850), 45), (20, 30, 40), -1)
    cv2.putText(canvas, message, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                .63, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def webcam(model, threshold, camera, save_screenshots: Path | None):
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)  # DirectShow on Windows
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera)  # Standard OpenCV fallback
    if not cap.isOpened():
        raise RuntimeError('Camera unavailable. Close Teams/Zoom; grant Windows Camera permission or try --camera 1.')
    print('Place ONE main object in the frame; press S to save a labelled frame or Q to quit.')
    if save_screenshots:
        save_screenshots.mkdir(parents=True, exist_ok=True)
    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera failed to return a frame.')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred = model.predict_rgb(rgb)
            decision = decide(pred.label, pred.confidence, threshold)
            drawn = overlay(frame, pred, decision)
            cv2.imshow('OfficeMate AI | S save | Q quit', drawn)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                destination = (save_screenshots or Path('results/camera')) / f'camera_{saved:03d}_{pred.label}.jpg'
                destination.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(destination), drawn)
                print('Saved:', destination)
                saved += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_files(model, paths, threshold: float, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    pictures = out / 'labelled_images'
    pictures.mkdir(exist_ok=True)
    for i, file in enumerate(paths):
        try:
            rgb = load_image(file)
            pred = model.predict_rgb(rgb)
            decision = decide(pred.label, pred.confidence, threshold)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            annotated = overlay(bgr, pred, decision)
            img_output = pictures / f'{i:04d}_{file.stem[:40]}.jpg'
            cv2.imwrite(str(img_output), annotated)
            rows.append({'file': str(file), 'predicted_class': pred.label,
                         'model_score': round(pred.confidence, 5),
                         'decision': decision.action,
                         'suggested_destination': decision.destination,
                         'annotated_image': str(img_output)})
            print(f'{file.name}: {pred.label} ({pred.confidence:.1%}), {decision.action}')
        except (OSError, ValueError) as exc:
            print(f'ERROR for {file}: {exc}')
    if not rows:
        raise RuntimeError('No images were classified successfully.')
    report_path = out / 'predictions.csv'
    with report_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('Predictions saved:', report_path)
    print('Annotated previews:', pictures)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=Path)
    group.add_argument('--folder', type=Path)
    group.add_argument('--camera', type=int, nargs='?', const=0)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    p.add_argument('--out', type=Path, default=Path('results/predictions'))
    args = p.parse_args()
    if not 0 <= args.threshold <= 1:
        p.error('--threshold must be between 0 and 1.')
    model = Predictor(args.model)
    if args.camera is not None:
        webcam(model, args.threshold, args.camera, args.out / 'camera_frames')
    else:
        paths = [args.image] if args.image else sorted(p for p in args.folder.rglob('*')
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            p.error('No JPEG/PNG/WEBP images were found.')
        run_files(model, paths, args.threshold, args.out)


if __name__ == '__main__':
    main()
