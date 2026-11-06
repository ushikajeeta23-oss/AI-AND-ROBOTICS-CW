"""Generate classification evidence for a designated validation or test split."""
import argparse
import csv
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from officemate.config import CLASSES, DEFAULT_MODEL, DEFAULT_THRESHOLD, TARGET_ACCURACY
from officemate.dataset import indexed_split, audit
from officemate.features import load_image
from officemate.inference import Predictor


def evaluate(data: Path, model_path: Path, out: Path, split: str = "test"):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Results folder is not empty: {out}; choose a new path for a new experiment.')
    counts, duplicates = audit(data)
    if duplicates:
        raise ValueError(f'Exact duplication detected across splits: {duplicates[:2]}')
    if split not in ('test', 'val'):
        raise ValueError('Use test for final evidence or val for model selection.')
    if any(counts[split][k] == 0 for k in CLASSES):
        raise ValueError(f'The {split} split must contain all ten classes.')
    if model_path.suffix.lower() == ".pt":
        import torch
        torch.set_num_threads(4)
    predictor = Predictor(model_path)
    predictions = []
    actual, predicted, timings = [], [], []
    for path, label in indexed_split(data, split):
        start = time.perf_counter()
        pred = predictor.predict_rgb(load_image(path))
        timings.append((time.perf_counter() - start) * 1000)
        if pred.label not in CLASSES:
            raise ValueError(f'Unknown model class: {pred.label}')
        actual.append(label)
        predicted.append(pred.label)
        predictions.append({'path': str(path), 'true_class': label,
                            'predicted_class': pred.label, 'confidence': round(pred.confidence, 6),
                            'accepted': pred.confidence >= DEFAULT_THRESHOLD,
                            'correct': pred.label == label,
                            'processing_ms': round(timings[-1], 3)})
    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, labels=list(CLASSES), average='macro', zero_division=0)
    full_report = classification_report(actual, predicted, labels=list(CLASSES),
                                        zero_division=0, output_dict=True)
    counts_matrix_array = confusion_matrix(actual, predicted, labels=list(CLASSES))
    counts_matrix = counts_matrix_array.tolist()
    accepted = [x for x in predictions if x['accepted']]
    accepted_accuracy = (sum(x['correct'] for x in accepted) / len(accepted)) if accepted else None
    scores = [row["confidence"] for row in predictions]
    results = {'split': split, 'n_images': len(actual), 'model': str(model_path),
               'review_threshold': DEFAULT_THRESHOLD,
               'review_count': len(predictions) - len(accepted),
               'review_rate': (len(predictions) - len(accepted)) / len(predictions),
               'accepted_accuracy': accepted_accuracy,
               'mean_processing_ms_including_image_load': sum(timings)/len(timings),
               'target_test_accuracy': TARGET_ACCURACY,
               'target_met': bool(accuracy >= TARGET_ACCURACY),
               'target_accuracy_and_macro_f1': TARGET_ACCURACY,
               'target_met_on_this_split': bool(min(accuracy, macro_f1) >= TARGET_ACCURACY),
               'confidence_min': min(scores), 'confidence_mean': sum(scores) / len(scores),
               'confidence_max': max(scores),
               'accuracy': float(accuracy), 'macro_f1': float(macro_f1),
               'class_order': list(CLASSES), 'confusion_matrix': counts_matrix,
               'per_class': full_report}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (out / 'classification_report.txt').write_text(classification_report(
        actual, predicted, labels=list(CLASSES), zero_division=0), encoding='utf-8')
    with (out / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(predictions[0]))
        w.writeheader(); w.writerows(predictions)
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(confusion_matrix=counts_matrix_array, display_labels=CLASSES).plot(
        ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45, values_format='d')
    ax.set_title(f'OfficeMate AI | {split} confusion matrix')
    fig.tight_layout(); fig.savefig(out / 'confusion_matrix.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = [i / 20 for i in range(21)]
    for correct, label in ((True, "Correct"), (False, "Incorrect")):
        ax.hist([r["confidence"] for r in predictions if r["correct"] == correct],
                bins=bins, alpha=0.65, label=label)
    ax.axvline(DEFAULT_THRESHOLD, linestyle="--", color="black", label="Review threshold")
    ax.set(xlabel="Top class score", ylabel="Images", title=f"{split.capitalize()} confidence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "confidence_distribution.png", dpi=150)
    plt.close(fig)
    errors = [x for x in predictions if not x['correct']]
    by_pair = {}
    for x in errors:
        key = (x['true_class'], x['predicted_class'])
        by_pair[key] = by_pair.get(key, 0) + 1
    lines = [f'# {split.capitalize()} error analysis', '',
             f'- Images: {len(actual)}', f'- Accuracy: {accuracy:.4f}', f'- Macro-F1: {macro_f1:.4f}',
             f'- Incorrect: {len(errors)}',
             f'- Review threshold: {DEFAULT_THRESHOLD:.2f}; review count: {results["review_count"]}',
             f'- Accuracy on accepted predictions: {accepted_accuracy if accepted_accuracy is not None else "not available"}',
             '', '## Most frequent confusions']
    if not by_pair:
        lines.append('- No misclassifications in this test set. Verify generalization on new environments.')
    else:
        for (truth, guess), n in sorted(by_pair.items(), key=lambda p: -p[1])[:6]:
            lines.append(f'- {truth} -> {guess}: {n} image(s)')
    lines += ["", "## Accepted errors",
              f"- {sum(row['accepted'] for row in errors)} incorrect predictions passed the score threshold.",
              "- Per-image scores and labels are in predictions.csv. Confusion counts alone cannot establish visual causes.",
              ""]
    if split == "val":
        lines.append("Independent final test evaluation pending.")
    (out / 'error_analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'{split.upper()} accuracy={accuracy:.4f}  macro-F1={macro_f1:.4f}  N={len(actual)}')
    print(f'Wrote {out / "metrics.json"}, confusion_matrix.png, predictions.csv and error_analysis.md')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/office_dataset'))
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--out', type=Path, default=Path('results/heldout_test'))
    a = p.parse_args(); evaluate(a.data, a.model, a.out, split=a.split)

if __name__ == '__main__':
    main()
