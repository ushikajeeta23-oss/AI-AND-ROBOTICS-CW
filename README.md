# OfficeMate AI

OfficeMate AI is a Python application for the PDE3802 assessment. It uses Ultralytics YOLOv8n-cls to identify one office object per image and Streamlit to show the prediction, confidence score, human-review status and handling recommendation.

The ten categories are bin, book, bottle, chair, keyboard, mouse, mug, notebook, phone and stapler.

## Run the app

Open `RUN_APP.bat` from this folder. It uses the existing Python 3.11 environment and local model files. The sidebar lets you select either experiment and choose image upload or camera input. The Model performance tab shows the recorded metrics and confusion matrix.

On a fresh clone, set up Python 3.11 once before using the launcher.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt -r requirements-ui.txt
.venv\Scripts\python.exe -m pip install -e .
```

Source photographs stay outside the GitHub repository. See [Dataset card](DATASET_CARD.md) for acquisition instructions. Dataset preparation needs the original local dataset, and repeat validation needs the restored Experiment 2 images.

The default review threshold is 0.80. Predictions below it require human review. For scores at or above the threshold, only approved empty plastic bottles, approved unplugged computer mice and approved small staplers are candidates for handling. The other seven categories are report-only. Physical handling requires separate authorization and checks for weight, grip width, reach and safety.

## Dataset and training

Experiment 1 used 1,981 Wikimedia Commons images split into 1,387 for training, 297 for validation and 297 for testing. The duplicate audit found possible overlap between splits.

Experiment 2 excluded 42 incorrectly labelled images and held out 523 others. It kept the former test photos outside development and assigned suspected duplicates to the same split. The resulting dataset had 1,132 training and 284 validation images. YOLOv8n-cls was fine-tuned from local pretrained weights for 12 epochs on CPU using 160-pixel inputs. See [Dataset card](DATASET_CARD.md) and [Experiment log](EXPERIMENT_LOG.md).

| Measure | Experiment 1 provisional test | Experiment 2 development validation |
|---|---:|---:|
| Accuracy | 76.09% | 80.28% |
| Macro-F1 | 76.17% | 80.00% |
| Human-review rate at 0.80 | 31.99% | 26.76% |
| Accuracy among accepted predictions | 90.59% | 91.35% |

The experiments used different splits and cannot establish a direct improvement. Experiment 2 fell short of the 90% accuracy and macro-F1 target. Its confidence scores are uncalibrated, and a high score can still accompany an incorrect prediction.

## Reproduce the assessment

Run the following commands from the project root to preview dataset preparation, evaluate the saved Experiment 2 checkpoint in a fresh folder, or run the tests.

```powershell
.venv\Scripts\python.exe phase2_build_experiment2.py --dry-run
.venv\Scripts\python.exe evaluate.py --data data/experiment2 --split val --model models/experiment2/officemate_yolov8n_cls.pt --out results/validation_repeat
.venv\Scripts\python.exe -m pytest
```

The training recipe is in `experiment2_yolov8n_cls.yaml` and the entry point is `train_experiment2.py`. The saved run arguments and checkpoints are under `runs/`. Supporting source and review tools are under `scripts/`. Model provenance and the upstream licence are recorded in `ULTRALYTICS_LICENSE.txt` and the experiment files.

## Limitations

The project implements classification and policy display. The proposed robot's navigation, lifting and 3D grasp verification remain outside the implementation. Independent final test evaluation pending. The [independent test protocol](INDEPENDENT_TEST_PROTOCOL.md) describes how to collect and evaluate a separate set of office photographs.
