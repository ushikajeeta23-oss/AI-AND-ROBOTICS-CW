# OfficeMate AI experiment log

## Experiment 1

YOLOv8n-cls trained on 1,387 images, with 297 validation images and 297 historical test images. Training stopped at epoch 19 and the best validation result was at epoch 7. Overfitting was observed. The test metrics below remain provisional because the duplicate audit found possible leakage. Frequent confusions included book → notebook, stapler → chair, phone → notebook and notebook → book.

## Experiment 2

Transfer learning started from the local generic YOLOv8n-cls checkpoint. The audited 1,132/284 group-aware development split excluded all former test images. The CPU run used 160 px inputs, batch size 32, four Torch threads, seed 42, AdamW with cosine scheduling, weight decay 0.001, dropout 0.20, five frozen initial layers, moderate crop and color augmentation and random erasing. The submitted run recorded the generic augmentation flag, and the final validation step fell back to single-scale prediction. The reproduction config leaves that inference flag at its default. The schedule allowed up to 12 epochs and early stopping with patience 4. The grouping preserved class coverage, with 93 to 130 training images per class. 12 completed epochs, peak training-loop top-1 80.99% at epoch 9.

The separate evaluation below used the exported best checkpoint through the same prediction API as the application. Ultralytics selected epoch 12 using the mean of validation top-1 and top-5 accuracy. Training loss reached 0.023. Validation loss reached a minimum of 0.666 at epoch 5 and ended at 0.725. See `runs/experiment2/yolov8n_cls_exp2/results.csv` and `results/experiment2/training_curves.png`.

| Metric | Experiment 1 provisional test (297) | Experiment 2 development validation (284) |
|---|---:|---:|
| Accuracy | 76.09% | 80.28% |
| Macro-F1 | 76.17% | 80.00% |
| Review rate | 31.99% | 26.76% |
| Accepted accuracy | 90.59% | 91.35% |

Both review measurements use threshold 0.80. Experiment 2 accepted 208 of 284 validation predictions and flagged 76 for review. The accepted subset is smaller and selected by score. Different images, cleaning and split rules prevent the table from establishing a like-for-like improvement. The Experiment 2 validation set was used for checkpoint selection. The development split fell short of the joint 90% accuracy and macro-F1 target. Independent final test evaluation pending.

### Experiment 2 validation by class

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| bin | 88.46% | 74.19% | 80.70% | 31 |
| book | 69.70% | 76.67% | 73.02% | 30 |
| bottle | 85.00% | 73.91% | 79.07% | 23 |
| chair | 90.00% | 81.82% | 85.71% | 33 |
| keyboard | 85.19% | 88.46% | 86.79% | 26 |
| mouse | 84.00% | 72.41% | 77.78% | 29 |
| mug | 82.86% | 100.00% | 90.62% | 29 |
| notebook | 71.43% | 86.21% | 78.12% | 29 |
| phone | 66.67% | 61.54% | 64.00% | 26 |
| stapler | 82.76% | 85.71% | 84.21% | 28 |

### Most frequent validation confusions

- book → notebook in 4 validation images.
- mouse → mug in 3 validation images.
- notebook → book in 3 validation images.
- phone → book in 3 validation images.
- bin → mug in 2 validation images.
- book → phone in 2 validation images.

The confusion matrix and score histogram are in `results/experiment2/validation/`. Additional work should target underrepresented object styles, clearer book/notebook and phone/accessory labels, more varied stapler and bottle images, and calibration using separately collected data. See the independent evaluation protocol for requirements.
