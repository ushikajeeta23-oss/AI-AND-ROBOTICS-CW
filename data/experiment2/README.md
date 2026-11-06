# OfficeMate AI — Experiment 2 working dataset

This folder is a separate working dataset built from saved Phase 2 review decisions.
The original dataset remains at `data/office_dataset` and was not modified.

- Train: 1132 images
- Validation: 284 images
- Final test: not created here. A genuinely new independent test set must be collected separately.
- Retained images: 1416
- Excluded as Incorrect: 42
- Held out as Uncertain: 4
- Held out because a label flag is unreviewed: 246

| Class | Train | Val |
| --- | --- | --- |
| bin | 122 | 31 |
| book | 117 | 30 |
| bottle | 93 | 23 |
| chair | 130 | 33 |
| keyboard | 108 | 26 |
| mouse | 115 | 29 |
| mug | 116 | 29 |
| notebook | 116 | 29 |
| phone | 100 | 26 |
| stapler | 115 | 28 |

The complete row-level record is in `manifest.csv`. Excluded and held-out images
remain in the original dataset and are listed under `results/experiment2/dataset/`.
Machine-detected near-duplicate clusters are kept in one split; see
`experiment2_duplicate_groups.csv`.

This split is for development. The former test photos remain outside it.
