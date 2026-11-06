# Phase 2 visual review

Start the review app from the project folder. Image previews require the original local dataset.

```bat
.venv\Scripts\python.exe -m streamlit run scripts\phase2_review_app.py --server.port 8502
```

## What to review

- **Label review:** the 49 HIGH label problems first. Mark each image Correct, Incorrect or Uncertain. Add a short note for Incorrect/Uncertain.
- **Duplicate pairs:** all 36 pairs. Mark each Duplicate, Near-duplicate, Distinct or Uncertain.
- The queue also contains MEDIUM/LOW items and model disagreements for context.
- **Source follow-up:** `source_followup.csv` lists 13 unique images: 11 with no artist/credit and 3 non-PD/CC0 images with no licence URL (one image has both issues). These need verification; do not invent missing attribution.

## Where decisions go

- `results/phase2/review/decisions/label_decisions.csv`
- `results/phase2/review/decisions/duplicate_decisions.csv`

A decision is written only when you press **Save decision**. The app never marks an image reviewed by default. You can stop and resume; saved rows are loaded again on the next start.

No original photograph, dataset split, model or Experiment 1 result is changed by the review app.
