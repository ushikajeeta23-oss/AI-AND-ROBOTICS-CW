# OfficeMate AI dataset card

The original corpus consists of 1,981 Wikimedia Commons images assigned to bin, book, bottle, chair, keyboard, mouse, mug, notebook, phone or stapler. Experiment 1 used 1,387 training, 297 validation and 297 test images. Historical test metrics are provisional because the audit found potential cross-split duplicates.

## Experiment 2 development split

The development split retains 1,416 images from the original training and validation pools. All ten classes are represented. The table reports train and validation counts.

| Class | Train | Validation |
|---|---:|---:|
| Bin | 122 | 31 |
| Book | 117 | 30 |
| Bottle | 93 | 23 |
| Chair | 130 | 33 |
| Keyboard | 108 | 26 |
| Mouse | 115 | 29 |
| Mug | 116 | 29 |
| Notebook | 116 | 29 |
| Phone | 100 | 26 |
| Stapler | 115 | 28 |
| **Total** | **1,132** | **284** |

The saved decisions contain 42 Incorrect, 4 Uncertain and 3 Correct. The 42 Incorrect images were excluded. The 523 holdouts consist of 246 unresolved label flags, 4 Uncertain labels, 3 Correct decisions needing a second look, 6 source checksum discrepancies and 264 otherwise eligible former test photos. The other 33 former test photos are covered by the label disposition categories. All 297 former test photos remain outside Experiment 2. The three Correct decisions refer to an optical-mouse icon, a mouse spider and a mug shot. Their original decisions remain in the review CSV. Retained images had no outstanding label-review flag and remain unreviewed.

The builder checks the class folders, source SHA-256 and image decoding. It combines Phase 1 clusters and pair evidence, including available feature matches, with a new pHash/dHash screen on the eligible images in both orientations. It uses SSIM to corroborate borderline candidates and checks across labels. Components connecting different labels are quarantined. Thirteen suspected groups have multiple retained members. There are zero manually confirmed duplicate groups and 36 unresolved pair decisions. The independent disk check found zero exact copies and zero recorded duplicate groups or edges spanning the new split. Perceptual screening can miss crops, large viewpoint changes and distinct photos of the same physical item.

## Source and attribution

`results/experiment2/dataset/source_manifest.csv` maps all 1,981 entries to their Commons file page, recorded asset URL, downloaded checksum, development disposition, creator or credit, and available licence details. The original API audit found 11 entries with neither artist nor credit and three attribution-licensed entries without a recorded licence URL. Source-page follow-up added named credits where available and retained notes on incomplete provenance and non-standard terms. The Missouri Historical Society mug file records custom attribution permission on its Commons page, with no standalone licence URL. The chair file's exact terms remain unresolved. The mouse diagram remains flagged for its non-standard reuse terms. Public-domain status and relevant source credits should be checked before redistribution. No source photos or training image montages are included in the submission. [Mug source record](https://commons.wikimedia.org/wiki/File:White_Ceramic_Coffee_Mug_With_Color_Transfer_Image_of_Palace_of_Manufactures.jpg)

## Reproduce the submitted split

Use the supplied `data/experiment2/manifest.csv` and run the command below from the project root after installing the dependencies. Use a real project contact address in the User-Agent.

```powershell
.venv\Scripts\python.exe restore_experiment2.py --contact your-email@example.edu
```

The restore script downloads only the retained development assets, checks every checksum and refuses changed bytes. Use a local archived original if a recorded URL is no longer available. An alternative image creates a different dataset version and requires a new manifest and metrics. The original acquisition and import scripts are supplied as provenance for the broader corpus.
