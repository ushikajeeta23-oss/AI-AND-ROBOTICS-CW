# Independent office-photo evaluation protocol

Independent final test evaluation pending.

1. Freeze checkpoint hashes, preprocessing and the 0.80 review threshold before collecting or inspecting final outcomes. If comparing models, freeze both and evaluate on the same sealed set.
2. Collect at least 30 new examples per class, preferably 50, with variation in offices, users, backgrounds, cameras, light, framing and object style. Use distinct physical objects where practical. Define books as bound reading material and notebooks as paper writing pads or books. Establish explicit criteria for phone accessories, nonportable bottle lookalikes and other ambiguous items before collection.
3. Record the physical object and session IDs, category, capture conditions and permission. Keep bursts or repeat angles with the same object. Avoid personal information. Have a second reviewer resolve labels against the written criteria without seeing predictions.
4. Screen against both development pools and the historical image catalog by source ID, SHA-256, pHash and dHash. Inspect near matches and capture groups. Record exclusions. Keep the accepted evaluation files and labels sealed until checkpoint selection is over.
5. Run the frozen evaluator once per model with the final set under `test/<class>`. Use a fresh output folder. Report sample size, class support, accuracy, macro-F1, per-class scores, confusion, review rate and accepted accuracy. Use paired model comparisons and 95% intervals with bootstrap resampling by physical object or capture group when there are repeated views. Report the number and accuracy of accepted predictions together.
6. Keep out-of-scope or multi-object inputs in a separate challenge set. Investigate high-confidence errors and poor recall. Collect a fresh sealed set if these results are later used to change the model or threshold.

With a frozen test set and a fresh output directory, the existing evaluator can score the checkpoint as follows.

```powershell
.venv\Scripts\python.exe evaluate.py --data data/independent_office --split test --model models/experiment2/officemate_yolov8n_cls.pt --out results/independent_office_exp2
```
