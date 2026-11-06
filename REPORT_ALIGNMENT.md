# PDE3802 report alignment

| Design topic | Implemented evidence | Boundary |
|---|---|---|
| Office item recognition | Ten-class YOLOv8n-cls, Streamlit, CLI and CSV | A single foreground class per RGB image |
| Handling recommendations | `src/officemate/policy.py` and score review at 0.80 | Separate human authorization and measurements |
| Eligible items | Approved empty plastic bottle, approved unplugged computer mouse and approved small stapler | Measured mass ≤200 g, grip width 25–65 mm and reach ≤300 mm, plus independent route, safety and grip checks |
| Remaining seven classes | Report-only for predictions passing the threshold | Lower scores route to human review |
| Camera and pose | Uploads, multi-upload and camera snapshots | No depth or 3D grasp-pose calculation |
| Robotic execution | Part A proposes stations, travel, arm and grasping | Part B provides no ROS, motor commands, navigation or physical handling evidence |
| Dataset | Original Commons dataset and separate Experiment 2 development split | Historical test is provisional, and development validation is used for checkpoint selection |
| Evaluation | Accuracy, macro-F1, per-class scores, confusion matrix, confidence distribution and selective metrics | The 90% accuracy and macro-F1 target must be assessed with new independent photos |

See [Experiment log](EXPERIMENT_LOG.md) for measured scores and [Independent evaluation protocol](INDEPENDENT_TEST_PROTOCOL.md) for the collection plan. Group contributions and any separate Part A submission claims require evidence from the students.
