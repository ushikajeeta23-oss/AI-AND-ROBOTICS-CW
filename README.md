# OfficeMate AI
### PDE3802 – Artificial Intelligence in Robotics | Assessment 1

## 1. Project Overview

OfficeMate AI is a Python-based office-item classification system developed for the PDE3802 Artificial Intelligence in Robotics module at Middlesex University Mauritius.

The project proposes an autonomous robot that could identify office objects and assist with organizing suitable items. For Part B of the assessment, we implemented an image-classification application using YOLOv8n-cls and Streamlit.

The application accepts photographs or camera input and displays the predicted object category, confidence score and recommended handling decision.

The implemented system focuses on object recognition. Physical navigation and robotic manipulation are part of the proposed system design and have not been implemented.

## 2. Recognized Objects

The system recognizes ten categories:

1. Bin
2. Book
3. Bottle
4. Chair
5. Keyboard
6. Mouse
7. Mug
8. Notebook
9. Phone
10. Stapler

Each image is expected to contain one main office object.

## 3. Technologies Used

- **Python 3.11:** Main programming language
- **YOLOv8n-cls:** Image classification model
- **PyTorch:** Deep learning framework
- **Ultralytics:** Model training and inference
- **OpenCV:** Image processing
- **Streamlit:** User interface
- **NumPy:** Numerical operations
- **scikit-learn:** Evaluation metrics
- **Matplotlib:** Visualizations and confusion matrices
- **Git and GitHub:** Version control

## 4. Project Structure

```text
OfficeMate_AI_Report_Aligned/
│
├── app.py
├── predict.py
├── evaluate.py
├── train_experiment2.py
├── phase2_build_experiment2.py
├── restore_experiment2.py
├── experiment2_yolov8n_cls.yaml
├── RUN_APP.bat
│
├── README.md
├── DATASET_CARD.md
├── EXPERIMENT_LOG.md
├── REPORT_ALIGNMENT.md
├── INDEPENDENT_TEST_PROTOCOL.md
│
├── requirements.txt
├── requirements-yolo.txt
├── requirements-ui.txt
├── pyproject.toml
│
├── data/
├── models/
├── results/
├── runs/
├── scripts/
├── src/
└── tests/
```

## 5. Installation

### Requirements

- Windows 10 or Windows 11
- Python 3.11
- Internet connection for the initial dependency installation
- Sufficient storage for the required libraries and model files

### Step 1: Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/ushikajeeta23-oss/AI-AND-ROBOTICS-CW.git
```

Open the cloned project folder.

### Step 2: Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### Step 3: Install dependencies

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt -r requirements-ui.txt
```

Then install the project package:

```powershell
.venv\Scripts\python.exe -m pip install -e .
```

## 6. Running the Application

### Method 1: Windows launcher

Double-click:

```text
RUN_APP.bat
```

### Method 2: Terminal

From the project folder, run:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address displayed in the terminal, normally:

```text
http://localhost:8501
```

The application will open in your browser.

## 7. Using OfficeMate AI

The application provides two input options.

**Image upload**

Users can upload one or more photographs containing office objects.

**Camera input**

Users can capture an image using a connected computer camera.

After receiving an image, the application:

1. Processes the input image.
2. Classifies the main object using YOLOv8n-cls.
3. Displays the predicted object category.
4. Displays the confidence score.
5. Checks the prediction against the confidence threshold.
6. Displays the recommended handling decision.

### Example output

For a photograph of a stapler, the application may display:

```text
Predicted Class: Stapler

Confidence: 100.0%

Policy: CHECK_BEFORE_LIFT

Suggested Destination: Approved stationery tray
```

This example is based on a recorded demonstration. Actual predictions depend on the input image and selected model.

## 8. Confidence and Handling Decisions

The application uses a confidence threshold of **0.80 (80%)**.

Predictions below this threshold are flagged for human review.

A confidence score represents the model's score for an individual prediction. It does not represent the overall accuracy of the classifier.

The handling decisions are:

| Object | Policy |
|---|---|
| Bin | REPORT_ONLY |
| Book | REPORT_ONLY |
| Bottle | CHECK_BEFORE_LIFT |
| Chair | REPORT_ONLY |
| Keyboard | REPORT_ONLY |
| Mouse | CHECK_BEFORE_LIFT |
| Mug | REPORT_ONLY |
| Notebook | REPORT_ONLY |
| Phone | REPORT_ONLY |
| Stapler | CHECK_BEFORE_LIFT |

The three conditionally handleable objects are bottles, computer mice and staplers.

However, additional checks are required before physical handling could be considered.

For example:

- A bottle must be an approved empty plastic bottle.
- A computer mouse must be unplugged and authorized for handling.
- A stapler must be small and suitable for the proposed gripper.
- The object must satisfy the proposed weight, grip-width, reach and safety restrictions.

The application does not physically lift objects or verify these conditions.

## 9. Dataset

The initial dataset contained **1,981 real photographs** obtained from Wikimedia Commons.

The images cover all ten classes.

### Experiment 1

| Dataset | Images |
|---|---:|
| Training | 1,387 |
| Validation | 297 |
| Testing | 297 |
| **Total** | **1,981** |

During evaluation, we identified incorrect labels and possible near-duplicate images across dataset splits.

These findings affected the reliability of the first experiment's test results.

### Experiment 2

A revised dataset was prepared after reviewing the original photographs.

The cleaning process excluded 42 confirmed incorrect images and held out another 523 images. Suspected near-duplicate groups were also considered during dataset preparation.

| Dataset | Images |
|---|---:|
| Training | 1,132 |
| Validation | 284 |
| **Total** | **1,416** |

The original Experiment 1 files were preserved.

The source photographs are not included directly in this GitHub repository. The dataset card contains the source information, acquisition procedure and relevant limitations.

See [DATASET_CARD.md](DATASET_CARD.md).

## 10. Model Training

The classification model used in both experiments was YOLOv8n-cls.

Transfer learning was used to adapt pretrained model weights to the ten office-item categories.

Experiment 1 was configured for up to 45 epochs and stopped early after 19 epochs.

Experiment 2 was trained for 12 epochs using CPU processing and 160 × 160 image inputs.

The models and experiment results were stored separately to preserve the original implementation.

The Experiment 2 training configuration is available in:

```text
experiment2_yolov8n_cls.yaml
```

The training script is:

```text
train_experiment2.py
```

## 11. Evaluation Results

Model performance was evaluated using classification accuracy, macro-F1 and confusion matrices.

### Results

| Metric | Experiment 1 | Experiment 2 |
|---|---:|---:|
| Accuracy | 76.09% | 80.28% |
| Macro-F1 | 76.17% | 80.00% |
| Human-review rate | 31.99% | 26.76% |
| Accepted-prediction accuracy | 90.59% | 91.35% |

**Important:** Experiment 1's figures are provisional test results because possible duplicate leakage was identified.

Experiment 2's figures are development validation results, not independent final-test results.

The experiments used different datasets, so their figures should not be interpreted as a directly controlled comparison.

The original target of 90% accuracy and macro-F1 was not achieved.

### Error Analysis

Experiment 1 showed several classification difficulties:

- Books were sometimes classified as notebooks.
- Staplers were occasionally classified as chairs.
- Phones were sometimes confused with notebooks.
- Some downloaded images did not represent their assigned physical office objects.

These errors were investigated during the dataset audit.

Experiment 2 used a revised dataset intended to reduce incorrect labels and limit duplicate leakage.

Although the second experiment produced higher reported validation metrics, a separate independent test is still required to establish its generalization performance.

Evaluation evidence, including confusion matrices and prediction results, is available in the `results` folder.

## 12. Reproducing Results

To run the available automated tests:

```powershell
.venv\Scripts\python.exe -m pytest
```

The finalized project passed **25 automated tests**.

To preview the Experiment 2 dataset preparation:

```powershell
.venv\Scripts\python.exe phase2_build_experiment2.py --dry-run
```

To repeat Experiment 2 validation, once its dataset has been restored:

```powershell
.venv\Scripts\python.exe evaluate.py --data data/experiment2 --split val --model models/experiment2/officemate_yolov8n_cls.pt --out results/validation_repeat
```

Use a new output directory if the specified results folder already exists.

The dataset and experiment documentation should be consulted before attempting to reproduce training or evaluation.

## 13. Troubleshooting

**Python is not recognized**

Confirm that Python 3.11 is installed.

Check:

```powershell
py -3.11 --version
```

**ModuleNotFoundError**

Install the project dependencies using the commands in Section 5.

**Streamlit does not open**

Start the application from the terminal and check whether it displays an error.

Confirm that port 8501 is not already being used.

**Model file not found**

Check that the required `.pt` model files are present in the `models` directory.

**Photographs are missing**

The complete source photographs are not bundled with the GitHub repository.

Refer to `DATASET_CARD.md` for the acquisition procedure and dataset requirements.

**Predictions are incorrect**

Try a clearer image containing one main object. Avoid excessive clutter, poor lighting and heavily cropped objects.

The model is not guaranteed to recognize every office item correctly.

## 14. Limitations

The main limitations of the current project are:

- The model does not achieve the original 90% performance target.
- Experiment 1 has possible dataset leakage.
- Some images remain unresolved following the dataset audit.
- An independent final evaluation of Experiment 2 has not yet been completed.
- The classifier processes images rather than detecting individual objects in larger scenes.
- The system does not estimate verified 3D object positions.
- Navigation and robotic grasping have not been physically implemented.

The current application demonstrates object recognition, confidence assessment and handling recommendations.

## 15. Future Improvements

Possible future developments include:

- Collecting more independently photographed office objects.
- Improving recognition accuracy for difficult categories.
- Performing independent final testing.
- Adding object detection and localization.
- Integrating depth-camera measurements.
- Implementing autonomous navigation.
- Connecting the recognition system to a physical robotic arm.
- Testing real-world grasping and object placement.

## 16. References and Additional Documentation

The project uses the Ultralytics YOLO framework and Python computer-vision libraries.

Relevant documentation:

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Wikimedia Commons](https://commons.wikimedia.org/)

Additional project files:

- `DATASET_CARD.md` — Dataset sources, preparation and limitations.
- `EXPERIMENT_LOG.md` — Model training and evaluation history.
- `REPORT_ALIGNMENT.md` — Relationship between the project implementation and the assessment requirements.
- `INDEPENDENT_TEST_PROTOCOL.md` — Procedure for independent final testing.

## 17. Project Information

**Project:** OfficeMate AI  
**Module:** PDE3802 – Artificial Intelligence in Robotics  
**Assessment:** Assessment 1  
**University:** Middlesex University Mauritius  
**Academic Year:** 2026–2027

**GitHub Repository:**  
https://github.com/ushikajeeta23-oss/AI-AND-ROBOTICS-CW

This project was developed for academic coursework. It demonstrates AI-based classification of office objects and proposes how recognition could support a future autonomous office-organizing robot.