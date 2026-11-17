Work on digital pathology. The objective is to investigate, in a reproducible manner, how choices regarding preprocessing, representation, classification, aggregation, and validation affect the robustness of the system.

- classification
- clustering

As a starting point, cluster by colors and then apply a CNN

### 2. Initial glossary

|Term|Brief description|
|---|---|
|**Baseline**|Simple and reproducible implementation used as a point of comparison.|
|**Calibration**|Correspondence between the confidence stated by the model and how often it is correct.|
|**Domain**|Data acquisition context, such as laboratory, scanner, or staining protocol.|
|**H&E**|Hematoxylin and eosin staining, common in histopathological images.|
|**Digital slide / WSI**|Digital image of an entire slide (_Whole Slide Image_), potentially gigapixels in size.|
|**Domain shift**|Change in the image distribution between training and use, caused by origin, staining, or acquisition.|
|**Patch / tile**|Smaller crop extracted from an image or slide, processable individually.|
|**Patient/slide split**|Separation that prevents images from the same patient or slide from appearing in both training and testing.|
|**External validation**|Evaluation on data collected from a different source than the one used for training.|

### 3. Learning objectives

- Apply image processing operations to a real problem, justifying the choices made.
- Analyze color, illumination, texture, noise, resolution, focus, and acquisition artifacts.
- Build and compare [image classification](https://www.dcc.ufrrj.br/moodle/mod/resource/view.php?id=17638 "Classificação de Imagens") baselines.
- Organize experiments with proper separation between development and evaluation.
- Measure predictive quality, uncertainty, computational cost, and robustness.
- Communicate limitations, biases, and risks of a computer vision application in healthcare.

## 4. Environment

- **Python 3.13** is required. TensorFlow publishes no wheels for 3.14+, so the
  project is pinned in `.python-version` and `requires-python`.
- Dependencies: `requirements.txt` (or `uv sync`, which reads `pyproject.toml`).

```bash
uv sync --python 3.13          # creates .venv and installs everything
uv run pytest -q               # smoke tests (synthetic tiles, no dataset needed)
```

## 5. Running the pipeline

All commands read the official manifests (`train.csv`, `validation.csv`,
`robustness.csv`, `labels.csv`) and the image folder distributed by the
discipline. Seed and configuration live in `configs/baseline.yaml`.

```bash
# 1. check environment, versions and config
uv run pathoshift describe

# 2. validate manifests and write normalised copies
uv run pathoshift prepare \
  --train data/train.csv --validation data/validation.csv \
  --robustness data/robustness.csv --labels data/labels.csv \
  --images-root data/images --output outputs/manifests

# 3. exploratory analysis: classes, dimensions, color, focus, background
uv run pathoshift inspect \
  --manifest data/validation.csv --images-root data/images --output outputs/eda

# 4. train the classical baseline (colour/HSV descriptors + linear model)
uv run pathoshift train \
  --manifest data/train.csv --images-root data/images --output outputs/baseline.joblib

# 5. predict validation (writes the submission contract)
uv run pathoshift predict \
  --manifest data/validation.csv --images-root data/images \
  --model outputs/baseline.joblib --output predictions_validation.csv

# 6. robustness: applies robustness.csv perturbations, then predicts
uv run pathoshift robustness \
  --manifest data/validation.csv --robustness data/robustness.csv \
  --images-root data/images --model outputs/baseline.joblib \
  --output predictions_robustness.csv

# 7. local evaluator: F1-macro, per-class F1, confusion matrix, error gallery
uv run pathoshift evaluate \
  --manifest data/validation.csv --predictions predictions_validation.csv \
  --robustness-predictions predictions_robustness.csv \
  --images-root data/images --output outputs/metrics.json
```

`predict`/`robustness` always write exactly `image_id,predicted_label,confidence`
with `confidence` in `[0, 1]`, and refuse to write when ids, labels or the
confidence range break the contract.

### Research question

Does stain/colour normalisation before the classifier reduce the macro-F1 drop
under the colour/brightness perturbation of `robustness.csv`?

### Status

- [x] Manifest ingestion, validation and the submission contract
- [x] Deterministic perturbations (colour/brightness, JPEG, blur, noise)
- [x] Local evaluator (accuracy, macro F1, per-class F1, confusion matrix)
- [x] Classical baseline (HSV histograms + channel statistics + linear model)
- [ ] Colour/stain normalisation proposal
- [ ] Lightweight CNN in TensorFlow
- [ ] Controlled comparison and validation-versus-robustness analysis
