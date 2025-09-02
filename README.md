
# Cancer Detector (Binary: cancerous vs. non-cancerous)

A starter end‑to‑end project to train a CNN on medical images and serve predictions via **Streamlit** or **Flask**.

> ⚠️ This is a **template**. For real clinical use, you must validate with proper datasets, follow regulatory guidelines, and involve qualified medical professionals.

## Features
- PyTorch + torchvision transfer learning (ResNet18 by default).
- Binary classification: `cancerous` vs `non_cancerous` (feel free to rename).
- Pandas/Numpy for logging and metrics tables.
- Streamlit app for drag‑and‑drop image prediction.
- Optional Flask REST API for programmatic inference.
- Clean training script: checkpointing, early stopping, mixed precision, and simple metrics CSV.

## Folder structure
```
cancer_detector/
├─ app_streamlit.py
├─ app_flask.py
├─ requirements.txt
├─ README.md
├─ src/
│  ├─ config.py
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  ├─ infer.py
│  └─ utils.py
└─ data/               # Put your images here (see below)
```

## Expected data layout
Place images into train/val/test splits with class subfolders:
```
data/
├─ train/
│  ├─ cancerous/
│  └─ non_cancerous/
├─ val/
│  ├─ cancerous/
│  └─ non_cancerous/
└─ test/
   ├─ cancerous/
   └─ non_cancerous/
```
Accepted formats: png, jpg, jpeg, bmp, tiff.

## Quickstart
### 1) Create a virtual env and install deps
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train
```bash
python -m src.train   --data_dir ./data   --epochs 10   --batch_size 32   --lr 3e-4   --model resnet18   --img_size 224   --out_dir ./runs/exp1
```
This will save best model to `runs/exp1/best.pt` and metrics to `runs/exp1/metrics.csv`.

### 3) Streamlit app (UI)
```bash
streamlit run app_streamlit.py -- --weights ./runs/exp1/best.pt --model resnet18 --img_size 224
```
Upload an image; the app will output **Cancerous / Non‑cancerous** with probability.

### 4) Flask API (optional)
```bash
python app_flask.py --weights ./runs/exp1/best.pt --model resnet18 --img_size 224 --host 0.0.0.0 --port 8000
```
Then POST an image file:
```bash
curl -X POST -F "file=@/path/to/image.png" http://localhost:8000/predict
```

## Notes
- You can switch backbones: `resnet18`, `resnet34`, `resnet50`.
- If your dataset is imbalanced, enable `--class_weights auto` in training.
- For grayscale medical images, the loader auto‑expands to 3 channels.
- For DICOM, convert to PNG/JPG first or extend `dataset.py` to read DICOM (pydicom).

## ⚕️ Disclaimer
This software is **not** a medical device. Predictions are for research/education only and **must not** be used for diagnosis or treatment.
