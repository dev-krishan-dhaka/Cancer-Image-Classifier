# 🩺 Cancer Image Classification

A deep learning project to classify medical images as **cancerous** or **non-cancerous** using **PyTorch + ResNet**.  
Includes a **Streamlit UI** for interactive demo and a **Flask API** for integration.

---

## 🚀 Features
- Train CNN models (ResNet18/34/50) with transfer learning.
- Evaluate with accuracy, precision, recall, F1-score, confusion matrix.
- Single-image inference (`src/infer.py`).
- Batch inference on test set → CSV report.
- Streamlit app for interactive demo.
- Flask REST API for deployment.
- Ready for deployment (Streamlit Cloud, Hugging Face Spaces, Docker).

---

## 📦 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/cancer-image-classifier.git
   cd cancer-image-classifier
   ```

2. **Create & activate virtual environment** (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   (Linux/Mac):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Dataset Preparation

Organize your dataset like this:

```
data/
  train/
    cancerous/
    non_cancerous/
  val/
    cancerous/
    non_cancerous/
  test/
    cancerous/
    non_cancerous/
```

- Place your images in the appropriate folders.
- Each folder should contain `.jpg` or `.png` images.

---

## 🏋️ Training

Example command:

```bash
python -m src.train   --data_dir ./data   --epochs 20   --batch_size 32   --lr 3e-4   --model resnet18   --img_size 224   --out_dir ./runs/exp1
```

This will train the model and save checkpoints to `./runs/exp1`.

---

## 🔍 Inference

### Single Image

```bash
python -m src.infer --weights ./runs/exp1/best.pt --image ./data/test/cancerous/sample.jpg
```

### Batch Inference

```bash
python -m src.infer_folder
```

Outputs: `test_predictions.csv` with predictions for all test images.

---

## 🎨 Visualization

- Confusion Matrix Heatmap
- Per-class Accuracy Bar Chart
- Misclassified Images Gallery


---

## 🌐 Deployment

### Streamlit App

```bash
streamlit run app_streamlit.py -- --weights ./runs/exp1/best.pt --model resnet18 --img_size 224

Live Demo--> https://cancer-image-classifier-idreqqnckjp4jhtldyzb4v.streamlit.app/
```


## 📊 Results

- Accuracy: ~92% (ResNet18, 20 epochs, data augmentation)
- Precision/Recall: Balanced
- ROC-AUC: 0.94

---

## 🚧 Next Steps

- Use larger models (ResNet50, EfficientNet)
- Add more data augmentation
- Add Grad-CAM for explainable AI
- Deploy API with Docker + Cloud

---

