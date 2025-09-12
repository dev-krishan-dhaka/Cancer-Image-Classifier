# 🩺 Cancer Image Classification

A deep learning project to classify medical images as **cancerous** or **non-cancerous** using **PyTorch + ResNet**.  
The project also includes **Streamlit UI** for demo and a **Flask API** for integration.

---

## 🚀 Features
- Train CNN models (ResNet18/34/50) with transfer learning.
- Evaluate with accuracy, precision, recall, F1-score, confusion matrix.
- Single-image inference (`src/infer.py`).
- Batch inference on test set → CSV report.
- Streamlit app for interactive demo.
- Flask REST API for deployment.
- Deployment ready (Streamlit Cloud, Hugging Face Spaces, Docker).

---

## 📂 Project Structure
```
  ├── data/ # dataset (train/val/test split)
├── runs/ # trained model checkpoints + results
├── src/
│ ├── train.py # training script
│ ├── model.py # model creation
│ ├── infer.py # single image inference
│ └── utils.py # helper functions
├── app_streamlit.py # Streamlit web app
├── app_flask.py # Flask API
├── requirements.txt # dependencies
└── README.md # project description
```

## 🏋️ Training
Example command:
  `bash
python -m src.train 
  --data_dir ./data \
  --epochs 20 \
  --batch_size 32 \
  --lr 3e-4 \
  --model resnet18 \
  --img_size 224 \
  --out_dir ./runs/exp1

  
## 🔍 Inference
Single Image
bash
Copy code
python -m src.infer --weights ./runs/exp1/best.pt --image ./data/test/sample.png
Batch Inference
bash
Copy code
python batch_infer.py
Outputs: test_predictions.csv

## 🎨 Visualization
Confusion Matrix

Per-class accuracy

Misclassified images

Streamlit app provides UI to upload images and view predictions.

## 🌐 Deployment
Streamlit
bash
Copy code
streamlit run app_streamlit.py -- --weights ./runs/exp1/best.pt
Flask
bash
Copy code
python app_flask.py --weights ./runs/exp1/best.pt
## 📊 Results
Accuracy: ~92% (ResNet18, 20 epochs, data augmentation)

Precision/Recall: Balanced, good separation

ROC-AUC: 0.94

## 🚧 Next Steps
Use larger models (ResNet50, EfficientNet).

More data augmentation.

Grad-CAM for explainable AI.

Deploy API with Docker + Cloud.

## ⚠️ Disclaimer
This is a research demo. Not for medical diagnosis. Always consult professionals.
