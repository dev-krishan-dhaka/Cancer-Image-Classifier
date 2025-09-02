import os
import glob
import pandas as pd
from src.infer import predict_image  # ✅ use your predict_image function
from sklearn.metrics import classification_report, confusion_matrix

def run_inference_on_folder(weights, img_size, test_dir, out_csv, model_name="resnet18"):
    results = []

    for label in ["cancerous", "non_cancerous"]:
        folder = os.path.join(test_dir, label)
        image_paths = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
        
        for img_path in image_paths:
            try:
                pred = predict_image(weights, img_path, model_name=model_name, img_size=img_size)
                results.append({
                    "file": img_path,
                    "true_label": label,
                    "pred_class": pred["pred_class"],
                    "prob_cancerous": pred["probs"]["cancerous"],
                    "prob_non_cancerous": pred["probs"]["non_cancerous"]
                })
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Inference completed! Results saved to {out_csv}")

    # 📊 Evaluation Metrics
    if not df.empty:
        y_true = df["true_label"]
        y_pred = df["pred_class"]

        print("\n📊 Evaluation Metrics:")
        print(classification_report(y_true, y_pred, target_names=["cancerous", "non_cancerous"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    else:
        print("⚠️ No results found. Please check your dataset.")

    return df


if __name__ == "__main__":
    weights = "./runs/fast_test/best.pt"   # path to your trained model
    img_size = 128
    test_dir = "./data/test"
    out_csv = "./runs/fast_test/test_predictions.csv"

    print("🚀 Running inference on test dataset...")
    run_inference_on_folder(weights, img_size, test_dir, out_csv)
