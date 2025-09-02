
import argparse, io, torch, streamlit as st
from PIL import Image
from src.infer import load_model, build_preprocess

def load(weights, model_name, device):
    model, class_names = load_model(weights, model_name, device)
    preprocess = build_preprocess()
    return model, class_names, preprocess

def main():
    sp = argparse.ArgumentParser(add_help=False)
    sp.add_argument("--weights", required=True)
    sp.add_argument("--model", default="resnet18")
    sp.add_argument("--img_size", type=int, default=224)
    args, _ = sp.parse_known_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st.set_page_config(page_title="Cancer Detector", page_icon="🩺")
    st.title("🩺 Cancer Image Detector")
    st.caption("Research demo — not for medical use.")

    model, class_names, preprocess = load(args.weights, args.model, dev)

    uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","bmp","tiff"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded", use_column_width=True)

        x = preprocess(image).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()

        pred_idx = int(torch.argmax(logits, dim=1).item())
        pred_label = class_names[pred_idx]
        st.subheader(f"Prediction: **{pred_label}**")
        st.write({cls: float(p) for cls, p in zip(class_names, probs)})

        if "cancer" in pred_label.lower():
            st.warning("This image is predicted as **cancerous**. Seek expert verification.", icon="⚠️")
        else:
            st.success("This image is predicted as **non-cancerous**. Still, verify with a professional.", icon="✅")

if __name__ == "__main__":
    main()
