
import argparse, io, torch
from flask import Flask, request, jsonify
from PIL import Image
from src.infer import load_model, build_preprocess

app = Flask(__name__)

def create_app(weights, model_name, img_size):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(weights, model_name, dev)
    preprocess = build_preprocess(img_size)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400
        f = request.files["file"]
        img = Image.open(f.stream).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
        pred_idx = int(torch.argmax(logits, dim=1).item())
        return jsonify({"pred_class": class_names[pred_idx], "probs": dict(zip(class_names, [float(p) for p in probs]))})
    return app

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = create_app(args.weights, args.model, args.img_size)
    app.run(host=args.host, port=args.port, debug=False)
