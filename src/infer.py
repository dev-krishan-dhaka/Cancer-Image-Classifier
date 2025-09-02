
import argparse, torch
from PIL import Image
from torchvision import transforms
from .model import create_model

def build_preprocess(img_size=224):
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert('RGB')),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def load_model(weights_path: str, model_name: str = "resnet18", device="cpu"):
    ckpt = torch.load(weights_path, map_location=device)
    class_names = ckpt.get("class_names", ["cancerous","non_cancerous"])
    model = create_model(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    return model, class_names

def predict_image(weights: str, image_path: str, model_name="resnet18", img_size=224, device="cpu"):
    model, class_names = load_model(weights, model_name, device)
    preprocess = build_preprocess(img_size)
    img = Image.open(image_path)
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()
    top_idx = int(torch.argmax(logits, dim=1).item())
    return {"pred_class": class_names[top_idx], "probs": dict(zip(class_names, probs))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="resnet18")
    p.add_argument("--img_size", type=int, default=224)
    args = p.parse_args()
    out = predict_image(args.weights, args.image, args.model, args.img_size, device="cpu")
    print(out)

if __name__ == "__main__":
    main()
