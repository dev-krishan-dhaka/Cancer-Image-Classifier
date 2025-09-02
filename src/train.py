
import argparse, os, time, csv, math
import numpy as np
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
from .config import TrainConfig
from .dataset import make_dataloaders
from .model import create_model
from .utils import set_seed, device, ensure_dir

def compute_class_weights(train_loader, num_classes=2):
    counts = [0]*num_classes
    for _, y in train_loader.dataset.samples:
        counts[y]+=1
    total = sum(counts)
    weights = [total/(c if c>0 else 1) for c in counts]
    s = sum(weights)
    weights = [w/s for w in weights]
    return torch.tensor(weights, dtype=torch.float)

def evaluate(model, loader, criterion, dev, return_probs=False):
    model.eval()
    all_y, all_p = [], []
    loss_sum = 0.0; n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item()*y.size(0); n += y.size(0)
            probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
            all_p.extend(probs); all_y.extend(y.detach().cpu().numpy())
    acc = accuracy_score(all_y, np.array(all_p)>=0.5)
    f1  = f1_score(all_y, np.array(all_p)>=0.5)
    try:
        auc = roc_auc_score(all_y, all_p)
    except:
        auc = float('nan')
    metrics = {"loss": loss_sum/n, "acc": acc, "f1": f1, "auc": auc}
    if return_probs: return metrics, np.array(all_p), np.array(all_y)
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/exp1")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--class_weights", type=str, default=None, help="'auto' or None")
    p.add_argument("--mixed_precision", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir, out_dir=args.out_dir, model_name=args.model, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, img_size=args.img_size,
        num_workers=args.num_workers, seed=args.seed, early_stopping_patience=args.early_stopping_patience,
        class_weights=args.class_weights, mixed_precision=args.mixed_precision
    )

    set_seed(cfg.seed); ensure_dir(cfg.out_dir)
    train_loader, val_loader, test_loader, class_names = make_dataloaders(cfg.data_dir, cfg.img_size, cfg.batch_size, cfg.num_workers)
    dev = device()
    model = create_model(cfg.model_name, num_classes=2, pretrained=True).to(dev)

    if cfg.class_weights == "auto":
        class_w = compute_class_weights(train_loader).to(dev)
    else:
        class_w = None
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.mixed_precision)

    best_val = float('inf')
    patience = cfg.early_stopping_patience
    history = []

    for epoch in range(1, cfg.epochs+1):
        model.train()
        train_loss = 0.0; n = 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optim.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.mixed_precision):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            train_loss += loss.item()*y.size(0); n += y.size(0)

        train_loss /= max(n,1)
        val_metrics = evaluate(model, val_loader, criterion, dev)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k,v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(cfg.out_dir, "metrics.csv"), index=False)

        print(f"Epoch {epoch}/{cfg.epochs}  train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  val_f1={val_metrics['f1']:.4f}  val_auc={val_metrics['auc']:.4f}")

        # Early stopping on val_loss
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"model": model.state_dict(), "class_names": class_names, "cfg": cfg.__dict__},
                       os.path.join(cfg.out_dir, "best.pt"))
            patience = cfg.early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    # Final test evaluation
    best = torch.load(os.path.join(cfg.out_dir, "best.pt"), map_location=dev)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(model, test_loader, criterion, dev)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(cfg.out_dir, "test_metrics.csv"), index=False)
    print("Test:", test_metrics)

if __name__ == "__main__":
    main()
