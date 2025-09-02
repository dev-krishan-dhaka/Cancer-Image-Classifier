
from dataclasses import dataclass

@dataclass
class TrainConfig:
    data_dir: str = "./data"
    out_dir: str = "./runs/exp1"
    model_name: str = "resnet18"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    img_size: int = 224
    num_workers: int = 4
    seed: int = 42
    early_stopping_patience: int = 5
    class_weights: str | None = None  # 'auto' or None
    mixed_precision: bool = True
