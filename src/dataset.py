
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

def _grayscale_to_rgb(img):
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img

def make_transforms(img_size: int):
    train_tfms = transforms.Compose([
        transforms.Lambda(_grayscale_to_rgb),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Lambda(_grayscale_to_rgb),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tfms, eval_tfms

def make_dataloaders(data_dir: str, img_size: int, batch_size: int, num_workers: int = 4):
    train_tfms, eval_tfms = make_transforms(img_size)
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(f"{data_dir}/test",  transform=eval_tfms)

    class_names = train_ds.classes  # ['cancerous', 'non_cancerous'] (expected)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, class_names
