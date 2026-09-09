import torch
import torch.nn as nn
from torchvision import datasets
from transform import cpu_transform
from torch.utils.data import DataLoader
from backbone import cnn
from transform import gpu_eval_transform, gpu_train_transform
from train_val_code import train, val
from pathlib import Path

TRAIN_DATASET_DIR = rf"D:\deep_learning\customIamgenet\data\train"
VAL_DATASET_DIR = rf"D:\deep_learning\customIamgenet\data\val"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    scaler = torch.amp.GradScaler(device=device)

    train_dataset = datasets.ImageFolder(
        TRAIN_DATASET_DIR,
        transform=cpu_transform,
    )  # (3,320,320) dtype = uint8

    val_dataset = datasets.ImageFolder(
        VAL_DATASET_DIR,
        transform=cpu_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        # prefetch_factor=3,
    )  # (256,3,320,320)

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        # prefetch_factor=3,
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Classes to index: {train_dataset.class_to_idx}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    num_classes = len(train_dataset.classes)
    print(num_classes)

    model = cnn(num_classes=num_classes).to(device)
    print(model)

    gpu_train_transform = gpu_train_transform.to(device)
    gpu_eval_transform = gpu_eval_transform.to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.02,
    )

    epochs = 150

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    start_epoch = 0
    ckpt_path = Path("./checkpoint/last_checkpoint.pth")

    if ckpt_path.exists():
        print(f"체크포인트: {ckpt_path} - 이어서 학습합니다.")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        print(f"epoch {start_epoch}부터 재개, best_val_acc={best_val_acc:.2f}")
    else:
        print("체크포인트 없음 — 처음부터 학습합니다.")

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            gpu_train_transform,
            scaler,
        )

        val_loss, val_acc = val(
            model,
            val_loader,
            loss_fn,
            device,
            gpu_eval_transform,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")

        with open("./log/log.txt", "a", encoding="UTF-8") as f:
            f.write(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}, "
                f"lr={current_lr:.6f},\n"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                r"./checkpoint/last_checkpoint.pth",
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"./state_dict/pretrained_cnn_{epoch}.pth")
            print(f"  Best model saved! ({best_val_acc:.2f}%)")
