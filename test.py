import torch.nn as nn
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from transform import cpu_transform, gpu_eval_transform
from backbone import cnn

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model_path = rf"./state_dict/pretrained_cnn_23.pth"

    test_dataset = datasets.ImageFolder(
        rf"D:\deep_learning\customIamgenet\data\test",
        transform=cpu_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Classes: {test_dataset.classes}")
    print(f"Val samples: {len(test_dataset)}")

    num_classes = len(test_dataset.classes)
    print(num_classes)

    model = cnn(num_classes=num_classes).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    loss_fn = nn.CrossEntropyLoss(
        weight=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.1,
    )

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    # 클래스별 집계용 (index 기준)
    class_correct_top1 = torch.zeros(num_classes)  # (500,)
    class_correct_top5 = torch.zeros(num_classes)  # (500,)
    class_total = torch.zeros(num_classes)  # (500,)

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            imgs = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            imgs = gpu_eval_transform(imgs)

            with torch.amp.autocast(device_type=device):
                outputs = model(imgs)  # (256,500)
                loss = loss_fn(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            total += labels.size(0)

            pred_top1 = outputs.topk(1, dim=1).indices  # (256,1)
            top1_correct_mask = pred_top1.squeeze(1) == labels
            correct_top1 += top1_correct_mask.sum().item()

            pred_top5 = outputs.topk(5, dim=1).indices
            top5_correct_mask = (pred_top5 == labels.unsqueeze(1)).any(dim=1)
            correct_top5 += top5_correct_mask.sum().item()

            labels_cpu = labels.cpu()
            class_total.index_add_(0, labels_cpu, torch.ones(labels.size(0)))
            class_correct_top1.index_add_(
                0, labels_cpu, top1_correct_mask.float().cpu()
            )
            class_correct_top5.index_add_(
                0, labels_cpu, top5_correct_mask.float().cpu()
            )

            print(f"\r평가 중... {i+1}/{len(test_loader)}", end="")

    avg_loss = total_loss / total
    top1_acc = correct_top1 / total * 100
    top5_acc = correct_top5 / total * 100

    print()
    print(f"Val Loss: {avg_loss:.4f}")
    print(f"Top-1 Acc: {top1_acc:.2f}%")
    print(f"Top-5 Acc: {top5_acc:.2f}%")

    class_acc_top1 = torch.where(
        class_total > 0,
        class_correct_top1 / class_total * 100,
        torch.zeros_like(class_total),
    )
    class_acc_top5 = torch.where(
        class_total > 0,
        class_correct_top5 / class_total * 100,
        torch.zeros_like(class_total),
    )

    results = []
    for idx, class_name in enumerate(test_dataset.classes):
        results.append(
            {
                "class": class_name,
                "total": int(class_total[idx].item()),
                "correct_top1": int(class_correct_top1[idx].item()),
                "acc_top1": class_acc_top1[idx].item(),
                "acc_top5": class_acc_top5[idx].item(),
            }
        )

    results_sorted = sorted(results, key=lambda x: x["acc_top1"])

    print("\n=== 클래스별 정확도 (못 맞추는 순) ===")
    print(f"{'Class':<25}{'Total':<8}{'Correct':<10}{'Top1 Acc':<12}{'Top5 Acc':<10}")
    for r in results_sorted:
        print(
            f"{r['class']:<25}{r['total']:<8}{r['correct_top1']:<10}"
            f"{r['acc_top1']:<12.2f}{r['acc_top5']:<10.2f}"
        )

    with open("./log/per_class_acc.txt", "w", encoding="UTF-8") as f:
        f.write(
            f"{'Class':<25}{'Total':<8}{'Correct':<10}{'Top1 Acc':<12}{'Top5 Acc':<10}\n"
        )
        for r in results_sorted:
            f.write(
                f"{r['class']:<25}{r['total']:<8}{r['correct_top1']:<10}"
                f"{r['acc_top1']:<12.2f}{r['acc_top5']:<10.2f}\n"
            )

    print("\n전체 클래스별 결과가 ./log/per_class_acc.txt 에 저장되었습니다.")
