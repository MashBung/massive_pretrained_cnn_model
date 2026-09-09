import torch


def train(model, train_loader, loss_fn, optimizer, device, gpu_train_transform, scaler):
    model.train()
    running_loss = 0.0
    Accuracy = 0
    total = 0
    total_batches = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )  # images(256,3,320,320) labels(256,)
        images = gpu_train_transform(images)  # 0~1 스케일

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            outputs = model(images)  # (256,500)
            loss = loss_fn(outputs, labels)  # images(256,500) labels(256,)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss = running_loss + loss.item()
        predicted = outputs.argmax(dim=1)
        total = total + labels.size(0)
        Accuracy = Accuracy + (predicted == labels).sum().item()

        print(f"\r Batch [{i+1}/{total_batches}]", end="")

    avg_loss = running_loss / total_batches
    accuracy = 100 * Accuracy / total
    return avg_loss, accuracy


def val(model, val_loader, loss_fn, device, gpu_val_transform):
    model.eval()
    running_loss = 0.0
    Accuracy = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            images = gpu_val_transform(images)

            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            running_loss = running_loss + loss.item()
            predicted = outputs.argmax(dim=1)
            total = total + labels.size(0)
            Accuracy = Accuracy + (predicted == labels).sum().item()

        avg_loss = running_loss / len(val_loader)
        accuracy = 100 * Accuracy / total
        return avg_loss, accuracy
