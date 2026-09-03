import torch


def train(
    model, train_loader, loss_function, optimizer, device, gpu_train_transform, scaler
):
    model.train()
    running_loss = 0.0
    Accuracy = 0
    total = 0
    total_batches = len(train_loader)

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images = gpu_train_transform(images)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = loss_function(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss = running_loss + loss.item()
        predicted = outputs.argmax(dim=1)
        total = total + labels.size(0)
        Accuracy = Accuracy + (predicted == labels).sum().item()

        print(f"\r  Batch [{i+1}/{total_batches}]", end="")

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * Accuracy / total
    return avg_loss, accuracy


def val(model, val_loader, loss_function, device, gpu_eval_transform):
    model.eval()
    running_loss = 0.0
    Accuracy = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            images = gpu_eval_transform(images)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = loss_function(outputs, labels)

            running_loss = running_loss + loss.item()
            predicted = outputs.argmax(dim=1)
            total = total + labels.size(0)
            Accuracy = Accuracy + (predicted == labels).sum().item()

        avg_loss = running_loss / len(val_loader)
        accuracy = 100 * Accuracy / total
        return avg_loss, accuracy
