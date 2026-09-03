import torch
from torchvision import datasets
from backbone import cnn
from transform import gpu_eval_transform, cpu_transform
from PIL import Image
import os

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model_path = r"./state_dict/pretrained_cnn_28.pth"
    image_path = r".\horse.png"

    # test_dataset = datasets.ImageFolder(
    #     r".\data\test",
    # )
    # classes = test_dataset.classes
    # num_classes = len(classes)
    class_dir = r".\data\test"
    classes = sorted(d.name for d in os.scandir(class_dir) if d.is_dir())
    num_classes = len(classes)

    model = cnn(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gpu_eval_transform = gpu_eval_transform.to(device)

    img = Image.open(image_path).convert("RGB")
    img_tensor = cpu_transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        img_tensor = gpu_eval_transform(img_tensor)

        with torch.amp.autocast(device_type=device):
            outputs = model(img_tensor)

        probs = torch.softmax(outputs, dim=1)
        top5_probs, top5_idx = probs.topk(5, dim=1)

    print(f"\n입력 이미지: {image_path}\n")
    print("Top-5 예측 결과:")
    for rank, (prob, idx) in enumerate(zip(top5_probs[0], top5_idx[0]), start=1):
        class_name = classes[idx.item()]
        print(f"  {rank}. {class_name} ({prob.item()*100:.2f}%)")
