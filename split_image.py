from pathlib import Path
import torchvision.transforms.v2 as tf
from torchvision.transforms import InterpolationMode
import random
from torchvision.io import read_image, write_png, ImageReadMode
import os

random.seed(42)

source_root = Path(rf".\customIamgenet\archive")
save_root = Path(rf".\customIamgenet\data")

resize = tf.Resize(
    (320, 320),
    interpolation=InterpolationMode.BILINEAR,
    antialias=True,
)

folders = [f.name for f in source_root.iterdir() if f.is_dir()]

for folder_name in folders:
    i = 0
    src = source_root / folder_name
    files = list(src.iterdir())
    random.shuffle(files)

    total = len(files)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }

    for split_name, split_files in splits.items():
        dst = save_root / split_name / folder_name
        dst.mkdir(parents=True, exist_ok=True)

        for f in split_files:
            img = read_image(str(f), mode=ImageReadMode.RGB)  # 읽기
            img = resize(img)  # 리사이즈
            write_png(img, str(dst / f.name))  # 저장
            i = i + 1
    print(i)

print("완료!")

os.system("shutdown /s /t 60")
