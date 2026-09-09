from pathlib import Path
from PIL import Image

source = Path(r"D:\deep_learning\customIamgenet\archive")

new_ext = ".png"
convert_exts = {".jpg", ".jpeg", ".bmp", ".webp"}

for img_path in source.rglob("*"):
    if img_path.suffix.lower() not in convert_exts:
        continue

    new_path = img_path.with_suffix(new_ext)

    with Image.open(img_path) as img:
        img.convert("RGB").save(new_path, "PNG")
    img_path.unlink()
    print(f"변환: {img_path.name} -> {new_path.name}")
