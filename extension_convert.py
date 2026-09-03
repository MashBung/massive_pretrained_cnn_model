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

# from pathlib import Path
# from PIL import Image

# source = Path(r"D:\deep_learning\customIamgenet\archive")

# new_ext = ".png"
# convert_exts = {".jpg", ".jpeg", ".bmp", ".webp"}  # png는 이미 png이므로 제외

# for img_path in source.rglob("*"):
#     if img_path.suffix.lower() not in convert_exts:
#         continue

#     new_path = img_path.with_suffix(new_ext)  # 같은 폴더, 같은 이름으로 저장

#     try:
#         with Image.open(img_path) as img:
#             img.convert("RGB").save(new_path, "PNG")
#         img_path.unlink()  # 변환 성공 후 원본 삭제
#         print(f"변환: {img_path.name} -> {new_path.name}")
#     except Exception as e:
#         print(f"실패: {img_path} - {e}")
