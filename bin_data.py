from pathlib import Path

for split in ["train", "val", "test"]:
    for i in range(500):
        Path("data", split, f"{i:05d}").mkdir(parents=True, exist_ok=True)
