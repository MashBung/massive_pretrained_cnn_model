import torchvision.transforms.v2 as tf
import torch
import torch.nn as nn
import kornia.augmentation as K

# CPU에서는 최소한만 (PIL → 텐서 변환)
cpu_transform = tf.Compose(
    [
        tf.ToImage(),
    ]
)


class GPUTrainTransform(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.aug = K.AugmentationSequential(
            K.RandomResizedCrop((320, 320), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            K.RandomRotation(degrees=10.0),
            data_keys=["input"],
        )
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.float() / 255.0  # uint8 → float32, 0~1
        x = self.aug(x)  # augmentation (GPU에서, 샘플별 독립 랜덤성 유지)
        x = (x - self.mean) / self.std  # normalize
        return x


class GPUEvalTransform(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        return x


gpu_train_transform = GPUTrainTransform(
    mean=[0.4484, 0.4371, 0.3758],
    std=[0.2749, 0.2660, 0.2726],
)

gpu_eval_transform = GPUEvalTransform(
    mean=[0.4484, 0.4371, 0.3758],
    std=[0.2749, 0.2660, 0.2726],
)


# import torchvision.transforms.v2 as tf
# import torch

# # cpu_transform = tf.Compose(
# #     [
# #         tf.ToImage(),
# #         tf.RandomHorizontalFlip(p=0.5),
# #         tf.RandomRotation(degrees=10),
# #     ]
# # )


# cpu_transform = tf.Compose(
#     [
#         tf.ToImage(),
#         tf.RandomResizedCrop(320, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
#         tf.RandomHorizontalFlip(p=0.5),
#         tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
#         tf.RandomRotation(degrees=10),
#     ]
# )

# meanstd_transform = tf.Compose(
#     [
#         tf.ToDtype(
#             torch.float32,
#             scale=True,
#         )
#     ]
# )

# gpu_transform = tf.Compose(
#     [
#         tf.ToDtype(
#             torch.float32,
#             scale=True,
#         ),
#         tf.Normalize(
#             mean=[0.4484, 0.4371, 0.3758],
#             std=[0.2749, 0.2660, 0.2726],
#         ),
#     ]
# )
