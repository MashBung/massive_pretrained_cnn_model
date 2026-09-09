import torchvision.transforms.v2 as tf
import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.geometry.transform as KG

cpu_transform = tf.Compose(
    [
        tf.ToImage(),
    ]
)


class GPUTrainTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = K.AugmentationSequential(
            K.RandomResizedCrop((320, 320), scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            K.RandomRotation(degrees=10.0),
            data_keys=["input"],
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.aug(x)


class GPUEvalTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = KG.Resize((320, 320))

    def forward(self, x):
        x = x.float() / 255.0
        return self.resize(x)


gpu_train_transform = GPUTrainTransform()
gpu_eval_transform = GPUEvalTransform()
