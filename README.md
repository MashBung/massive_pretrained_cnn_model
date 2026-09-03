# massive_pretrained_cnn_model

객체 탐지의 백본으로 사용하기 위해 CNN을 PyTorch로 직접 설계하고, **500-class 커스텀 ImageNet-style 데이터셋(약 30만 장)으로 처음부터(from scratch) 사전학습**한 프로젝트와 모델입니다.

torchvision의 pretrained weight를 가져다 쓰는 대신, 백본 구조 설계 → 데이터 파이프라인 구축 → 학습 병목 해결 → 과적합 대응까지 전 과정을 직접 구현했습니다.

## 학습 로그

<p align="center">
  <img width="737" height="344" alt="cnn_log" src="https://github.com/user-attachments/assets/5b54fca1-ee3e-47cf-980e-ad723a155eb2" />
</p>
<p align="center">
  <img width="570" height="379" alt="cnn_log1" src="https://github.com/user-attachments/assets/e665fa95-2b8f-43f2-b488-dbebdd0390d8" />
  <img width="491" height="329" alt="cnn_log2" src="https://github.com/user-attachments/assets/48b45407-1d58-446c-baf7-c2adc937b1d3" />
</p>
<p align="center">
  <img width="197" height="149" alt="cnn_result" src="https://github.com/user-attachments/assets/8f1d34c8-ba6a-4758-b7e9-2db3d1e5e447" />
</p>

## 핵심 결과

| 항목 | 값 |
|---|---|
| 훈련 정확도 (Top-1) | 85.37% |
| 검증 정확도 (Top-1) | 68.05% |
| 클래스 수 | 500 |
| 입력 해상도 | 320 × 320 |
| 학습 환경 | RTX 5070 Ti |

## 사전학습 모델을 만든 이유

- 객체 탐지에서 **warm start**가 가능하고, **입력 해상도와 stride 구조를 처음부터 탐지기에 맞춘 백본**이 필요했습니다. ImageNet 224 기준으로 학습된 공개 weight와 달리, 320 입력에서 C3/C4/C5가 정확히 stride 8/16/32로 나오도록 설계했습니다.
- 백본을 미리 학습하지 않으면 객체 탐지 훈련에서 객체와 배경을 구별하는 표현력을 확보하는 데 시간이 오래 걸립니다.
- 백본 내부 구조(stem, residual block, 활성함수)에 대한 설계 결정을 직접 내리고, CNN이 실제로 무엇을 학습하는지 확인해 볼 수 있습니다.

## 아키텍처

`backbone.py`

```
Input 3×320×320
 └─ Stem      Conv3×3 s2 → BN → SiLU            32ch  @160   (stride 2)
 └─ Stage1    ResidualBlock ×2                   32ch  @160
 └─ Stage2    ResidualBlock ×2 (첫 블록 s2)      64ch  @80    (stride 4)
 └─ Stage3    ResidualBlock ×2 (첫 블록 s2)     128ch  @40    (stride 8)   ← C3
 └─ Stage4    ResidualBlock ×2 (첫 블록 s2)     256ch  @20    (stride 16)  ← C4
 └─ Stage5    ResidualBlock ×2 (첫 블록 s2)     512ch  @10    (stride 32)  ← C5
 └─ Head      GAP → Flatten → Dropout(0.3) → Linear(512, 500)
```

**ResidualBlock**: `Conv3×3 → BN → SiLU → Conv3×3 → BN` + shortcut, 이후 SiLU.
채널이 바뀌거나 stride≠1인 경우 shortcut에 1×1 Conv + BN projection을 사용합니다.

### 설계 결정과 이유

| 결정 | 이유 |
|---|---|
| **Stem에서 MaxPool 제거** (ResNet 원본은 Conv s2 + MaxPool s2로 4× 축소) | 초기 단계에서 공간 정보 손실을 줄여 소형 객체 탐지에 유리한 고해상도 feature를 유지하기 위함. 대신 Stage2에서 stride 2로 축소. |
| **SiLU 활성함수** (ReLU 대신) | 음수 영역에서 완전히 0이 되지 않아 gradient flow가 부드럽고, 최신 탐지기(YOLOv5~v8, EfficientNet)에서 채택된 검증된 선택. |
| **Stage별 채널 32→64→128→256→512** | ResNet-18과 동일한 5-stage 구조를 유지하되 첫 stage 채널을 64→32로 줄여 320 해상도에서의 연산량 조절. |
| **Stride 8/16/32 출력** | FPN 연결을 위한 표준 multi-scale 구성. |

## 데이터 파이프라인

### 데이터셋

- 500 클래스, 약 275K train / 31K val / 31K test
- `split_image.py`: 클래스별 8:1:1 분할(seed 고정), 320×320 리사이즈 후 PNG로 사전 저장
- `extension_convert.py`: jpg/bmp/webp 혼재 → PNG 통일
- 정규화 mean/std는 학습 데이터에서 직접 계산: `mean=[0.4484, 0.4371, 0.3758]`, `std=[0.2749, 0.2660, 0.2726]`

### GPU 증강으로 CPU 병목 해결

`transform.py`

초기에는 torchvision transform으로 CPU에서 증강을 수행했는데, **DataLoader가 GPU 처리 속도를 따라가지 못해 GPU 활용률이 낮게 유지**되는 문제가 있었습니다.

해결책으로 **CPU에서는 PIL → uint8 텐서 변환만 수행하고, 증강·정규화 전체를 Kornia로 GPU에서 배치 단위로 처리**하도록 파이프라인을 재구성했습니다.

```python
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
```

- Kornia는 배치 내 **샘플별 독립적인 랜덤 파라미터**를 적용하므로 CPU 증강과 동일한 다양성을 유지
- mean/std를 `register_buffer`로 등록해 `.to(device)` 시 함께 이동
- 결과: `batch_size=256, num_workers=2`로 **GPU 활용률 약 86%** 안정 유지

## 학습 설정

`train.py`, `train_val_code.py`

| 항목 | 설정 |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=0.02) |
| Scheduler | CosineAnnealingLR (T_max=150, eta_min=1e-6) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Precision | AMP (fp16 autocast + GradScaler) |
| Batch size | 256 |
| Regularization | Dropout 0.3, label smoothing, weight decay, RandomResizedCrop / ColorJitter / Rotation |

### 체크포인트 전략

- `checkpoint/last_checkpoint.pth`: model / optimizer / scheduler / scaler state + epoch + best_val_acc를 매 epoch 저장 → **중단 후 완전 재개 가능**
- `state_dict/pretrained_cnn_{epoch}.pth`: val acc 갱신 시 model state_dict만 저장 → 다운스트림 백본으로 바로 로드

## 학습 과정에서 겪은 문제와 해결

**1. CPU 데이터 로딩 병목**
증상: GPU 활용률 저하, epoch당 시간 과다.
해결: GPU 증강 파이프라인으로 전환. 320×320 PNG를 사전 저장해 디코딩 비용도 절감.

**2. 과적합**
증상: epoch 20~22 시점 train acc ~88% vs val acc ~66%로 격차 확대.
대응: label smoothing, weight decay 상향, Dropout, 증강 강도 조절.
한계: 데이터 규모(클래스당 ~550장)가 근본 원인으로, 최종 val ~68%에서 수렴.

**3. 학습 재개 로직 버그**
epoch 루프 중복, 체크포인트 저장 경로 오류, state_dict 키 오타 등을 수정하며 재개 로직을 안정화.

## 평가

- `test.py` — Top-1 / Top-5 정확도 및 **500개 클래스별 정확도**를 계산해 `log/per_class_acc.txt`에 저장 (가장 못 맞추는 클래스 순으로 정렬). 어떤 클래스가 혼동되는지 분석하는 데 사용.
- `image_test.py` — 단일 이미지 추론, Top-5 확률 출력.

## 다운스트림 활용: 탐지기 백본으로 사용

이 모델은 이후 **FCOS 객체 탐지기**의 백본으로 사용됩니다. 분류 head를 제거하고 Stage3/4/5 출력을 FPN에 연결합니다.

```python
model = cnn(num_classes=500)
model.load_state_dict(torch.load("state_dict/pretrained_cnn_28.pth"))

# C3, C4, C5 추출 (128/256/512ch, stride 8/16/32)
x = model.stem(x)
x = model.stage1(x)
x = model.stage2(x)
c3 = model.stage3(x)
c4 = model.stage4(c3)
c5 = model.stage5(c4)
```

## 프로젝트 구조

```
.
├── backbone.py          # CNN 백본 정의 (ResidualBlock, cnn)
├── transform.py         # CPU/GPU transform (Kornia 증강)
├── train.py             # 학습 엔트리포인트, 체크포인트 재개
├── train_val_code.py    # train / val 루프 (AMP)
├── test.py              # Top-1/Top-5, 클래스별 정확도 평가
├── image_test.py        # 단일 이미지 추론
├── split_image.py       # 데이터 분할 + 리사이즈 전처리
└── extension_convert.py # 이미지 확장자 통일
```

## 실행

```bash
pip install torch torchvision kornia pillow

# 1. 데이터 전처리 (경로 수정 후)
python extension_convert.py
python split_image.py

# 2. 학습
python train.py

# 3. 평가
python test.py
python image_test.py
```

## 다음 단계

- Forward에서 C3/C4/C5를 직접 반환하는 `features()` 메서드 추가
- 더 강한 정규화(Mixup / CutMix, Stochastic Depth) 실험
- FCOS 탐지기와 연결한 end-to-end 결과# massive_pretrained_cnn_model

객체 탐지의 백본으로 사용하기 위해 CNN을 PyTorch로 직접 설계하고, **500-class 커스텀 ImageNet-style 데이터셋(약 30만 장)으로 처음부터(from scratch) 사전학습**한 프로젝트와 모델입니다.

torchvision의 pretrained weight를 가져다 쓰는 대신, 백본 구조 설계 → 데이터 파이프라인 구축 → 학습 병목 해결 → 과적합 대응까지 전 과정을 직접 구현했습니다.

## 학습 로그

<p align="center">
  <img width="737" height="344" alt="cnn_log" src="https://github.com/user-attachments/assets/5b54fca1-ee3e-47cf-980e-ad723a155eb2" />
</p>
<p align="center">
  <img width="570" height="379" alt="cnn_log1" src="https://github.com/user-attachments/assets/e665fa95-2b8f-43f2-b488-dbebdd0390d8" />
  <img width="491" height="329" alt="cnn_log2" src="https://github.com/user-attachments/assets/48b45407-1d58-446c-baf7-c2adc937b1d3" />
</p>
<p align="center">
  <img width="197" height="149" alt="cnn_result" src="https://github.com/user-attachments/assets/8f1d34c8-ba6a-4758-b7e9-2db3d1e5e447" />
</p>

## 핵심 결과

| 항목 | 값 |
|---|---|
| 훈련 정확도 (Top-1) | 85.37% |
| 검증 정확도 (Top-1) | 68.05% |
| 클래스 수 | 500 |
| 입력 해상도 | 320 × 320 |
| 학습 환경 | RTX 5070 Ti |

## 사전학습 모델을 만든 이유

- 객체 탐지에서 **warm start**가 가능하고, **입력 해상도와 stride 구조를 처음부터 탐지기에 맞춘 백본**이 필요했습니다. ImageNet 224 기준으로 학습된 공개 weight와 달리, 320 입력에서 C3/C4/C5가 정확히 stride 8/16/32로 나오도록 설계했습니다.
- 백본을 미리 학습하지 않으면 객체 탐지 훈련에서 객체와 배경을 구별하는 표현력을 확보하는 데 시간이 오래 걸립니다.
- 백본 내부 구조(stem, residual block, 활성함수)에 대한 설계 결정을 직접 내리고, CNN이 실제로 무엇을 학습하는지 확인해 볼 수 있습니다.

## 아키텍처

`backbone.py`

```
Input 3×320×320
 └─ Stem      Conv3×3 s2 → BN → SiLU            32ch  @160   (stride 2)
 └─ Stage1    ResidualBlock ×2                   32ch  @160
 └─ Stage2    ResidualBlock ×2 (첫 블록 s2)      64ch  @80    (stride 4)
 └─ Stage3    ResidualBlock ×2 (첫 블록 s2)     128ch  @40    (stride 8)   ← C3
 └─ Stage4    ResidualBlock ×2 (첫 블록 s2)     256ch  @20    (stride 16)  ← C4
 └─ Stage5    ResidualBlock ×2 (첫 블록 s2)     512ch  @10    (stride 32)  ← C5
 └─ Head      GAP → Flatten → Dropout(0.3) → Linear(512, 500)
```

**ResidualBlock**: `Conv3×3 → BN → SiLU → Conv3×3 → BN` + shortcut, 이후 SiLU.
채널이 바뀌거나 stride≠1인 경우 shortcut에 1×1 Conv + BN projection을 사용합니다.

### 설계 결정과 이유

| 결정 | 이유 |
|---|---|
| **Stem에서 MaxPool 제거** (ResNet 원본은 Conv s2 + MaxPool s2로 4× 축소) | 초기 단계에서 공간 정보 손실을 줄여 소형 객체 탐지에 유리한 고해상도 feature를 유지하기 위함. 대신 Stage2에서 stride 2로 축소. |
| **SiLU 활성함수** (ReLU 대신) | 음수 영역에서 완전히 0이 되지 않아 gradient flow가 부드럽고, 최신 탐지기(YOLOv5~v8, EfficientNet)에서 채택된 검증된 선택. |
| **Stage별 채널 32→64→128→256→512** | ResNet-18과 동일한 5-stage 구조를 유지하되 첫 stage 채널을 64→32로 줄여 320 해상도에서의 연산량 조절. |
| **Stride 8/16/32 출력** | FPN 연결을 위한 표준 multi-scale 구성. |

## 데이터 파이프라인

### 데이터셋

- 500 클래스, 약 275K train / 31K val / 31K test
- `split_image.py`: 클래스별 8:1:1 분할(seed 고정), 320×320 리사이즈 후 PNG로 사전 저장
- `extension_convert.py`: jpg/bmp/webp 혼재 → PNG 통일
- 정규화 mean/std는 학습 데이터에서 직접 계산: `mean=[0.4484, 0.4371, 0.3758]`, `std=[0.2749, 0.2660, 0.2726]`

### GPU 증강으로 CPU 병목 해결

`transform.py`

초기에는 torchvision transform으로 CPU에서 증강을 수행했는데, **DataLoader가 GPU 처리 속도를 따라가지 못해 GPU 활용률이 낮게 유지**되는 문제가 있었습니다.

해결책으로 **CPU에서는 PIL → uint8 텐서 변환만 수행하고, 증강·정규화 전체를 Kornia로 GPU에서 배치 단위로 처리**하도록 파이프라인을 재구성했습니다.

```python
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
```

- Kornia는 배치 내 **샘플별 독립적인 랜덤 파라미터**를 적용하므로 CPU 증강과 동일한 다양성을 유지
- mean/std를 `register_buffer`로 등록해 `.to(device)` 시 함께 이동
- 결과: `batch_size=256, num_workers=2`로 **GPU 활용률 약 86%** 안정 유지

## 학습 설정

`train.py`, `train_val_code.py`

| 항목 | 설정 |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=0.02) |
| Scheduler | CosineAnnealingLR (T_max=150, eta_min=1e-6) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Precision | AMP (fp16 autocast + GradScaler) |
| Batch size | 256 |
| Regularization | Dropout 0.3, label smoothing, weight decay, RandomResizedCrop / ColorJitter / Rotation |

### 체크포인트 전략

- `checkpoint/last_checkpoint.pth`: model / optimizer / scheduler / scaler state + epoch + best_val_acc를 매 epoch 저장 → **중단 후 완전 재개 가능**
- `state_dict/pretrained_cnn_{epoch}.pth`: val acc 갱신 시 model state_dict만 저장 → 다운스트림 백본으로 바로 로드

## 학습 과정에서 겪은 문제와 해결

**1. CPU 데이터 로딩 병목**
증상: GPU 활용률 저하, epoch당 시간 과다.
해결: GPU 증강 파이프라인으로 전환. 320×320 PNG를 사전 저장해 디코딩 비용도 절감.

**2. 과적합**
증상: epoch 20~22 시점 train acc ~88% vs val acc ~66%로 격차 확대.
대응: label smoothing, weight decay 상향, Dropout, 증강 강도 조절.
한계: 데이터 규모(클래스당 ~550장)가 근본 원인으로, 최종 val ~68%에서 수렴.

**3. 학습 재개 로직 버그**
epoch 루프 중복, 체크포인트 저장 경로 오류, state_dict 키 오타 등을 수정하며 재개 로직을 안정화.

## 평가

- `test.py` — Top-1 / Top-5 정확도 및 **500개 클래스별 정확도**를 계산해 `log/per_class_acc.txt`에 저장 (가장 못 맞추는 클래스 순으로 정렬). 어떤 클래스가 혼동되는지 분석하는 데 사용.
- `image_test.py` — 단일 이미지 추론, Top-5 확률 출력.

## 다운스트림 활용: 탐지기 백본으로 사용

이 모델은 이후 **FCOS 객체 탐지기**의 백본으로 사용됩니다. 분류 head를 제거하고 Stage3/4/5 출력을 FPN에 연결합니다.

```python
model = cnn(num_classes=500)
model.load_state_dict(torch.load("state_dict/pretrained_cnn_28.pth"))

# C3, C4, C5 추출 (128/256/512ch, stride 8/16/32)
x = model.stem(x)
x = model.stage1(x)
x = model.stage2(x)
c3 = model.stage3(x)
c4 = model.stage4(c3)
c5 = model.stage5(c4)
```

## 프로젝트 구조

```
.
├── backbone.py          # CNN 백본 정의 (ResidualBlock, cnn)
├── transform.py         # CPU/GPU transform (Kornia 증강)
├── train.py             # 학습 엔트리포인트, 체크포인트 재개
├── train_val_code.py    # train / val 루프 (AMP)
├── test.py              # Top-1/Top-5, 클래스별 정확도 평가
├── image_test.py        # 단일 이미지 추론
├── split_image.py       # 데이터 분할 + 리사이즈 전처리
└── extension_convert.py # 이미지 확장자 통일
```

## 실행

```bash
pip install torch torchvision kornia pillow

# 1. 데이터 전처리 (경로 수정 후)
python extension_convert.py
python split_image.py

# 2. 학습
python train.py

# 3. 평가
python test.py
python image_test.py
```

## 다음 단계

- Forward에서 C3/C4/C5를 직접 반환하는 `features()` 메서드 추가
- 더 강한 정규화(Mixup / CutMix, Stochastic Depth) 실험
- FCOS 탐지기와 연결한 end-to-end 결과
