# massive_pretrained_cnn_model

객체 탐지의 백본을로 사용하기 위해 CNN을 PyTorch로 직접 설계하고, **500-class 커스텀 ImageNet 데이터셋(약 26만장)으로 처음부터(from scratch) 사전학습**한 CNN 모델입니다.

torchvision의 pretrained weight 또는 외부 모델을 사용하는 대신 백본 구조 설계 → 데이터 파이프라인 → 학습 까지 과정을 구현했습니다.

## 사전학습 모델을 만든 이유

- 객체 탐지에서 **warm start** 시작하여 초기화된 상태부터 시작하지않아 학습 속도가 높아집니다.
- 백본 내부 구조에 대한 설계 결정을 직접 작성하고 CNN이 실제로 무엇을 학습하는지 확인해 볼 수 있습니다.

## 아키텍처

`backbone.py`

```
Input 
 └─ Stem      Conv3×3 s2 → BN → SiLU            32ch  @160   (stride 2)
 └─ Stage1    ResidualBlock ×2                   32ch  @160
 └─ Stage2    ResidualBlock ×2 (첫 블록 s2)      64ch  @80    (stride 4)
 └─ Stage3    ResidualBlock ×2 (첫 블록 s2)     128ch  @40    (stride 8)   
 └─ Stage4    ResidualBlock ×2 (첫 블록 s2)     256ch  @20    (stride 16) 
 └─ Stage5    ResidualBlock ×2 (첫 블록 s2)     512ch  @10    (stride 32)  
 └─ Head      AdaptiveAvgPool → Flatten → Dropout(0.3) → Linear(512, 500)
```

**ResidualBlock**

```
Input
└─ pipe_x     Conv → BN   (채널과 stride가 다를겨우)
└─ pipe_x     nn.Identity ()(변화가 없을경우 Input 그대로 넘김) 
└─ out        Conv → BN → SiLU
└─ out        Conv → BN
└─ return     pipe_x + out → SiLU 
```

- 층이 깊어질수록 gradient가 곱셈 누적으로 소실되어, 초기 층이 학습의 영향을 받지 못한다. 이전 특징맵을 합쳐서 기울기 소실을 막습니다.

## 설계 결정과 이유

| 결정 | 이유 |
|---|---|
| **MaxPool 제거** | MaxPool은 n*n중 하나만 최댓값을 고르고 추가연산이 들어 주변 픽셀을 가중치를 반영하고 추가연산 없는 stride로 채널 변환 합니다. |
| **SiLU 활성함수** | 음수 영역에서 완전히 0값으로 만들지 않아 죽은 뉴런이 생깁니다. 많은 모델들이 채택한 활성화 함수입니다. |
| **Stage별 채널 32/160→64/80→128/40→256/20→512/10** | 탐지기가 사용하는 스케일 관례에 맞추었습니다. |

## 데이터 파이프라인

- 500 클래스, 약 25만장 train/ 3만장 test / 3만장 val
- `split_image.py`: 클래스별 8:1:1 분할(seed 고정), 320×320 리사이즈 후 PNG로 사전 저장
- `extension_convert.py`: jpg/bmp/webp 혼재 → PNG 통일

## GPU 증강으로 CPU 병목 해결

`transform.py`

torchvision transform은 배치내 처리를 각각 수행하지 못하기 때문에 **CPU가 이미지 변환까지 담당하게되어 GPU활용도가 낮아지는 문제점이 있습니다**

CPU는 **uint8 텐서 변환만 수행하고 증강 단계는 Kornia를 이용하여 GPU 처리하도록** 구성했습니다.

```
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
```
- kornia는 배치 내 **샘플별 독립적인 수행**하므로 배치내 다양한 증강이 이루어집니다.
- cpu 스로틀링과 병목을 막기위해 GPU가 처리합니다.

## 학습 설정

`train.py`, `train_val_code.py`

| 항목 | 설정 |
|---|---|
| Optimizer | AdamW (lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0.02) |
| Scheduler | CosineAnnealingLR (T_max=150, eta_min=1e-6) |
| Loss | CrossEntropyLoss (weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.1,) |
| Precision | AMP (fp16 autocast + GradScaler) |
| Batch size | 256 |

## 핵심 결과

| 항목 | 값 |
|---|---|
| 훈련 정확도 (Top-1) | 83.13% |
| 검증 정확도 (Top-1) | 67.64% |
| 클래스 수 | 500 |
| 입력 해상도 | 320 × 320 |
| 학습 환경 | RTX 5070 Ti |

- 정확도(Accuracy)를 활용하여 훈련과 검증의 정확도를 측정하였습니다.
- `test.py` — Top-1 / Top-5 정확도 및 **500개 클래스별 정확도**를 계산해 `log/per_class_acc.txt`에 저장 (가장 못 맞추는 클래스 순으로 정렬). 어떤 클래스가 혼동되는지 분석하는 데 사용합니다.
- `image_test.py` — 단일 이미지 추론, Top-5 확률 출력합니다.

## 학습 로그 발췌

<img width="717" height="257" alt="image" src="https://github.com/user-attachments/assets/9db24a1f-121e-4659-a63c-45c907fcf72d" />

<img width="195" height="153" alt="image" src="https://github.com/user-attachments/assets/57301eb4-75e5-4dc0-ae17-d14796eaaf8a" />

## CNN이 무엇을 학습하는지 유추

<img width="195" height="153" alt="image" src="https://github.com/user-attachments/assets/57301eb4-75e5-4dc0-ae17-d14796eaaf8a" />

예측 1

<img width="388" height="392" alt="image" src="https://github.com/user-attachments/assets/4a8fe3d7-c47c-47e7-b8af-ee859fb17eba" />

예측 2

<img width="382" height="554" alt="image" src="https://github.com/user-attachments/assets/1234891c-91b1-493b-a4fe-1a83405adb21" />

예측 3

<img width="385" height="560" alt="image" src="https://github.com/user-attachments/assets/57c82287-5c0d-40d6-9ba2-1fdc2973b1fc" />

예측 4

<img width="390" height="554" alt="image" src="https://github.com/user-attachments/assets/c117eeb4-99ef-4770-a55f-82bb0c376f42" />

예측 5

<img width="408" height="557" alt="image" src="https://github.com/user-attachments/assets/37879ded-7b69-41e2-8c71-570b0a0c3eaf" />

- 모든 예측에서 말과 똑같은 4족 보행 동물입니다.
- 높은 순위일수록 말과 비슷한 외형적 특징을 가집니다.
- 순위가 낮아질수록 말과 비슷한 외형적 특징에서 멀어집니다.

CNN 저수준 특징(4족보행 동물, 색깔)부터 고수준 특징(귀의 방향)까지 선, 형태, 경계등을 학습하여 조건이 가장 많고 비슷한 객체를 더 높이 채첨하는 방식이란걸 유추할 수 있습니다

## 다운스트림 활용: 탐지기 백본으로 사용

이 모델은 이후 **객체 탐지기**의 백본으로 사용됩니다. 분류 head를 제거하고 Stage3/4/5 출력을 객체탐지에 연결합니다.

```
# C3, C4, C5 추출 (128/256/512, stride 8/16/32)
x = model.stem(x)
x = model.stage1(x)
x = model.stage2(x)
c3 = model.stage3(x)
c4 = model.stage4(c3)
c5 = model.stage5(c4)
```
