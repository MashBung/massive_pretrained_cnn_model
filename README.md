# massive_pretrained_cnn_model

객체 탐지의 백본으로 사용하기 위해 CNN을 PyTorch로 직접 설계하고, 500-class 커스텀 ImageNet-style 데이터셋(약 30만 장)으로 처음부터(from scratch) 사전학습한 프로젝트와 모델입니다.

torchvision의 pretrained weight를 가져다 쓰는 대신, 백본 구조 설계, 데이터 파이프라인, 학습 병목등 전 과정을 구현했습니다.

