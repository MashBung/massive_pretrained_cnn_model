# massive_pretrained_cnn_model

객체 탐지의 백본으로 사용하기 위해 CNN을 PyTorch로 직접 설계하고, 500-class 커스텀 ImageNet-style 데이터셋(약 30만 장)으로 처음부터(from scratch) 사전학습한 프로젝트와 모델입니다.

torchvision의 pretrained weight를 가져다 쓰는 대신, 백본 구조 설계, 데이터 파이프라인, 학습 병목등 전 과정을 구현했습니다.

<img width="737" height="344" alt="cnn_log" src="https://github.com/user-attachments/assets/5b54fca1-ee3e-47cf-980e-ad723a155eb2" />

<img width="570" height="379" alt="cnn_log1" src="https://github.com/user-attachments/assets/e665fa95-2b8f-43f2-b488-dbebdd0390d8" />

<img width="491" height="329" alt="cnn_log2" src="https://github.com/user-attachments/assets/48b45407-1d58-446c-baf7-c2adc937b1d3" />

<img width="197" height="149" alt="cnn_result" src="https://github.com/user-attachments/assets/8f1d34c8-ba6a-4758-b7e9-2db3d1e5e447" />


핵심결과

훈련 정확도 85.37

검증 정확도 68.05

클래스 수 500

입력 해상도 320*320

학습 환경 5070ti


