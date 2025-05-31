# 2025 월간 데이콘 건설용 자갈 암석 종류 분류 AI 경진 대회

public, private 리더보드에서 양호한 점수를 기록하였습니다.

- publice LB score(#4) : 0.90859
- private LB score(#4) : 0.91245

## 1. 환경 설정
### 1.1 저장소 코드 복사

```bash
git clone https://github.com/ujinyang/dacon2025_rock_image_classification.git
cd dacon2025_rock_image_classification
```

### 1.2 패키지 설치

wand (ImageMagicK) 의 경우, 추가 라이브러리를 설치하여야 합니다.

```bash
sudo apt install -y unzip imagemagick libopencv-dev libmagickwand-dev
pip install -r requirements.txt
```

## 2. 폴더 구조

```bash
├── train
│   ├── Andesite
│   │   ├── TRAIN_00000.jpg
│   │   ├── TRAIN_00000.jpg
│   │   ├── TRAIN_00001.jpg
│   ├── ...
│   └── Etc
│       └── TRAIN_xxxxx.jpg
├── test
│   ├── TEST_00000.jpg
│   ├── TEST_00001.jpg
│   ├── ...
│   └── TEST_xxxxx.jpg
├── sample_submission.csv
├── train.csv
├── test.csv
├── basslibrary_model_submit.ipynb
└── ckpt
```

## 3. 모델 훈련

※ 노트북 파일에 실행결과가 포함되어 있지 않은 대신, 별도의 출력로그를 첨부하였습니다.

### 3.1 eva02_base 모델

모델의 학습 파라메터는 아래처럼 설정되어 학습하였습니다.

```bash
CFG = {}
CFG['OVERSAMPLING'] = True
CFG['GRADIENT_CHECKPOINT'] = False
CFG['ACCUMULATION_STEPS'] = 1
CFG['MAX_CLASSES'] = 7
CFG['SEED'] = 42
CFG['N_SPLIT'] = 5
CFG['LABEL_SMOOTHING'] = 0.05
CFG['RESIZE_TO_FIT'] = False
CFG['OPTIMIZER'] = 'AdamW'
CFG['INTERPOLATION'] = 'robidouxsharp'
CFG['PRECISION'] = '16'
CFG['MODEL_NAME'] = "timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
CFG['IMG_SIZE'] = 448
CFG['BATCH_SIZE'] = 48 if CFG['GRADIENT_CHECKPOINT'] else 16
CFG['LR'] = [ 1e-5 / float(8) * CFG['BATCH_SIZE'] * CFG['ACCUMULATION_STEPS'], 1e-6  / float(8) * CFG['BATCH_SIZE'] * CFG['ACCUMULATION_STEPS']] # 1e-6 => 1e-5
```

출력로그 : [logs/model_train.log](https://github.com/ujinyang/dacon2025_rock_image_classification/blob/main/logs/model_train.log)


## 4. 모델 결과 제출

※ 노트북 파일에 실행결과가 포함되어 있지 않은 대신, 별도의 출력로그를 첨부합니다.

출력로그 : [logs/model_submit.log](https://github.com/ujinyang/dacon2025_rock_image_classification/blob/main/logs/model_submit.log)


### 4.1 모델 앙상블

이미지가 많고 훈련시간이 길어, crossvalidation 학습 및 앙상블 학습을 하지 못했습니다.


### 4.2 모델 체크포인트
당분간 체크포인트 파일을 다운로드 하실 수 있도록 구글 드라이브에 올렸습니다.

|No.| model_name (google_drive link)                                                                               |fold|epoch|val_loss|val_score|
|---|--------------------------------------------------------------------------------------------------------------|----|-----|--------|---------|
| 1 | [eva02_base_patch14_448.mim_in22k_ft_in22k_in1k-ema](https://drive.google.com/file/d/1vcxHeH-2obONtzo1aJahC9c-IAy9-qEU/view?usp=drive_link) |  0 |  13 | 0.4401 |  0.9150 |
| 2 | [eva02_base_patch14_448.mim_in22k_ft_in22k_in1k](https://drive.google.com/file/d/1tb1mmKRGBsLIGNskeTU-CGh3v69pqD9C/view?usp=drive_link)     |  0 |  12 | 0.4380 |  0.9155 |

