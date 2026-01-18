# K리그 패스 도착점 예측 대회

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-1.0+-yellow.svg)

## 대회 개요

| 항목 | 내용 |
|------|------|
| **대회명** | K리그 이벤트 데이터 기반 패스 도착점 예측 |
| **목표** | 에피소드의 마지막 패스가 도착하는 (x, y) 좌표 예측 |
| **평가지표** | Mean Euclidean Distance (유클리드 거리) |
| **최종 점수** | **13.20** |

## 핵심 전략

### 1. 분위수 회귀 (Quantile Regression)
- **문제**: 롱패스(50m+)가 전체의 38%를 차지하며 에러의 주범
- **해결**: RMSE(평균 예측) 대신 **분위수 회귀(q=0.5, 중앙값)** 사용
- **효과**: CV 13.51 → 13.24 (약 **0.27점 개선**)

### 2. Y축 대칭 증강 (Data Augmentation)
- 축구장은 Y축 기준 대칭
- Y 관련 피처를 반전시켜 데이터 **2배 증강**
- Y좌표: `68 - y`, 각도: `-angle`, dy: `-dy`

### 3. 다중 모델 앙상블
- **LightGBM** + **CatBoost** 분위수 회귀
- 최적 앙상블 비율: LGB 0.4 + CAT 0.6

### 4. GroupKFold CV
- `game_id` 기준 분할로 일반화 성능 향상
- CV-LB 괴리 최소화 (13.21 vs 13.20)

## 프로젝트 구조

```
├── 01_EDA.ipynb                 # 탐색적 데이터 분석
├── 02_Preprocessing.ipynb       # 전처리 & 피처 엔지니어링
├── 03_Modeling.ipynb            # 모델링 & 앙상블
├── README.md                    # 프로젝트 설명
└── submission.csv               # 최종 제출 파일
```

## 노트북 설명

### 1. EDA (`01_EDA.ipynb`)

데이터의 특성을 파악하고 모델링 전략을 수립합니다.

**주요 내용:**
- 데이터 기본 탐색 (필드 크기: 105m x 68m)
- 타겟 분포 분석 (Y축 대칭성 발견)
- 패스 거리별 에러 분석 (롱패스가 에러 주범)
- 이벤트 시퀀스 분석 (Carry 후 롱패스 비율 높음)
- 구역별 패스 패턴 분석

**핵심 발견:**
| 발견 | 설명 | 적용 |
|------|------|------|
| Y축 대칭 | end_y가 Y=34 기준 대칭 분포 | Y축 증강 적용 |
| 롱패스 문제 | 50m+ 패스가 38%, 에러 15m+ | 분위수 회귀 적용 |
| Carry 후 롱패스 | Carry 후 롱패스 비율 13.7%p 높음 | 시퀀스 피처 생성 |

### 2. 전처리 (`02_Preprocessing.ipynb`)

EDA 인사이트를 바탕으로 피처를 생성합니다.

**주요 내용:**
- 기본 피처 생성 (시간, 이동, 위치)
- 시퀀스 피처 생성 (마지막 K개 이벤트)
- Wide Format 변환
- 피처 선택 (상위 8%)
- Y축 대칭 증강 함수 정의

**생성된 피처:**
```
- 위치: start_x, start_y, x_zone, lane
- 이동: dx, dy, dist, speed, angle
- 골대: to_goal_dist, to_goal_angle
- 시퀀스: ep_idx_norm, is_final_team
- 범주형: type_id, res_id, is_home
```

**타겟 누출 방지:**
- 마지막 이벤트의 `end_x`, `end_y`는 타겟이므로 피처에서 제외
- 관련 파생 피처(`dx`, `dy`, `dist` 등)도 마스킹 처리

### 3. 모델링 (`03_Modeling.ipynb`)

분위수 회귀와 앙상블을 활용한 최종 모델입니다.

**주요 내용:**
- 분위수 회귀 학습 (LightGBM + CatBoost)
- 앙상블 가중치 최적화
- 제출 파일 생성

**모델 파라미터:**
```python
# LightGBM 분위수 회귀
lgb_params = {
    'objective': 'quantile',
    'alpha': 0.5,  # 중앙값
    'learning_rate': 0.03,
    'num_leaves': 127,
    'min_data_in_leaf': 80,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8
}

# CatBoost 분위수 회귀
loss_function = 'Quantile:alpha=0.5'
```

**앙상블:**
- LightGBM: 40%
- CatBoost: 60%

## 성능 개선 히스토리

| 버전 | 방법 | CV | 제출 점수 |
|------|------|-----|----------|
| v1 | 베이스라인 (XGBoost) | 14.2+ | - |
| v2 | GroupKFold 적용 | 13.8+ | 13.8+ |
| v3 | 타겟 누출 제거 | 13.6+ | 13.6+ |
| v4 | 3모델 앙상블 | 13.51 | 13.53 |
| v5 | Y축 증강 | 13.51 | 13.53 |
| **v6** | **분위수 회귀** | **13.21** | **13.20** |

## 실행 방법

### 환경 설정
```bash
pip install numpy pandas scikit-learn lightgbm catboost matplotlib seaborn
```

### 실행 순서
```bash
# 1. EDA
jupyter notebook 01_EDA.ipynb

# 2. 전처리
jupyter notebook 02_Preprocessing.ipynb

# 3. 모델링 (최종 제출 파일 생성)
jupyter notebook 03_Modeling.ipynb
```

## 참고 자료

- [LightGBM Quantile Regression](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [CatBoost Quantile Loss](https://catboost.ai/docs/concepts/loss-functions-regression.html)
- [SoccerMap: Deep Learning for Pass Prediction](https://arxiv.org/abs/2010.10202)

## 향후 개선 방향

1. **Neural Network**: LSTM/Transformer로 시퀀스 모델링
2. **Pseudo Labeling**: 테스트 예측값으로 재학습
3. **다중 분위수 앙상블**: q=0.4, 0.5, 0.6 등 여러 분위수 조합
4. **구역별 모델**: 시작 위치에 따른 개별 모델 학습

## License

This project is licensed under the MIT License.

---

Made with ⚽ for K-League Data Challenge
