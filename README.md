# ESS 배터리 수명 예측

초기 100-cycle 정보로 배터리의 최종 Cycle Life를 예측하고, 외부 Batch에서의 일반화 성능을 평가한다.

## 프로젝트 개요

- **데이터:** MIT–Stanford Battery Dataset (Severson et al., 2019)
- **학습·검증:** Batch1(2017-05-12) 46개 → Train34 / Holdout Valid12
- **최종 평가:** Batch2(2018-02-20), Target이 있는 39개 셀
- **태스크:** Regression · 제공된 `cycle_life` 예측. Batch3는 EDA만 수행

## 파일 구조

```text
├── 30-ESSHealth-DAY1.ipynb   # EDA·Feature 계산
├── day2_regression.py       # 모델 개발·평가·보고
├── project_paths.py         # 데이터 경로 설정
├── results/                 # Feature·분할·CV·예측·성능
├── figures/                 # 분석 결과 그림
├── models/                  # 확정 모델
├── requirements.txt
└── README.md
```

## 환경 설정 및 실행

Python 3.12 기준. 라이브러리 버전은 `requirements.txt`에 기록했다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python day2_regression.py --stage final-test
```

이미 평가가 완료되어 위 명령은 **재학습·새 예측 없이 저장된 성능만 조회**한다. 모델·freeze·Test receipt를 함께 유지한다.
`--stage report`는 저장된 예측으로 `reproduced/report/`에 보고서를 생성한다.
EDA 재현 시 원본 MAT를 `data/`에 배치한다. 경로는 `SKALA_BATTERY_DATA_DIR`로 변경할 수 있으며, 저장 결과 조회에는 원본 데이터가 필요 없다.

## EDA

| 분석 | 핵심 발견과 모델링 시사점 |
| --- | --- |
| Cycle Life 분포 | 중앙값은 Batch1 858.5 / Batch2 472 / Batch3 1005.5 cycles. 배치별 분포 차이로 외부 평가가 필요하다. |
| 열화 곡선 | 초기 용량 변화와 전체 수명 열화는 다르다. 일부 후반 열화 가속·Knee 후보는 미래 정보이므로 입력에서 제외한다. |
| ΔQ(V) | `log_var_delta_Q`와 수명의 Spearman ρ는 Batch1 −0.871 / Batch2 −0.709. 핵심 Feature로 유지한다. |
| 충전 속도·정책 | 정책별 평균 수명이 달랐으나 배치별 관계도 달랐다. 충전 설정은 별도 후보로 비교하고 정책 평균 수명은 입력에서 제외한다. |
| 초기 Feature 상관 | 충전시간–수명 상관은 Batch1 +0.613 / Batch2 −0.393. 통합 상관만으로 Feature를 선택하지 않는다. |

계산 과정은 [DAY1 Notebook](30-ESSHealth-DAY1.ipynb), 결과 그림은 `figures/`에 포함한다.

## Modeling

### 피처 엔지니어링 전략

최종 입력은 **8개**다: `log_var_delta_Q`, `mean_QD`, `std_QD`, `mean_IR`, `mean_Tavg`, `mean_Tmax`, `mean_chargetime`, `delta_QD_100_10`.
summary 통계는 Cycle2–100, ΔQ(V)는 Qdlin100−Qdlin10의 분산에 log10을 적용했다.
Target·셀 ID·배치명·전체 수명곡선·Knee는 입력에서 제외하고, 결측 대체와 스케일링은 Train fold 안에서만 fit했다.

### 모델 선택 및 검증

- **후보:** Core1 / Expanded8 / 충전 설정 포함11 × Ridge / RandomForest / GradientBoosting, 총 9개. 모두 log Target 학습 후 exp로 복원해 평가했다.
- **검증:** 셀 단위 Holdout, seed42. Train 내부 outer 5-fold×2 CV와 inner 3-fold 튜닝. 최종 모델은 Train34에만 fit했다.
- **최종:** Expanded8 + RandomForest, 150 trees / max_depth=3 / min_samples_leaf=2. 최소 Valid 후보와 0.5%p 이내에서 CV 허용범위와 입력 수를 고려해 선택했다.
- **평가 통제:** Batch1에서 선택·확정 후 Batch2 predict 1회. Test 이후 재튜닝하지 않았다.

[후보 비교](results/development_comparison.csv) · [분할](results/batch1_split.csv) · [CV 점수](results/nested_cv_folds.csv) · [최종 설정](results/model_freeze.json)

## 성능 결과

| 구분 | MAPE (%) / Gap (%p) | 비고 |
| --- | ---: | --- |
| Train (Batch1 CV) | 9.498 | outer 10개 fold 평균 |
| Valid (Batch1 Holdout) | 8.575 | 12개 셀 |
| Test (Batch2) | 32.828 | 39개 셀 |
| Gap (Train–Valid) | −0.924 | Valid − Train CV |
| Gap (Valid–Test) | +24.253 | Test − Valid; 외부 일반화 저하 |
| Gap (Target–Test) | +23.728 | Test − 논문 참고 Target 9.1%; 목표 미달 |

RMSE(Train/Valid/Test): **90.889 / 100.033 / 169.845 cycles**, R²: **0.273 / 0.767 / 0.400**.
Valid가 CV보다 낮다고 과적합 부재를 증명하지는 않는다. 논문과 정제·분할 조건이 같다고 확인하지 않아 9.1%는 참고값으로 비교한다.
전체 정밀도는 [성능 CSV](results/model_performance.csv)에 보존했다.

## 오류 분석

- Batch2 **30/39개**가 Train 최소 수명 534 cycles보다 짧고, **35/39개를 과대예측**했다. 평균 예측−실제는 **+136.765 cycles**다.
- 최대 오차는 **Batch2_C006**: 실제 393 / 예측 679.768 cycles, APE **72.969%**. 여러 큰 오차 셀에서 Train 범위 밖 Feature가 관찰됐다.
- 수명·입력 분포 차이와 RF의 학습 Target 범위 밖 외삽 한계가 주요 원인 후보다. 인과적 기여도는 확정하지 않았다. 향후 외삽 가능한 회귀와 배치 차이를 고려한 검증을 독립 데이터에서 비교할 필요가 있다.

[저장된 Test 예측](results/test_predictions.csv) · [오류 분석 그림](figures/day2_error_analysis.png)

## ESS 도메인 해석

초기 수명 예측은 셀 선별과 점검·교체 우선순위 판단을 보조할 수 있다. 수명 과대예측은 점검·교체 지연, 과소예측은 조기 교체 비용으로 이어질 수 있어 평균 오차와 함께 오차 방향을 확인해야 한다.

현재 외부 Batch 오차가 커 현장 적용 근거는 부족하다. 셀 결과를 Pack 수명이나 ESS 잔여수명으로 직접 일반화할 수 없으며, 온도·운영 조건·calendar aging·예측 불확실성에 대한 현장 검증이 필요하다.
또한 Holdout은 후보 선택에 사용했고 Valid 10개 충전 정책 중 8개가 Train과 겹친다. Batch2도 DAY1 EDA에서 확인했으므로 완전한 blind test는 아니다. DAY1에서 계획한 원래/log Target 비교는 미수행이다.

## 참고문헌

- Severson et al. (2019). [Data-driven prediction of battery cycle life before capacity degradation](https://www.nature.com/articles/s41560-019-0356-8). *Nature Energy*, 4, 383–391.
- [과제 지정 Kaggle Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)

## 팀 구성

- **박진근:** Feature 설계, 모델링 Pipeline 프로젝트 초기 세팅 및 구현, 성능 평가·통합
- **정지우:** EDA·Feature 분석, 모델링 전략·성능·ESS 해석 검토
