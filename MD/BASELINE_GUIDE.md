# BDA 학습자 수료 예측 - 베이스라인 가이드

## 📋 개요

이 베이스라인 코드는 BDA 학습자 수료 예측 대회를 위한 전체 파이프라인을 제공합니다.

---

## 🚀 실행 방법

### 1. 데이터 업로드
`baseline.ipynb`의 파일 업로드 셀을 실행하여 데이터를 `/content/open/` 폴더에 업로드하세요.

### 2. 순차적 실행
노트북의 셀을 위에서부터 순차적으로 실행하세요.

---

## 📊 파이프라인 구조

### 1️⃣ **라이브러리 임포트**
- pandas, numpy, matplotlib, seaborn
- scikit-learn (전처리, 모델, 평가)

### 2️⃣ **데이터 로드**
- `train.csv`: BDA 9기 학습자 데이터
- `test.csv`: BDA 10기 학습자 데이터
- `sample_submission.csv`: 제출 양식

### 3️⃣ **EDA (탐색적 데이터 분석)**
- 데이터 기본 정보 확인
- 결측치 확인
- 타겟 변수 분포 확인
- 시각화:
  - 타겟 변수 분포 (막대 그래프, 파이 차트)
  - 수치형 변수 분포
  - 상관관계 히트맵

### 4️⃣ **데이터 전처리**
- **타겟 변수 분리**: `y_train`, `X_train`
- **ID 컬럼 제거**: 모델 학습에 불필요
- **변수 타입 분리**: 범주형 vs 수치형
- **결측치 처리**:
  - 수치형: 중앙값으로 대체
  - 범주형: 최빈값으로 대체
- **범주형 인코딩**: LabelEncoder 사용

### 5️⃣ **특징 추출 (Feature Engineering)**
- **통계 특징**:
  - `numeric_mean`: 수치형 변수 평균
  - `numeric_std`: 수치형 변수 표준편차
  - `numeric_max`: 수치형 변수 최댓값
  - `numeric_min`: 수치형 변수 최솟값
- **상호작용 특징**:
  - 변수 간 곱셈
  - 변수 간 나눗셈

### 6️⃣ **데이터 스케일링**
- StandardScaler 사용
- Train과 Test에 동일하게 적용

### 7️⃣ **모델 학습**
- **Train/Validation 분할**: 80:20
- **모델 1**: Random Forest
  - n_estimators=100
  - max_depth=10
- **모델 2**: Gradient Boosting
  - n_estimators=100
  - max_depth=5
  - learning_rate=0.1
- **평가**: F1 Score (weighted)
- **최고 성능 모델 선택**

### 8️⃣ **모델 평가**
- Classification Report
- Confusion Matrix
- 특징 중요도 분석

### 9️⃣ **예측 및 제출**
- 전체 Train 데이터로 재학습
- Test 데이터 예측
- `submission.csv` 생성

---

## 🎯 개선 아이디어

### 데이터 전처리
- [ ] 이상치 탐지 및 처리
- [ ] 범주형 변수 One-Hot Encoding
- [ ] 수치형 변수 로그 변환

### 특징 추출
- [ ] 도메인 지식 기반 특징 생성
- [ ] 다항식 특징 (Polynomial Features)
- [ ] 변수 간 비율 특징

### 모델링
- [ ] XGBoost, LightGBM, CatBoost 시도
- [ ] 하이퍼파라미터 튜닝 (GridSearchCV, Optuna)
- [ ] 앙상블 (Voting, Stacking)
- [ ] 교차 검증 (K-Fold Cross Validation)

### 평가
- [ ] Stratified K-Fold 사용
- [ ] 다양한 평가 지표 확인 (Precision, Recall, AUC)

---

## 📈 예상 결과

- **Validation F1 Score**: 0.70 ~ 0.85 (데이터에 따라 다름)
- **제출 파일**: `/content/open/submission.csv`

---

## 💡 팁

1. **데이터 이해**: EDA를 통해 데이터의 특성을 충분히 파악하세요
2. **특징 중요도**: 중요한 특징을 파악하여 추가 특징 생성에 활용하세요
3. **하이퍼파라미터**: 모델 성능을 높이기 위해 튜닝하세요
4. **앙상블**: 여러 모델을 결합하여 성능을 향상시키세요
5. **교차 검증**: 과적합을 방지하고 일반화 성능을 높이세요

---

## 🔗 참고 자료

- [scikit-learn 공식 문서](https://scikit-learn.org/)
- [Pandas 공식 문서](https://pandas.pydata.org/)
- [대회 페이지](https://dacon.io/competitions/official/236664/overview/description)

---

## 📝 파일 구조

```
bda/
├── open/
│   ├── train.csv              # 학습 데이터
│   ├── test.csv               # 테스트 데이터
│   ├── sample_submission.csv  # 제출 양식
│   └── submission.csv         # 생성된 제출 파일
├── baseline.ipynb             # 베이스라인 노트북 (실행용)
├── baseline_code.py           # 베이스라인 코드 (참고용)
└── BASELINE_GUIDE.md          # 이 파일
```

---

**Good Luck! 🍀**
