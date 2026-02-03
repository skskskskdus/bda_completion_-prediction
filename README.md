# 데이콘 x BDA 제 2회 학습자 수료 예측 AI 경진대회

## 📋 프로젝트 개요

**대회명**: 데이콘 x BDA 제 2회 학습자 수료 예측 AI 경진대회  
**주제**: 학습자의 수료 여부를 예측하는 AI 알고리즘 개발  
**대회 유형**: 알고리즘 | 월간 데이콘 | 정형 데이터  
**평가 지표**: F1 Score

### 🎯 목표

BDA 9기 학습자 데이터를 분석하여 **10기 학습자의 수료 여부를 예측**하는 AI 알고리즘 개발

### 📖 배경

- **BDA(Big Data Analysis)**: (사)한국빅데이터학회 산하 전국 대학생 연합 빅데이터 학회
- 누적 학회원 수 **6,000명** 이상, 전국 **70개 이상** 대학 네트워크 보유
- Python, SQL 등 프로그래밍부터 데이터 분석까지 체계적인 커리큘럼 제공
- 온라인 교육 특성상 중도 이탈 발생 → **수료 예측을 통한 맞춤형 학습 관리 필요**

### 🔍 문제 정의

학습자의 설문 정보를 바탕으로 교육 과정을 **수료할 학습자를 식별**하는 모델 개발

---

## 📅 대회 일정

| 일정 | 날짜 |
|------|------|
| 대회 시작 | 2025.01.12 |
| 팀 병합 마감 | 2025.02.16 |
| 대회 종료 | 2025.02.23 |
| 코드 제출 마감 | 2025.02.26 |
| 코드 검증 | 2025.03.13 |
| 최종 수상자 발표 | 2025.03.16 |

---

## 📂 프로젝트 구조

```
bda/
├── open/                    # 데이터 폴더
│   ├── train.csv           # 학습 데이터 (BDA 9기)
│   ├── test.csv            # 테스트 데이터 (BDA 10기)
│   └── sample_submission.csv  # 제출 양식
├── baseline.ipynb          # 베이스라인 노트북
├── colab_upload.py         # Colab 파일 업로드 유틸리티
├── upload_widget.py        # ipywidgets 기반 업로드 위젯
└── README.md               # 프로젝트 문서
```

---

## 🚀 시작하기

### 1. 데이터 준비

```python
# Colab에서 데이터 확인
!ls -lh /content/open/
```

### 2. 베이스라인 실행

```python
# baseline.ipynb 실행
# 데이터 로드 → EDA → 모델 학습 → 예측 → 제출
```

---

## 📊 데이터 설명

### 데이터 구조

#### **train.csv** (BDA 9기 학습자 데이터)
- **샘플 수**: 748명
- **특징 수**: 46개 (ID 포함)
- **구성**:
  - **독립변수 (X)**: 학습자 설문 정보 (44개 특징)
    - 학교, 전공, 직무, 수강 분반
    - 유입 경로, BDA 선택 이유
    - 희망 진로, 투입 가능 시간
    - 관심 기업, 희망 도메인 등
  - **종속변수 (y)**: `completed` (수료 여부)
    - `0`: 미수료
    - `1`: 수료

#### **test.csv** (BDA 10기 학습자 데이터)
- **샘플 수**: 814명
- **특징 수**: 45개 (ID 포함, `completed` 제외)
- **구성**: train.csv와 동일한 설문 정보 (수료 여부 없음)

#### **sample_submission.csv** (제출 양식)
- **ID**: 학습자 고유 ID
- **completed**: 예측할 수료 여부 (0 또는 1)

### 데이터 컬럼 상세 설명

#### 기본 정보
- **ID**: 샘플별 고유 ID
- **generation**: BDA 기수
- **school1**: 대학교
- **major type**: 복수전공 여부
- **major1_1**: 제1전공
- **major1_2**: 제2전공
- **major_data**: 제1전공 전공자 여부
- **job**: 현재 직무

#### 수강 정보
- **class1~4**: 수강 분반
- **re_registration**: 학기당 새로운 학회원을 모집할 때 재등록 여부
- **previous_class_3~9**: 각 기수를 수강했을 시 분반

#### 학습자 배경
- **contest_award**: 공모전 수상 경력
- **nationality**: 내/외국인 여부
- **inflow_route**: 유입 경로
- **major_field**: 전공 분야
- **completed_semester**: 대학교 이수학기
- **certificate_acquisition**: 취득한 자격증

#### 학습 동기 및 목표
- **whyBDA**: BDA를 선택한 이유
- **what_to_gain**: BDA에서 얻고싶은 것
- **desired_career_path**: 희망 진로
- **desired_job**: 희망 직무
- **desired_certificate**: 취득을 희망하는 자격증
- **desired_job_except_data**: 데이터 외 희망 직무
- **interested_company**: 관심있는 기업명
- **expected_domain**: 희망하는 도메인

#### 학습 참여 의향
- **hope_for_group**: 조별활동 희망 여부
- **project_type**: 팀/개인 중 프로젝트에 참여하고 싶은 형태
- **time_input**: 하루에 BDA에 투입 가능한 시간
- **contest_participation**: 데이터 관련 대회 경험
- **idea_contest**: 아이디어 공모전에 대한 경험
- **onedayclass_topic**: 원데이 클래스 주제

#### 현직자 강연 선호도
- **incumbents_level**: 어느 정도 연차의 현직자를 원하는지
- **incumbents_lecture**: 어떤 주제의 현직자 강의를 원하는지
- **incumbents_company_level**: 강연 현직자가 어느정도 규모의 회사를 다니는 사람이었으면 좋겠는지
- **incumbents_lecture_type**: 온, 오프라인 중 원하는 현직자 강연 형태
- **incumbents_lecture_scale**: 원하는 현직자 강의 규모
- **incumbents_lecture_scale_reason**: 현직자 강의 규모 선택 이유

#### 타겟 변수
- **completed**: **(TARGET)** 수료 여부
  - `0`: 미수료
  - `1`: 수료
  - ⚠️ **중요**: 본 대회는 중도 탈퇴 예측이 아닌, 학습 과정을 끝까지 완료하여 **'수료'에 도달한 학습자를 예측**하는 것이 목표입니다.

> **참고**: `test.csv`는 `completed` 컬럼을 제외하고 `train.csv`와 동일한 구성입니다.

---

## 🤖 모델링 방법론

### 핵심 개념

이 대회는 **9기 학습자의 패턴을 학습하여 10기 학습자의 수료 여부를 예측**하는 문제입니다.

```
학습 단계 (Training)
├─ 데이터: train.csv (9기 학습자)
├─ 독립변수 (X): 9기 학습자의 설문 정보
├─ 종속변수 (y): 9기 학습자의 실제 수료 여부
└─ 목표: "어떤 특성을 가진 학습자가 수료하는가?" 패턴 학습

예측 단계 (Prediction)
├─ 데이터: test.csv (10기 학습자)
├─ 독립변수 (X): 10기 학습자의 설문 정보
├─ 종속변수 (y): 10기 학습자의 수료 여부 (예측)
└─ 목표: 학습한 패턴을 10기 학습자에게 적용
```

### 가정

> **"9기 학습자와 10기 학습자는 비슷한 특성을 가지고 있다"**
> 
> "9기에서 수료한 학습자의 특성을 가진 10기 학습자도 수료할 가능성이 높다"

### 예시

**9기 데이터로 학습:**
```
학생 A: 컴공 전공, 하루 3시간 투입, 네이버 관심 → 수료 ✅
학생 B: 경영 전공, 하루 1시간 투입, 관심사 없음 → 미수료 ❌
```

**모델이 학습한 패턴:**
```
"IT 전공 + 투입 시간 많음 + 명확한 관심사 → 수료 확률 높음"
"비전공 + 투입 시간 적음 + 불명확한 목표 → 수료 확률 낮음"
```

**10기 학습자 예측:**
```
학생 C: 컴공 전공, 하루 3시간 투입, 카카오 관심
→ 모델 예측: 수료 (1) ✅
```

---

## 🏆 평가 방식

**평가 지표**: F1 Score

```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 🔗 참고 링크

- [대회 페이지](https://dacon.io/competitions/official/236664/overview/description)
- [BDA 홈페이지](https://bdaprogram.oopy.io/)
- [BDA 소개영상](https://youtu.be/t1EMrx_rL0Y?si=eQnXxPWDHVn2t1Nz)

---

## 👥 주최 / 주관

- **주최**: 빅데이터분석학회 (BDA)
- **주관**: 데이콘 (DACON)

---

## 📝 License

이 프로젝트는 데이콘 대회 규정을 따릅니다.

---

## 🤖 사용된 모델 및 라이브러리

### KoE5 (Korean E5 Embedding Model)

본 프로젝트에서는 **텍스트 임베딩 및 모델 학습**을 위해 한국어 임베딩 모델 **KoE5**를 사용했습니다.

- **모델**: [nlpai-lab/KoE5](https://huggingface.co/nlpai-lab/KoE5)
- **GitHub**: [nlpai-lab/KURE](https://github.com/nlpai-lab/KURE)
- **라이선스**: MIT License

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nlpai-lab/KoE5")
embeddings = model.encode(sentences)
```

