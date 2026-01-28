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

- **train.csv**: BDA 9기 학습자 데이터 (학습용)
- **test.csv**: BDA 10기 학습자 데이터 (예측용)
- **sample_submission.csv**: 제출 파일 양식

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
