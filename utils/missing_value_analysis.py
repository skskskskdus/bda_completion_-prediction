# ============================================================
# 결측치 분석 및 처리 전략
# ============================================================

import pandas as pd
import numpy as np

# 데이터 로드 (예시)
# train = pd.read_csv('/content/open/train.csv')

# ============================================================
# 1. 결측치 확인 및 분석
# ============================================================

print("="*60)
print("📊 결측치 분석")
print("="*60)

# 결측치 개수 및 비율
missing_df = pd.DataFrame({
    '결측치 개수': train.isnull().sum(),
    '결측치 비율(%)': (train.isnull().sum() / len(train) * 100).round(2)
})
missing_df = missing_df[missing_df['결측치 개수'] > 0].sort_values('결측치 개수', ascending=False)

print("\n[결측치가 있는 컬럼]")
print(missing_df)

# ============================================================
# 2. 각 컬럼별 고유값 확인
# ============================================================

print("\n" + "="*60)
print("🔍 각 컬럼별 고유값 분석")
print("="*60)

# 결측치가 많은 컬럼 우선 분석
high_missing_cols = missing_df[missing_df['결측치 비율(%)'] > 50].index.tolist()

print("\n[결측치 50% 이상 컬럼 상세 분석]")
for col in high_missing_cols:
    print(f"\n📌 {col}")
    print(f"  - 결측치: {train[col].isnull().sum()}개 ({train[col].isnull().sum()/len(train)*100:.1f}%)")
    print(f"  - 고유값 개수: {train[col].nunique()}")
    if train[col].dtype == 'object':
        print(f"  - 상위 5개 값:")
        print(train[col].value_counts().head())
    else:
        print(f"  - 통계:")
        print(train[col].describe())

# 결측치가 적은 컬럼 분석
low_missing_cols = missing_df[(missing_df['결측치 비율(%)'] > 0) & (missing_df['결측치 비율(%)'] <= 50)].index.tolist()

print("\n" + "="*60)
print("\n[결측치 50% 이하 컬럼 상세 분석]")
for col in low_missing_cols:
    print(f"\n📌 {col}")
    print(f"  - 결측치: {train[col].isnull().sum()}개 ({train[col].isnull().sum()/len(train)*100:.1f}%)")
    print(f"  - 고유값 개수: {train[col].nunique()}")
    print(f"  - 데이터 타입: {train[col].dtype}")
    
    if train[col].dtype == 'object':
        print(f"  - 상위 5개 값:")
        print(train[col].value_counts().head())
    else:
        print(f"  - 통계:")
        print(train[col].describe())

# ============================================================
# 3. 결측치 처리 전략
# ============================================================

print("\n" + "="*60)
print("💡 컬럼별 결측치 처리 전략")
print("="*60)

# 결측치 처리 전략 정리
treatment_strategy = {
    # 결측치 90% 이상 - 삭제 고려
    'contest_award': {
        '결측치 비율': '100%',
        '처리 방법': '❌ 컬럼 삭제',
        '이유': '모든 값이 결측치로 정보 없음'
    },
    'idea_contest': {
        '결측치 비율': '100%',
        '처리 방법': '❌ 컬럼 삭제',
        '이유': '모든 값이 결측치로 정보 없음'
    },
    
    # 결측치 80% 이상 - 삭제 또는 특수 처리
    'class3': {
        '결측치 비율': '98.1%',
        '처리 방법': '❌ 컬럼 삭제 또는 ⚠️ "미수강" 카테고리 생성',
        '이유': '대부분 결측치, 수강하지 않은 것으로 해석 가능'
    },
    'class4': {
        '결측치 비율': '99.9%',
        '처리 방법': '❌ 컬럼 삭제',
        '이유': '거의 모든 값이 결측치'
    },
    'previous_class_3~8': {
        '결측치 비율': '80.5%',
        '처리 방법': '⚠️ "미수강" 또는 "신규학생" 카테고리 생성',
        '이유': '이전 기수를 수강하지 않은 신규 학생으로 해석'
    },
    'contest_participation': {
        '결측치 비율': '99.2%',
        '처리 방법': '⚠️ "경험없음" 카테고리로 대체',
        '이유': '대회 경험이 없는 것으로 해석'
    },
    
    # 결측치 50~80% - 특수 처리
    'class2': {
        '결측치 비율': '77.4%',
        '처리 방법': '⚠️ "미수강" 카테고리 생성 또는 중앙값 대체',
        '이유': '2학기 미수강 학생'
    },
    'major1_2': {
        '결측치 비율': '58.7%',
        '처리 방법': '⚠️ "단일전공" 또는 "없음" 카테고리',
        '이유': '복수전공이 없는 학생'
    },
    
    # 결측치 5% 이하 - 일반적 대체
    'completed_semester': {
        '결측치 비율': '3.7%',
        '처리 방법': '✅ 중앙값 또는 평균값 대체',
        '이유': '수치형 변수, 결측치 비율 낮음'
    },
    'major_field': {
        '결측치 비율': '3.1%',
        '처리 방법': '✅ 최빈값 또는 "기타" 카테고리',
        '이유': '범주형 변수, 결측치 비율 낮음'
    },
    'major type': {
        '결측치 비율': '2.9%',
        '처리 방법': '✅ 최빈값 대체',
        '이유': '범주형 변수, 결측치 비율 낮음'
    },
    'major1_1': {
        '결측치 비율': '2.7%',
        '처리 방법': '✅ 최빈값 또는 "미정" 카테고리',
        '이유': '범주형 변수, 결측치 비율 낮음'
    },
    'nationality': {
        '결측치 비율': '0.1%',
        '처리 방법': '✅ 최빈값 대체',
        '이유': '결측치 1개, 최빈값으로 대체'
    }
}

# 전략 출력
for col, strategy in treatment_strategy.items():
    print(f"\n📌 {col}")
    for key, value in strategy.items():
        print(f"  {key}: {value}")


# ============================================================
# 4. 실제 결측치 처리 코드 (예시)
# ============================================================

print("\n" + "="*60)
print("🔧 결측치 처리 실행")
print("="*60)

# 복사본 생성
train_processed = train.copy()

# 1) 100% 결측치 컬럼 삭제
cols_to_drop = ['contest_award', 'idea_contest']
train_processed = train_processed.drop(columns=cols_to_drop)
print(f"\n✅ 삭제된 컬럼: {cols_to_drop}")

# 2) 고결측치 컬럼 - 특수 카테고리 생성
# class2, class3, class4 -> "미수강" 처리
for col in ['class2', 'class3', 'class4']:
    if col in train_processed.columns:
        train_processed[col] = train_processed[col].fillna(-1)  # -1 = 미수강
        print(f"✅ {col}: 결측치를 -1(미수강)로 대체")

# previous_class 컬럼들 -> "신규학생" 처리
prev_class_cols = [col for col in train_processed.columns if col.startswith('previous_class')]
for col in prev_class_cols:
    train_processed[col] = train_processed[col].fillna('신규학생')
    print(f"✅ {col}: 결측치를 '신규학생'으로 대체")

# contest_participation -> "경험없음"
if 'contest_participation' in train_processed.columns:
    train_processed['contest_participation'] = train_processed['contest_participation'].fillna('경험없음')
    print(f"✅ contest_participation: 결측치를 '경험없음'으로 대체")

# major1_2 -> "단일전공"
if 'major1_2' in train_processed.columns:
    train_processed['major1_2'] = train_processed['major1_2'].fillna('단일전공')
    print(f"✅ major1_2: 결측치를 '단일전공'으로 대체")

# 3) 저결측치 수치형 컬럼 - 중앙값 대체
if 'completed_semester' in train_processed.columns:
    median_val = train_processed['completed_semester'].median()
    train_processed['completed_semester'] = train_processed['completed_semester'].fillna(median_val)
    print(f"✅ completed_semester: 결측치를 중앙값({median_val})으로 대체")

# 4) 저결측치 범주형 컬럼 - 최빈값 대체
categorical_low_missing = ['major type', 'major1_1', 'major_field', 'nationality']
for col in categorical_low_missing:
    if col in train_processed.columns and train_processed[col].isnull().sum() > 0:
        mode_val = train_processed[col].mode()[0]
        train_processed[col] = train_processed[col].fillna(mode_val)
        print(f"✅ {col}: 결측치를 최빈값('{mode_val}')으로 대체")

# 최종 결측치 확인
print("\n" + "="*60)
print("📊 처리 후 결측치 확인")
print("="*60)
remaining_missing = train_processed.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if len(remaining_missing) == 0:
    print("✅ 모든 결측치가 처리되었습니다!")
else:
    print("⚠️ 남은 결측치:")
    print(remaining_missing)

print(f"\n최종 데이터 shape: {train_processed.shape}")
