"""
BDA 학습자 수료 예측 - 베이스라인 코드
데이터 로드 → EDA → 전처리 → 특징 추출 → 모델 학습 → 예측 → 제출
"""

# ============================================================
# 1. 라이브러리 임포트
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Colab)
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

print("✅ 라이브러리 임포트 완료!")


# ============================================================
# 2. 데이터 로드
# ============================================================
# 데이터 경로 설정
DATA_PATH = '/content/open/'

# 데이터 로드
train = pd.read_csv(DATA_PATH + 'train.csv')
test = pd.read_csv(DATA_PATH + 'test.csv')
submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')

print("✅ 데이터 로드 완료!")
print(f"📊 Train 데이터: {train.shape}")
print(f"📊 Test 데이터: {test.shape}")
print(f"📊 Submission: {submission.shape}")


# ============================================================
# 3. 데이터 확인 (EDA)
# ============================================================
print("\n" + "="*60)
print("📋 데이터 기본 정보")
print("="*60)

# Train 데이터 정보
print("\n[Train 데이터 정보]")
print(train.info())

print("\n[Train 데이터 샘플]")
print(train.head())

print("\n[Train 데이터 통계]")
print(train.describe())

# 결측치 확인
print("\n[결측치 확인]")
print("Train 결측치:")
print(train.isnull().sum())
print(f"\nTest 결측치:")
print(test.isnull().sum())

# 타겟 변수 분포 확인
print("\n[타겟 변수 분포]")
if 'completion' in train.columns:
    target_col = 'completion'
elif 'target' in train.columns:
    target_col = 'target'
else:
    # 타겟 컬럼 찾기
    target_col = train.columns[-1]

print(f"타겟 컬럼: {target_col}")
print(train[target_col].value_counts())
print(f"\n수료율: {train[target_col].mean()*100:.2f}%")


# ============================================================
# 4. 시각화 (EDA)
# ============================================================
print("\n" + "="*60)
print("📊 데이터 시각화")
print("="*60)

# 타겟 변수 분포
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
train[target_col].value_counts().plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
plt.title('타겟 변수 분포')
plt.xlabel('수료 여부')
plt.ylabel('빈도')
plt.xticks(rotation=0)

plt.subplot(1, 3, 2)
train[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
plt.title('타겟 변수 비율')
plt.ylabel('')

# 수치형 변수 분포
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

if len(numeric_cols) > 0:
    plt.subplot(1, 3, 3)
    train[numeric_cols[0]].hist(bins=30, color='#95E1D3', edgecolor='black')
    plt.title(f'{numeric_cols[0]} 분포')
    plt.xlabel(numeric_cols[0])
    plt.ylabel('빈도')

plt.tight_layout()
plt.show()

# 상관관계 히트맵
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    correlation = train[numeric_cols + [target_col]].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('변수 간 상관관계')
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. 데이터 전처리
# ============================================================
print("\n" + "="*60)
print("🔧 데이터 전처리")
print("="*60)

# ID 컬럼 저장 및 제거
train_id = train['ID'] if 'ID' in train.columns else (train['id'] if 'id' in train.columns else None)
test_id = test['ID'] if 'ID' in test.columns else (test['id'] if 'id' in test.columns else None)

# 타겟 변수 분리
y_train = train[target_col]
X_train = train.drop(columns=[target_col])

# ID 컬럼 제거
if 'ID' in X_train.columns:
    X_train = X_train.drop(columns=['ID'])
elif 'id' in X_train.columns:
    X_train = X_train.drop(columns=['id'])

if 'ID' in X_test.columns:
    X_test = X_test.drop(columns=['ID'])
elif 'id' in X_test.columns:
    X_test = X_test.drop(columns=['id'])
else:
    X_test = test.copy()

print(f"✅ X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"✅ X_test: {X_test.shape}")

# 범주형 변수와 수치형 변수 분리
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n📊 범주형 변수 ({len(categorical_cols)}개): {categorical_cols}")
print(f"📊 수치형 변수 ({len(numeric_cols)}개): {numeric_cols}")

# 결측치 처리
print("\n[결측치 처리]")
# 수치형 변수: 중앙값으로 대체
for col in numeric_cols:
    if X_train[col].isnull().sum() > 0:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)
        print(f"  - {col}: 중앙값({median_val})으로 대체")

# 범주형 변수: 최빈값으로 대체
for col in categorical_cols:
    if X_train[col].isnull().sum() > 0:
        mode_val = X_train[col].mode()[0]
        X_train[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)
        print(f"  - {col}: 최빈값({mode_val})으로 대체")

print("✅ 결측치 처리 완료!")

# 범주형 변수 인코딩
print("\n[범주형 변수 인코딩]")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Train + Test 합쳐서 fit
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined)
    
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le
    print(f"  - {col}: {len(le.classes_)}개 클래스 인코딩")

print("✅ 인코딩 완료!")


# ============================================================
# 6. 특징 추출 (Feature Engineering)
# ============================================================
print("\n" + "="*60)
print("🎯 특징 추출 (Feature Engineering)")
print("="*60)

# 특징 추출 함수
def create_features(df):
    """특징 추출 함수"""
    df_new = df.copy()
    
    # 1. 수치형 변수 통계 특징
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        df_new['numeric_mean'] = df_new[numeric_cols].mean(axis=1)
        df_new['numeric_std'] = df_new[numeric_cols].std(axis=1)
        df_new['numeric_max'] = df_new[numeric_cols].max(axis=1)
        df_new['numeric_min'] = df_new[numeric_cols].min(axis=1)
        print("  ✅ 수치형 통계 특징 생성")
    
    # 2. 변수 간 상호작용 특징 (예시)
    # 실제 데이터에 맞게 수정 필요
    if len(numeric_cols) >= 2:
        df_new[f'{numeric_cols[0]}_x_{numeric_cols[1]}'] = df_new[numeric_cols[0]] * df_new[numeric_cols[1]]
        df_new[f'{numeric_cols[0]}_div_{numeric_cols[1]}'] = df_new[numeric_cols[0]] / (df_new[numeric_cols[1]] + 1)
        print("  ✅ 상호작용 특징 생성")
    
    return df_new

# 특징 추출 적용
X_train_fe = create_features(X_train)
X_test_fe = create_features(X_test)

print(f"\n✅ 특징 추출 완료!")
print(f"📊 최종 Train 특징 수: {X_train_fe.shape[1]}")
print(f"📊 최종 Test 특징 수: {X_test_fe.shape[1]}")


# ============================================================
# 7. 데이터 스케일링
# ============================================================
print("\n[데이터 스케일링]")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_test_scaled = scaler.transform(X_test_fe)

print("✅ 스케일링 완료!")


# ============================================================
# 8. 모델 학습
# ============================================================
print("\n" + "="*60)
print("🤖 모델 학습")
print("="*60)

# Train/Validation 분할
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train
)

print(f"📊 Train: {X_tr.shape}, Validation: {X_val.shape}")

# 모델 1: Random Forest
print("\n[Random Forest 학습]")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_tr, y_tr)

# 검증 세트 예측
y_val_pred_rf = rf_model.predict(X_val)
f1_rf = f1_score(y_val, y_val_pred_rf, average='weighted')
print(f"✅ Random Forest F1 Score: {f1_rf:.4f}")

# 모델 2: Gradient Boosting
print("\n[Gradient Boosting 학습]")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_tr, y_tr)

# 검증 세트 예측
y_val_pred_gb = gb_model.predict(X_val)
f1_gb = f1_score(y_val, y_val_pred_gb, average='weighted')
print(f"✅ Gradient Boosting F1 Score: {f1_gb:.4f}")

# 최고 성능 모델 선택
if f1_rf > f1_gb:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_f1 = f1_rf
else:
    best_model = gb_model
    best_model_name = "Gradient Boosting"
    best_f1 = f1_gb

print(f"\n🏆 최고 성능 모델: {best_model_name} (F1: {best_f1:.4f})")

# 분류 리포트
print("\n[분류 리포트]")
print(classification_report(y_val, best_model.predict(X_val)))

# Confusion Matrix
print("\n[Confusion Matrix]")
cm = confusion_matrix(y_val, best_model.predict(X_val))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'{best_model_name} - Confusion Matrix')
plt.ylabel('실제값')
plt.xlabel('예측값')
plt.show()


# ============================================================
# 9. 특징 중요도 분석
# ============================================================
print("\n" + "="*60)
print("📊 특징 중요도 분석")
print("="*60)

# 특징 중요도
feature_importance = pd.DataFrame({
    'feature': X_train_fe.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n[Top 10 중요 특징]")
print(feature_importance.head(10))

# 시각화
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='#4ECDC4')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('중요도')
plt.title('Top 15 특징 중요도')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ============================================================
# 10. 테스트 데이터 예측
# ============================================================
print("\n" + "="*60)
print("🔮 테스트 데이터 예측")
print("="*60)

# 전체 Train 데이터로 재학습
print("전체 Train 데이터로 모델 재학습 중...")
best_model.fit(X_train_scaled, y_train)

# 테스트 데이터 예측
y_test_pred = best_model.predict(X_test_scaled)

print(f"✅ 예측 완료!")
print(f"📊 예측 분포:")
print(pd.Series(y_test_pred).value_counts())


# ============================================================
# 11. 제출 파일 생성
# ============================================================
print("\n" + "="*60)
print("📤 제출 파일 생성")
print("="*60)

# 제출 파일 생성
submission[target_col] = y_test_pred

# 저장
submission_path = DATA_PATH + 'submission.csv'
submission.to_csv(submission_path, index=False)

print(f"✅ 제출 파일 저장 완료: {submission_path}")
print("\n[제출 파일 샘플]")
print(submission.head(10))

print("\n" + "="*60)
print("🎉 모든 작업 완료!")
print("="*60)
print(f"📊 최종 모델: {best_model_name}")
print(f"📊 검증 F1 Score: {best_f1:.4f}")
print(f"📊 제출 파일: {submission_path}")
