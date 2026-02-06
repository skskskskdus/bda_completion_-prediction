"""
Keras DNN 기반 경량 딥러닝 모델
- 간단한 Multi-Layer Perceptron
- 빠른 학습, 가벼운 모델
"""

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU available: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU available, using CPU")

print("="*80)
print("Keras DNN 기반 경량 딥러닝 모델 - Binary Classification")
print("="*80)

def preprocess_data(train, test):
    """데이터 전처리"""
    train_processed = train.copy()
    test_processed = test.copy()

    # 1. ID 컬럼 제거
    if 'ID' in train_processed.columns:
        train_processed = train_processed.drop('ID', axis=1)
    if 'ID' in test_processed.columns:
        test_processed = test_processed.drop('ID', axis=1)

    # 2. 결측치 처리
    for col in train_processed.columns:
        if col == 'completed':
            continue
        if train_processed[col].dtype == 'object':
            train_processed[col] = train_processed[col].fillna('Unknown')
            if col in test_processed.columns:
                test_processed[col] = test_processed[col].fillna('Unknown')
        else:
            train_processed[col] = train_processed[col].fillna(train_processed[col].median())
            if col in test_processed.columns:
                test_processed[col] = test_processed[col].fillna(train_processed[col].median())

    # 3. previous_class 컬럼 제거
    cols_to_drop = ['previous_class_1', 'previous_class_2', 'previous_class_3',
                    'previous_class_4', 'previous_class_5']
    train_processed = train_processed.drop(cols_to_drop, axis=1, errors='ignore')
    test_processed = test_processed.drop(cols_to_drop, axis=1, errors='ignore')

    # 4. Major 매핑
    def map_major_to_category(major_name):
        major_map = {
            '경영': '경영학', '회계': '경영학', '마케팅': '경영학', '재무': '경영학',
            '컴퓨터': 'IT(컴퓨터 공학 포함)', '소프트웨어': 'IT(컴퓨터 공학 포함)',
            '정보통신': 'IT(컴퓨터 공학 포함)', '전자': 'IT(컴퓨터 공학 포함)',
            '경제': '경제통상학', '무역': '경제통상학', '국제': '경제통상학',
            '수학': '자연과학', '물리': '자연과학', '화학': '자연과학', '생물': '자연과학', '통계': '자연과학',
            '의학': '의약학', '간호': '의약학', '약학': '의약학', '치의': '의약학',
            '교육': '교육학', '유아': '교육학',
            '국문': '인문학', '영문': '인문학', '사학': '인문학', '철학': '인문학', '언어': '인문학',
            '사회': '사회과학', '정치': '사회과학', '행정': '사회과학', '심리': '사회과학',
            '미술': '예체능', '음악': '예체능', '체육': '예체능', '디자인': '예체능'
        }
        major_name_str = str(major_name)
        for keyword, category in major_map.items():
            if keyword in major_name_str:
                return category
        return '기타'

    if 'major1_2' in test_processed.columns:
        test_processed['major1_1'] = test_processed['major1_2'].apply(map_major_to_category)
        test_processed = test_processed.drop('major1_2', axis=1)

    # 5. onedayclass_topic multi-hot encoding
    keywords = ['Python', 'SQL', '머신러닝', 'AI', '딥러닝', 'Tableau', 'Power BI', 'R']

    def extract_topics(topic_str):
        topic_str = str(topic_str)
        return {f'topic_{kw}': int(kw in topic_str) for kw in keywords}

    for df in [train_processed, test_processed]:
        if 'onedayclass_topic' in df.columns:
            topic_features = df['onedayclass_topic'].apply(extract_topics).apply(pd.Series)
            df = pd.concat([df, topic_features], axis=1)
            df['topic_count'] = topic_features.sum(axis=1)
            df = df.drop('onedayclass_topic', axis=1)

    train_processed = pd.concat([
        train_processed.drop('onedayclass_topic', axis=1, errors='ignore'),
        train_processed['onedayclass_topic'].apply(extract_topics).apply(pd.Series)
    ], axis=1)
    train_processed['topic_count'] = train_processed[[f'topic_{kw}' for kw in keywords]].sum(axis=1)

    test_processed = pd.concat([
        test_processed.drop('onedayclass_topic', axis=1, errors='ignore'),
        test_processed['onedayclass_topic'].apply(extract_topics).apply(pd.Series)
    ], axis=1)
    test_processed['topic_count'] = test_processed[[f'topic_{kw}' for kw in keywords]].sum(axis=1)

    # 6. class_number -> num_classes
    def count_classes(class_str):
        if pd.isna(class_str) or class_str == 'Unknown':
            return 0
        return len(str(class_str).split(','))

    train_processed['num_classes'] = train_processed['class_number'].apply(count_classes)
    train_processed = train_processed.drop('class_number', axis=1)

    test_processed['num_classes'] = test_processed['class_number'].apply(count_classes)
    test_processed = test_processed.drop('class_number', axis=1)

    return train_processed, test_processed

def encode_features(train, test):
    """Feature Encoding - Data Leakage 방지"""
    train_encoded = train.copy()
    test_encoded = test.copy()

    # Target 분리
    y = train_encoded['completed'].values
    train_encoded = train_encoded.drop('completed', axis=1)

    # Categorical/Numerical 분리
    categorical_cols = train_encoded.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = train_encoded.select_dtypes(exclude=['object']).columns.tolist()

    print(f"\nCategorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")

    encoders = {}

    # Categorical Encoding (Train only fit!)
    for col in categorical_cols:
        le = LabelEncoder()

        # Train fit
        train_values = train_encoded[col].astype(str)
        le.fit(train_values)
        encoders[col] = le

        # Train transform
        train_encoded[col] = le.transform(train_values)

        # Test transform (Unknown 처리)
        test_values = test_encoded[col].astype(str)
        test_encoded[col] = test_values.apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    # Numerical Scaling (Train only fit!)
    scaler = StandardScaler()
    train_encoded[numerical_cols] = scaler.fit_transform(train_encoded[numerical_cols])
    test_encoded[numerical_cols] = scaler.transform(test_encoded[numerical_cols])
    encoders['scaler'] = scaler

    return train_encoded, test_encoded, y, encoders

def create_dnn_model(input_dim, dropout_rate=0.3):
    """DNN 모델 생성"""
    model = keras.Sequential([
        # Input layer
        layers.InputLayer(input_shape=(input_dim,)),

        # Hidden layers with BatchNorm and Dropout
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    return model

# 데이터 로드
print("\nLoading data...")
train = pd.read_csv('open/train.csv', encoding='utf-8-sig')
test = pd.read_csv('open/test.csv', encoding='utf-8-sig')
test_ids = test['ID'].values

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 전처리
print("\nPreprocessing...")
train_processed, test_processed = preprocess_data(train, test)
X_train, X_test, y, encoders = encode_features(train_processed, test_processed)

print(f"\nProcessed Train shape: {X_train.shape}")
print(f"Processed Test shape: {X_test.shape}")

# K-Fold Cross Validation
print("\n" + "="*80)
print("K-Fold Cross Validation with Keras DNN")
print("="*80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
oof_probs = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
test_probs = np.zeros(len(X_test))

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\nFold {fold}/5")
    print("-" * 40)

    X_tr, X_val = X_train.iloc[train_idx].values, X_train.iloc[val_idx].values
    y_tr, y_val = y[train_idx], y[val_idx]

    # Class weight 계산
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    class_weight = {0: 1.0, 1: pos_weight}

    # 모델 생성
    model = create_dnn_model(X_train.shape[1])

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )

    # 학습
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Validation 예측
    val_probs = model.predict(X_val, verbose=0).flatten()

    # Threshold 최적화
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.2, 0.8, 0.01):
        val_preds_temp = (val_probs >= threshold).astype(int)
        f1_temp = f1_score(y_val, val_preds_temp, zero_division=0)

        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold

    val_preds = (val_probs >= best_threshold).astype(int)

    # Metrics
    f1 = f1_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds, zero_division=0)
    recall = recall_score(y_val, val_preds, zero_division=0)

    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Epochs trained: {len(history.history['loss'])}")

    fold_metrics.append({
        'fold': fold,
        'threshold': best_threshold,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'epochs': len(history.history['loss'])
    })

    # OOF 저장
    oof_preds[val_idx] = val_preds
    oof_probs[val_idx] = val_probs

    # Test 예측 (확률 평균)
    test_probs += model.predict(X_test.values, verbose=0).flatten() / 5

# Overall OOF Metrics
print("\n" + "="*80)
print("Overall OOF Performance")
print("="*80)

# 최적 threshold (전체 OOF)
best_oof_threshold = 0.5
best_oof_f1 = 0

for threshold in np.arange(0.2, 0.8, 0.01):
    oof_preds_temp = (oof_probs >= threshold).astype(int)
    f1_temp = f1_score(y, oof_preds_temp, zero_division=0)

    if f1_temp > best_oof_f1:
        best_oof_f1 = f1_temp
        best_oof_threshold = threshold

oof_preds_final = (oof_probs >= best_oof_threshold).astype(int)

oof_f1 = f1_score(y, oof_preds_final)
oof_precision = precision_score(y, oof_preds_final, zero_division=0)
oof_recall = recall_score(y, oof_preds_final, zero_division=0)

print(f"\nBest OOF Threshold: {best_oof_threshold:.3f}")
print(f"OOF F1 Score: {oof_f1:.4f}")
print(f"OOF Precision: {oof_precision:.4f} ({oof_precision*100:.2f}%)")
print(f"OOF Recall: {oof_recall:.4f} ({oof_recall*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y, oof_preds_final)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  TN: {tn:4d}  |  FP: {fp:4d}")
print(f"  FN: {fn:4d}  |  TP: {tp:4d}")

print(f"\nDetailed:")
print(f"  True Negatives (TN):  {tn:4d} - 실제 미수료를 미수료로 예측")
print(f"  False Positives (FP): {fp:4d} - 미수료를 수료로 잘못 예측")
print(f"  False Negatives (FN): {fn:4d} - 수료를 미수료로 잘못 예측")
print(f"  True Positives (TP):  {tp:4d} - 실제 수료를 수료로 예측")

# Fold별 평균
print("\n" + "="*80)
print("Fold-wise Metrics Summary")
print("="*80)
fold_df = pd.DataFrame(fold_metrics)
print(fold_df.to_string(index=False))
print(f"\nAverage F1: {fold_df['f1'].mean():.4f} (±{fold_df['f1'].std():.4f})")
print(f"Average Precision: {fold_df['precision'].mean():.4f}")
print(f"Average Recall: {fold_df['recall'].mean():.4f}")
print(f"Average Epochs: {fold_df['epochs'].mean():.1f}")

# Test 예측
print("\n" + "="*80)
print("Test Prediction")
print("="*80)

# Threshold는 OOF 최적값 사용
test_preds_final = (test_probs >= best_oof_threshold).astype(int)

print(f"Using Threshold: {best_oof_threshold:.3f}")
print(f"Predicted Completions: {test_preds_final.sum()}/{len(test_preds_final)} ({test_preds_final.mean()*100:.1f}%)")
print(f"Train Completions: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

# Submission 저장
submission = pd.DataFrame({
    'ID': test_ids,
    'completed': test_preds_final
})

output_file = 'submission_dl_keras.csv'
submission.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\nSubmission saved: {output_file}")

print("\n" + "="*80)
print("Keras DNN Model Training Completed!")
print("="*80)
