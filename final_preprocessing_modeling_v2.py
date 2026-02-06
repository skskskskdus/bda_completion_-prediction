"""
Binary F1 Score 최적화 (Data Leakage 방지 버전)
- Train으로만 fit, Test는 transform
- 데이터 누수 방지
"""

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("="*80)
print("[1] 데이터 로드")
print("="*80)

train = pd.read_csv('open/train.csv', encoding='utf-8-sig')
test = pd.read_csv('open/test.csv', encoding='utf-8-sig')
submission = pd.read_csv('open/sample_submission.csv')

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")
print(f"Target 분포: 수료={train['completed'].sum()}건, 미수료={len(train)-train['completed'].sum()}건")


# ============================================================================
# 2. major1_1/major1_2 매핑 함수
# ============================================================================
def map_major_to_category(major_name):
    """Test의 학과명을 Train의 11개 카테고리로 매핑"""
    if pd.isna(major_name) or str(major_name).strip() == '없음':
        return '없음'

    major_name = str(major_name).lower()

    # IT(컴퓨터 공학 포함)
    if any(k in major_name for k in ['컴퓨터', '소프트웨어', 'ai', '인공지능', '정보',
                                      'sw', '데이터', '빅데이터', 'it', '사이버', '전자',
                                      '반도체', '시스템', '디지털']):
        return 'IT(컴퓨터 공학 포함)'

    # 경영학
    if any(k in major_name for k in ['경영', '마케팅', '회계', '금융', '무역', '유통',
                                      '물류', 'mba', '글로벌경영']):
        return '경영학'

    # 경제통상학
    if any(k in major_name for k in ['경제', '통상', '국제']):
        return '경제통상학'

    # 자연과학
    if any(k in major_name for k in ['수학', '통계', '물리', '화학', '생물', '생명',
                                      '바이오', '환경', '지리', '천문', '지질', '수리',
                                      '응용통계', '빅데이터분석']):
        return '자연과학'

    # 사회과학
    if any(k in major_name for k in ['사회', '심리', '정치', '행정', '언론', '미디어',
                                      '커뮤니케이션', '광고', '홍보', '문헌정보',
                                      '도시', '지역', '소비자']):
        return '사회과학'

    # 인문학
    if any(k in major_name for k in ['국어', '영어', '문학', '어학', '언어', '역사',
                                      '철학', '중국', '일본', '한문', '불어', '독어',
                                      '스페인', '러시아', '아랍', '문화', '인문']):
        return '인문학'

    # 교육학
    if any(k in major_name for k in ['교육', '유아교육', '초등교육']):
        return '교육학'

    # 의약학
    if any(k in major_name for k in ['의학', '의예', '약학', '간호', '의료', '보건',
                                      '수의', '치의', '한의', '의공']):
        return '의약학'

    # 예체능
    if any(k in major_name for k in ['예술', '디자인', '음악', '체육', '스포츠', '미술',
                                      '무용', '연극', '영화', '패션', '의상', '조형',
                                      '시각', '공간디자인']):
        return '예체능'

    # 법학
    if any(k in major_name for k in ['법학', '법률']):
        return '법학'

    # 공학 (IT 제외)
    if any(k in major_name for k in ['공학', '기계', '건축', '토목', '화공', '재료',
                                      '산업', '에너지', '항공', '조선', '자동차',
                                      '신소재', '나노', '융합']):
        return '자연과학'

    return '기타'


# ============================================================================
# 3. onedayclass_topic Multi-Hot 인코딩
# ============================================================================
def extract_topic_keywords(df):
    """onedayclass_topic에서 키워드 추출하여 Multi-Hot 인코딩"""
    keywords = {
        'Python': ['python', 'py'],
        'SQL': ['sql'],
        'AI': ['ai', '인공지능', '머신러닝', '딥러닝', 'ml', 'dl'],
        'Tableau': ['tableau', '태블로'],
        '시각화': ['시각화', 'matplotlib', 'seaborn', 'visualization'],
        '웹크롤링': ['크롤링', 'crawling', '웹 크롤링'],
        'Hadoop': ['hadoop', '하둡', '데이터 엔지니어'],
        '통계': ['통계', 'statistics']
    }

    for key, search_terms in keywords.items():
        df[f'topic_{key}'] = 0
        for term in search_terms:
            df[f'topic_{key}'] |= df['onedayclass_topic'].str.contains(term, case=False, na=False).astype(int)

    # 선택한 토픽 개수
    topic_cols = [f'topic_{k}' for k in keywords.keys()]
    df['topic_count'] = df[topic_cols].sum(axis=1)

    return df, topic_cols


# ============================================================================
# 4. 전처리 함수 (Train 기준으로만 통계량 계산)
# ============================================================================
def preprocess_data(train_df, test_df):
    """
    전체 전처리 파이프라인
    *** 중요: Train으로만 통계량 계산, Test는 적용만 ***
    """

    train_processed = train_df.copy()
    test_processed = test_df.copy()

    print("\n" + "="*80)
    print("[2] 전처리 시작 (Data Leakage 방지)")
    print("="*80)

    # --------------------------------------------------
    # 4.1 삭제할 컬럼
    # --------------------------------------------------
    drop_cols = ['ID', 'generation', 'contest_award', 'idea_contest',
                 'contest_participation', 'interested_company',
                 'incumbents_lecture_scale_reason']

    train_processed = train_processed.drop(columns=[c for c in drop_cols if c in train_processed.columns], errors='ignore')
    test_processed = test_processed.drop(columns=[c for c in drop_cols if c in test_processed.columns], errors='ignore')
    print(f"  삭제 컬럼: {len(drop_cols)}개")

    # --------------------------------------------------
    # 4.2 major1_1/major1_2 매핑
    # --------------------------------------------------
    print("\n[3] major1_1/major1_2 매핑...")
    train_processed['major1_1_mapped'] = train_processed['major1_1'].apply(map_major_to_category)
    test_processed['major1_1_mapped'] = test_processed['major1_1'].apply(map_major_to_category)

    train_processed['major1_2_mapped'] = train_processed['major1_2'].apply(map_major_to_category)
    test_processed['major1_2_mapped'] = test_processed['major1_2'].apply(map_major_to_category)

    train_processed = train_processed.drop(columns=['major1_1', 'major1_2'], errors='ignore')
    test_processed = test_processed.drop(columns=['major1_1', 'major1_2'], errors='ignore')

    print(f"  major1_1_mapped: {train_processed['major1_1_mapped'].nunique()}개 카테고리")
    print(f"  major1_2_mapped: {train_processed['major1_2_mapped'].nunique()}개 카테고리")

    # --------------------------------------------------
    # 4.3 onedayclass_topic Multi-Hot
    # --------------------------------------------------
    print("\n[4] onedayclass_topic Multi-Hot 인코딩...")
    train_processed, topic_cols = extract_topic_keywords(train_processed)
    test_processed, _ = extract_topic_keywords(test_processed)
    train_processed = train_processed.drop(columns=['onedayclass_topic'], errors='ignore')
    test_processed = test_processed.drop(columns=['onedayclass_topic'], errors='ignore')
    print(f"  추출된 토픽: {len(topic_cols)}개")

    # --------------------------------------------------
    # 4.4 completed_semester 이상치 처리
    # *** Train으로만 중앙값 계산 ***
    # --------------------------------------------------
    print("\n[5] completed_semester 이상치 처리 (Train 기준)")
    median_sem = train_processed['completed_semester'].median()
    print(f"  Train 중앙값: {median_sem}")

    # Train 이상치 처리
    train_processed.loc[train_processed['completed_semester'] > 12, 'completed_semester'] = median_sem
    train_processed.loc[train_processed['completed_semester'] < 0, 'completed_semester'] = median_sem
    train_processed['completed_semester'].fillna(median_sem, inplace=True)

    # Test 이상치 처리 (Train 중앙값 사용)
    test_processed.loc[test_processed['completed_semester'] > 12, 'completed_semester'] = median_sem
    test_processed.loc[test_processed['completed_semester'] < 0, 'completed_semester'] = median_sem
    test_processed['completed_semester'].fillna(median_sem, inplace=True)

    # --------------------------------------------------
    # 4.5 class 컬럼 처리
    # --------------------------------------------------
    print("\n[6] class 컬럼 파생변수 생성...")
    class_cols = ['class1', 'class2', 'class3', 'class4']

    for col in class_cols:
        if col in train_processed.columns:
            train_processed[col].fillna(0, inplace=True)
            test_processed[col].fillna(0, inplace=True)

    train_processed['num_classes'] = (train_processed[class_cols] > 0).sum(axis=1)
    test_processed['num_classes'] = (test_processed[class_cols] > 0).sum(axis=1)

    train_processed = train_processed.drop(columns=class_cols, errors='ignore')
    test_processed = test_processed.drop(columns=class_cols, errors='ignore')

    # --------------------------------------------------
    # 4.6 previous_class 컬럼 처리
    # --------------------------------------------------
    print("\n[7] previous_class 컬럼 처리...")
    prev_cols = [col for col in train_processed.columns if col.startswith('previous_class')]

    for col in prev_cols:
        train_processed[col].fillna('해당없음', inplace=True)
        test_processed[col].fillna('해당없음', inplace=True)

    if prev_cols:
        train_processed['is_returning'] = (train_processed[prev_cols[0]] != '해당없음').astype(int)
        test_processed['is_returning'] = (test_processed[prev_cols[0]] != '해당없음').astype(int)

    # --------------------------------------------------
    # 4.7 결측치 처리
    # *** Train의 최빈값으로 계산 ***
    # --------------------------------------------------
    print("\n[8] 결측치 처리 (Train 기준)")

    categorical_cols = ['major type', 'major_field', 'nationality']
    for col in categorical_cols:
        if col in train_processed.columns:
            # Train 최빈값
            mode_val = train_processed[col].mode()
            if len(mode_val) > 0:
                mode_val = mode_val[0]
            else:
                mode_val = '미응답'

            print(f"  {col} 최빈값: {mode_val}")

            train_processed[col].fillna(mode_val, inplace=True)
            test_processed[col].fillna(mode_val, inplace=True)

    # --------------------------------------------------
    # 4.8 F1 최적화 파생변수
    # --------------------------------------------------
    print("\n[9] F1 최적화 파생변수 생성...")

    train_processed['is_reregistered'] = (train_processed['re_registration'] == '예').astype(int)
    test_processed['is_reregistered'] = (test_processed['re_registration'] == '예').astype(int)

    train_processed['is_data_major'] = train_processed['major_data'].astype(int)
    test_processed['is_data_major'] = test_processed['major_data'].astype(int)

    train_processed['has_major2'] = (train_processed['major1_2_mapped'] != '없음').astype(int)
    test_processed['has_major2'] = (test_processed['major1_2_mapped'] != '없음').astype(int)

    # --------------------------------------------------
    # 4.9 고카디널리티 컬럼 처리
    # --------------------------------------------------
    high_card_cols = ['desired_job', 'desired_certificate', 'desired_job_except_data',
                      'certificate_acquisition']

    for col in high_card_cols:
        if col in train_processed.columns:
            train_processed[col].fillna('없음', inplace=True)
            test_processed[col].fillna('없음', inplace=True)

    # expected_domain 삭제
    if 'expected_domain' in train_processed.columns:
        train_processed = train_processed.drop(columns=['expected_domain'], errors='ignore')
        test_processed = test_processed.drop(columns=['expected_domain'], errors='ignore')

    print(f"\n  전처리 완료!")
    print(f"  Train shape: {train_processed.shape}")
    print(f"  Test shape: {test_processed.shape}")

    return train_processed, test_processed


# ============================================================================
# 5. 인코딩 (Train으로만 fit, Test는 transform)
# ============================================================================
def encode_features(train_df, test_df):
    """
    범주형 변수 인코딩
    *** 중요: Train으로만 fit, Test는 transform ***
    """

    print("\n" + "="*80)
    print("[10] 피처 인코딩 (Train fit, Test transform)")
    print("="*80)

    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    # Target 분리
    if 'completed' in train_encoded.columns:
        y = train_encoded['completed'].values
        train_encoded = train_encoded.drop(columns=['completed'])
    else:
        y = None

    # 범주형 컬럼
    categorical_cols = train_encoded.select_dtypes(include=['object']).columns.tolist()

    print(f"  범주형 컬럼: {len(categorical_cols)}개")

    # Label Encoding
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()

        # *** Train으로만 fit ***
        train_values = train_encoded[col].astype(str)
        le.fit(train_values)

        # Train transform
        train_encoded[col] = le.transform(train_values)

        # Test transform (unknown 처리)
        test_values = test_encoded[col].astype(str)
        test_encoded[col] = test_values.apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

        # unknown 카테고리가 있는지 확인
        unknown_count = (test_encoded[col] == -1).sum()
        if unknown_count > 0:
            print(f"    {col}: Test에 {unknown_count}개 unknown 카테고리 (-1로 처리)")

        encoders[col] = le

    print(f"\n  인코딩 완료!")
    print(f"  최종 Train shape: {train_encoded.shape}")
    print(f"  최종 Test shape: {test_encoded.shape}")

    return train_encoded, test_encoded, y, encoders


# ============================================================================
# 6. K-Fold 학습 및 평가
# ============================================================================
def train_with_kfold(X, y, X_test, n_splits=5):
    """K-Fold CV 학습"""

    print("\n" + "="*80)
    print(f"[11] K-Fold Training (K={n_splits})")
    print("="*80)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(X))
    oof_preds = np.zeros(len(X))
    test_probs = []

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/{n_splits}]")
        print("-" * 60)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Class 불균형 처리
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        # XGBoost 모델
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_lambda=1.0,
            reg_alpha=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Validation 예측
        val_probs = model.predict_proba(X_val)[:, 1]

        # Threshold 최적화
        best_threshold = 0.25
        best_f1 = 0

        for threshold in np.arange(0.2, 0.8, 0.01):
            val_preds_temp = (val_probs >= threshold).astype(int)
            f1_temp = f1_score(y_val, val_preds_temp, zero_division=0)

            if f1_temp > best_f1:
                best_f1 = f1_temp
                best_threshold = threshold

        val_preds = (val_probs >= best_threshold).astype(int)

        # 평가 지표
        f1 = f1_score(y_val, val_preds, zero_division=0)
        precision = precision_score(y_val, val_preds, zero_division=0)
        recall = recall_score(y_val, val_preds, zero_division=0)

        print(f"  Best Threshold: {best_threshold:.3f}")
        print(f"  F1 Score:    {f1:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_val, val_preds)
        tn, fp, fn, tp = cm.ravel()
        print(f"  혼동행렬: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # OOF 저장
        oof_probs[val_idx] = val_probs
        oof_preds[val_idx] = val_preds

        fold_scores.append({
            'fold': fold,
            'threshold': best_threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })

        # Test 예측
        test_probs.append(model.predict_proba(X_test)[:, 1])

    # --------------------------------------------------
    # OOF 결과
    # --------------------------------------------------
    print("\n" + "="*80)
    print("[12] OOF (Out-of-Fold) 결과")
    print("="*80)

    # Global Threshold 최적화
    best_global_threshold = 0.5
    best_global_f1 = 0

    for threshold in np.arange(0.2, 0.8, 0.005):
        oof_preds_temp = (oof_probs >= threshold).astype(int)
        f1_temp = f1_score(y, oof_preds_temp, zero_division=0)

        if f1_temp > best_global_f1:
            best_global_f1 = f1_temp
            best_global_threshold = threshold

    oof_preds_final = (oof_probs >= best_global_threshold).astype(int)

    oof_f1 = f1_score(y, oof_preds_final, zero_division=0)
    oof_precision = precision_score(y, oof_preds_final, zero_division=0)
    oof_recall = recall_score(y, oof_preds_final, zero_division=0)

    print(f"\nGlobal Best Threshold: {best_global_threshold:.3f}")
    print(f"\nOOF F1 Score:    {oof_f1:.4f}")
    print(f"OOF Precision:   {oof_precision:.4f}")
    print(f"OOF Recall:      {oof_recall:.4f}")

    # OOF Confusion Matrix
    cm_oof = confusion_matrix(y, oof_preds_final)
    tn, fp, fn, tp = cm_oof.ravel()
    print(f"\n혼동행렬:")
    print(f"  TN (실제0, 예측0): {tn}")
    print(f"  FP (실제0, 예측1): {fp}")
    print(f"  FN (실제1, 예측0): {fn}")
    print(f"  TP (실제1, 예측1): {tp}")

    # --------------------------------------------------
    # Test 예측
    # --------------------------------------------------
    test_probs_mean = np.mean(test_probs, axis=0)
    test_preds = (test_probs_mean >= best_global_threshold).astype(int)

    print(f"\n[Test 예측]")
    print(f"  Threshold: {best_global_threshold:.3f}")
    print(f"  수료 예측: {test_preds.sum()}건 ({test_preds.mean()*100:.1f}%)")
    print(f"  미수료 예측: {(1-test_preds).sum()}건 ({(1-test_preds.mean())*100:.1f}%)")

    return {
        'oof_probs': oof_probs,
        'oof_preds': oof_preds_final,
        'test_preds': test_preds,
        'test_probs': test_probs_mean,
        'best_threshold': best_global_threshold,
        'oof_f1': oof_f1,
        'oof_precision': oof_precision,
        'oof_recall': oof_recall,
        'fold_scores': fold_scores
    }


# ============================================================================
# 7. 메인 실행
# ============================================================================
if __name__ == "__main__":

    # 전처리
    train_processed, test_processed = preprocess_data(train, test)

    # 인코딩
    X_train, X_test, y, encoders = encode_features(train_processed, test_processed)

    # 학습 및 평가
    results = train_with_kfold(X_train, y, X_test, n_splits=5)

    # 제출 파일 생성
    print("\n" + "="*80)
    print("[13] 제출 파일 생성")
    print("="*80)

    submission['completed'] = results['test_preds']
    submission.to_csv('submission_final_v2.csv', index=False)

    print(f"\n저장 완료: submission_final_v2.csv")
    print(f"\n최종 제출 파일:")
    print(submission['completed'].value_counts())

    print("\n" + "="*80)
    print("[완료] 모든 작업 완료!")
    print("="*80)
    print(f"최종 OOF F1 Score: {results['oof_f1']:.4f}")
    print(f"최종 OOF Precision: {results['oof_precision']:.4f}")
    print(f"최종 OOF Recall: {results['oof_recall']:.4f}")
    print(f"Best Threshold: {results['best_threshold']:.3f}")
