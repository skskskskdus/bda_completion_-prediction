import nbformat as nbf
import os

notebook_path = r'c:\Users\dell\bda\baseline.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    # 1. Update ID column dropping
    if cell.cell_type == 'code' and "if 'id' in X_train.columns:" in cell.source:
        cell.source = cell.source.replace("'id' in X_train.columns", "'ID' in X_train.columns")
        cell.source = cell.source.replace("columns=['id']", "columns=['ID']")
        cell.source = cell.source.replace("'id' in test.columns", "'ID' in test.columns")
        
    # 2. Update preprocess_data function to include LabelEncoding
    if cell.cell_type == 'code' and "def preprocess_data(df, is_train=True, train_stats=None):" in cell.source:
        new_source = """# 5. 데이터 전처리 및 특징 추출 (Preprocessing & Feature Engineering)
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, is_train=True, train_stats=None):
    df = df.copy()
    
    # 1. ID 컬럼 제거 (대소문자 모두 대응)
    id_cols = ['ID', 'id']
    df = df.drop(columns=[col for col in id_cols if col in df.columns])
    
    # --- 1단계: 결측치 처리 ---
    cols_to_drop = ['contest_award', 'idea_contest']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 특수 카테고리 
    for col in ['class2', 'class3', 'class4']:
        if col in df.columns:
            df[col] = df[col].fillna(-1)
            
    # 이전 기수 수강 / 대회 경험 / 복수전공 등 문자열 결측치 처리
    string_na_map = {
        'previous_class': '신규학생',
        'contest_participation': '경험없음',
        'major1_2': '단일전공'
    }
    for prefix, fill_val in string_na_map.items():
        cols = [col for col in df.columns if col.startswith(prefix)]
        for col in cols:
            df[col] = df[col].fillna(fill_val)
        
    # 3. 일반적 대체 (중앙값/최빈값)
    if is_train:
        stats = {}
        stats['comp_sem_median'] = df['completed_semester'].median() if 'completed_semester' in df.columns else 0
        stats['modes'] = {col: df[col].mode()[0] for col in ['major type', 'major1_1', 'major_field', 'nationality'] if col in df.columns}
        stats['encoders'] = {}
    else:
        stats = train_stats
        
    if 'completed_semester' in df.columns:
        df['completed_semester'] = df['completed_semester'].fillna(stats['comp_sem_median'])
    for col, mode_val in stats['modes'].items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)
        
    # --- 2단계: 특징 추출 ---
    if 'class1' in df.columns:
        df['total_classes'] = df['class1'].fillna(0) + \\
                             df.get('class2', 0).replace(-1, 0) + \\
                             df.get('class3', 0).replace(-1, 0) + \\
                             df.get('class4', 0).replace(-1, 0)
    
    if 'completed_semester' in df.columns:
        df['completed_semester'] = df['completed_semester'].apply(lambda x: x if x <= 12 else 0)
    
    if 'onedayclass_topic' in df.columns:
        df['topic_count'] = df['onedayclass_topic'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
    
    if 'time_input' in df.columns:
        df['has_time_input'] = df['time_input'].apply(lambda x: 1 if x > 0 else 0)
        
    # --- 3단계: 범주형 인코딩 (Label Encoding) ---
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in cat_cols:
        if is_train:
            le = LabelEncoder()
            # 결측값이 있을 수 있으므로 문자열 변환 후 학습
            df[col] = df[col].astype(str)
            le.fit(df[col])
            stats['encoders'][col] = le
        else:
            le = stats['encoders'].get(col)
            if le:
                df[col] = df[col].astype(str)
                # 새로운 카테고리가 있을 경우에 대비해 처리 (또는 train 시 combined fit 사용 권장)
                # 여기서는 train stats의 encoder를 사용
                # le.classes_에 없는 값은 가장 빈번한 값으로 처리하거나 별도 처리 필요
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        
        if le:
            df[col] = le.transform(df[col])
            
    if is_train:
        return df, stats
    return df

# 적용
X_train, train_stats = preprocess_data(train.drop(columns=[target_col]), is_train=True)
X_test = preprocess_data(test, is_train=False, train_stats=train_stats)
y_train = train[target_col]

print("✅ 전처리, 특징 추출 및 인코딩 완료!")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
X_train.head()"""
        cell.source = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully updated {notebook_path}")
