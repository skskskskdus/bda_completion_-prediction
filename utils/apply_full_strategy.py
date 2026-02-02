import json
import os

notebook_path = r'c:\Users\dell\bda\baseline.ipynb'
# We will replace or modify the previous simplistic 'feature_engineering' cell
# or add it after the previous analysis.
# Let's find the 'feature_engineering' cell if it exists, otherwise add after 'viz_top_features'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Improved Preprocessing and Feature Engineering based on MISSING_VALUE_STRATEGY.md
preprocessing_source = [
    "# 5. 데이터 전처리 및 특징 추출 (Preprocessing & Feature Engineering)\n",
    "def preprocess_data(df, is_train=True, train_stats=None):\n",
    "    df = df.copy()\n",
    "    \n",
    "    # --- 1단계: 결측치 처리 (MISSING_VALUE_STRATEGY.md 반영) ---\n",
    "    \n",
    "    # 1. 100% 결측치 컬럼 삭제\n",
    "    cols_to_drop = ['contest_award', 'idea_contest']\n",
    "    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])\n",
    "    \n",
    "    # 2. 특수 카테고리 생성\n",
    "    # 수강 관련 (-1 = 미수강)\n",
    "    for col in ['class2', 'class3', 'class4']:\n",
    "        if col in df.columns:\n",
    "            df[col] = df[col].fillna(-1)\n",
    "            \n",
    "    # 이전 기수 수강 (\"신규학생\")\n",
    "    prev_cols = [col for col in df.columns if col.startswith('previous_class')]\n",
    "    for col in prev_cols:\n",
    "        df[col] = df[col].fillna('신규학생')\n",
    "        \n",
    "    # 대회 경험 (\"경험없음\")\n",
    "    if 'contest_participation' in df.columns:\n",
    "        df['contest_participation'] = df['contest_participation'].fillna('경험없음')\n",
    "        \n",
    "    # 복수전공 (\"단일전공\")\n",
    "    if 'major1_2' in df.columns:\n",
    "        df['major1_2'] = df['major1_2'].fillna('단일전공')\n",
    "        \n",
    "    # 3. 일반적 대체 (중앙값/최빈값)\n",
    "    if is_train:\n",
    "        stats = {}\n",
    "        stats['comp_sem_median'] = df['completed_semester'].median()\n",
    "        stats['modes'] = {col: df[col].mode()[0] for col in ['major type', 'major1_1', 'major_field', 'nationality'] if col in df.columns}\n",
    "    else:\n",
    "        stats = train_stats\n",
    "        \n",
    "    df['completed_semester'] = df['completed_semester'].fillna(stats['comp_sem_median'])\n",
    "    for col, mode_val in stats['modes'].items():\n",
    "        df[col] = df[col].fillna(mode_val)\n",
    "        \n",
    "    # --- 2단계: 특징 추출 (Feature Engineering) ---\n",
    "    \n",
    "    # 1. 총 수업 수 (미수강 -1을 0으로 처리하여 계산)\n",
    "    df['total_classes'] = df['class1'].fillna(0) + \\\n",
    "                         df['class2'].replace(-1, 0) + \\\n",
    "                         df['class3'].replace(-1, 0) + \\\n",
    "                         df['class4'].replace(-1, 0)\n",
    "    \n",
    "    # 2. 이상치 처리: completed_semester (12학기 캡핑)\n",
    "    df['completed_semester'] = df['completed_semester'].apply(lambda x: x if x <= 12 else 0)\n",
    "    \n",
    "    # 3. 주제 개수 추출\n",
    "    if 'onedayclass_topic' in df.columns:\n",
    "        df['topic_count'] = df['onedayclass_topic'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)\n",
    "    \n",
    "    # 4. 시간 투입 여부\n",
    "    if 'time_input' in df.columns:\n",
    "        df['has_time_input'] = df['time_input'].apply(lambda x: 1 if x > 0 else 0)\n",
    "    \n",
    "    if is_train:\n",
    "        return df, stats\n",
    "    return df\n",
    "\n",
    "# 적용\n",
    "train_processed, train_stats = preprocess_data(train, is_train=True)\n",
    "test_processed = preprocess_data(test, is_train=False, train_stats=train_stats)\n",
    "\n",
    "print(\"✅ 전처리 및 특징 추출 완료 (MISSING_VALUE_STRATEGY 적용)!\")\n",
    "print(f\"삭제된 컬럼: ['contest_award', 'idea_contest']\")\n",
    "print(f\"새로 추가된 컬럼: {['total_classes', 'topic_count', 'has_time_input']}\")\n",
    "train_processed.head()\n"
]

# Find the indices of cells to replace or where to insert
cells = nb.get('cells', [])
new_cells_list = []
replaced = False

for cell in cells:
    # If we find the previous feature_engineering cell, replace its content or the whole cell
    if cell.get('id') == 'feature_engineering':
        cell['source'] = preprocessing_source
        new_cells_list.append(cell)
        replaced = True
    else:
        new_cells_list.append(cell)

if not replaced:
    # If not found, insert after viz_top_features as fallback
    found_idx = -1
    for i, cell in enumerate(new_cells_list):
        if cell.get('id') == 'viz_top_features':
            found_idx = i
            break
    if found_idx != -1:
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "feature_engineering",
            "metadata": {},
            "outputs": [],
            "source": preprocessing_source
        }
        new_cells_list.insert(found_idx + 1, new_cell)
        replaced = True

if replaced:
    nb['cells'] = new_cells_list
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully updated notebook with comprehensive missing value strategy.")
else:
    print("Error: Could not find insertion point in notebook.")
