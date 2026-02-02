import json
import os

notebook_path = r'c:\Users\dell\bda\baseline.ipynb'
after_id = 'viz_top_features' # Add after the top features visualization

feature_engineering_source = [
    "# 5. 데이터 전처리 및 특징 추출 (Feature Engineering)\n",
    "def extract_features(df):\n",
    "    df = df.copy()\n",
    "    \n",
    "    # 1. 수치형 변수 결측치 처리 (0으로 채움)\n",
    "    df['class2'] = df['class2'].fillna(0)\n",
    "    df['class3'] = df['class3'].fillna(0)\n",
    "    df['class4'] = df['class4'].fillna(0)\n",
    "    \n",
    "    # 2. 파생 변수 생성: 총 수업 수\n",
    "    df['total_classes'] = df['class1'] + df['class2'] + df['class3'] + df['class4']\n",
    "    \n",
    "    # 3. 이상치 처리: completed_semester\n",
    "    # 20241 같은 값은 오류로 보임. 상한선(예: 12학기)으로 캡핑하거나 0으로 대체\n",
    "    df['completed_semester'] = df['completed_semester'].apply(lambda x: x if x <= 12 else 0)\n",
    "    \n",
    "    # 4. 텍스트 데이터 특징 추출: 원데이클래스 주제 개수\n",
    "    df['topic_count'] = df['onedayclass_topic'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)\n",
    "    \n",
    "    # 5. 시간 관련 변수 활용: time_input (이상치 제거는 하지 않고 그대로 사용하되 파생 변수 고려)\n",
    "    df['has_time_input'] = df['time_input'].apply(lambda x: 1 if x > 0 else 0)\n",
    "    \n",
    "    return df\n",
    "\n",
    "train_eng = extract_features(train)\n",
    "test_eng = extract_features(test)\n",
    "\n",
    "print(\"✅ 특징 추출 완료!\")\n",
    "print(f\"새로 추가된 컬럼: {['total_classes', 'topic_count', 'has_time_input']}\")\n",
    "train_eng[['total_classes', 'completed_semester', 'topic_count', 'has_time_input']].head()\n"
]

viz_new_features_source = [
    "# 신규 특징과 타겟 변수의 상관관계 시각화\n",
    "new_features = ['total_classes', 'completed_semester', 'topic_count', 'has_time_input', 'time_input']\n",
    "\n",
    "plt.figure(figsize=(18, 12))\n",
    "for i, col in enumerate(new_features):\n",
    "    plt.subplot(2, 3, i+1)\n",
    "    if col == 'has_time_input':\n",
    "        sns.barplot(x=col, y=target_col, data=train_eng)\n",
    "        plt.title(f'Completion Rate by {col}')\n",
    "    else:\n",
    "        sns.violinplot(x=target_col, y=col, data=train_eng, palette='Set2')\n",
    "        plt.title(f'{col} Distribution by {target_col}')\n",
    "    plt.grid(axis='y', linestyle='--', alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# 범주형 변수 수료율 분석 (Job)\n",
    "plt.figure(figsize=(12, 6))\n",
    "job_completion = train_eng.groupby('job')[target_col].mean().sort_values(ascending=False)\n",
    "job_completion.plot(kind='bar', color='skyblue')\n",
    "plt.title('Completion Rate by Job', fontsize=15)\n",
    "plt.ylabel('Completion Rate')\n",
    "plt.xticks(rotation=45)\n",
    "plt.axhline(y=train_eng[target_col].mean(), color='r', linestyle='--', label='Average Rate')\n",
    "plt.legend()\n",
    "plt.show()\n"
]

new_cells = [
    {
     "cell_type": "markdown",
     "id": "section5_header",
     "metadata": {},
     "source": ["## 5. 특징 추출 및 데이터 분석"]
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "id": "feature_engineering",
     "metadata": {},
     "outputs": [],
     "source": feature_engineering_source
    },
    {
     "cell_type": "code",
     "execution_count": None,
     "id": "viz_new_features",
     "metadata": {},
     "outputs": [],
     "source": viz_new_features_source
    }
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

updated_cells = []
found = False
for cell in nb.get('cells', []):
    updated_cells.append(cell)
    if cell.get('id') == after_id:
        updated_cells.extend(new_cells)
        found = True

if not found:
    print(f"Error: Cell ID '{after_id}' not found")
else:
    nb['cells'] = updated_cells
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Successfully added feature engineering cells after '{after_id}'")
