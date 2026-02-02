# ============================================================
# 결측치 처리 전략 요약 문서
# ============================================================

# 📊 결측치 분석 및 처리 전략

## 1. 결측치 현황

### 🔴 결측치 100% (삭제 권장) -> 컬럼 삭제
- **contest_award**: 748개 (100%) - 공모전 수상 경력
- **idea_contest**: 748개 (100%) - 아이디어 공모전 경험

**처리 방법**: ❌ **컬럼 삭제**
- 모든 값이 결측치로 정보 없음
- 모델 학습에 기여할 수 없음

---

### 🟠 결측치 80% 이상 (특수 처리 필요) -> 미수강으로 분류

#### class3, class4
- **class3**: 734개 (98.1%)
- **class4**: 747개 (99.9%)

**처리 방법**: ⚠️ **"미수강" 카테고리 생성** 또는 **삭제**
- 대부분의 학생이 3, 4학기를 수강하지 않음
- 결측치를 `-1` 또는 `"미수강"`으로 대체
- 또는 정보가 부족하므로 삭제 고려

#### previous_class_3~8
- **previous_class_3~8**: 각 602개 (80.5%) -> 미수강으로 분류

**처리 방법**: ⚠️ **"미수강" 카테고리 생성**
- 이전 기수를 수강하지 않은 신규 학생
- 결측치를 `"미수강"`으로 대체
- 재등록 여부와 연관성 있을 수 있음

#### contest_participation
- **contest_participation**: 742개 (99.2%) -> 없음으로 분류

**처리 방법**: ⚠️ **"없음" 카테고리**
- 데이터 대회 경험이 없는 것으로 해석
- 결측치를 `"없음"`으로 대체

---

### 🟡 결측치 50~80% (의미 있는 결측)

#### class2
- **class2**: 579개 (77.4%) -> 미수강으로 분류

**처리 방법**: ⚠️ **"미수강" 카테고리** 또는 **중앙값 대체**
- 2학기를 수강하지 않은 학생
- 결측치를 `-1` 또는 중앙값으로 대체

#### major1_2
- **major1_2**: 439개 (58.7%) -> 단일전공(심화전공)으로 분류

**처리 방법**: ⚠️ **"단일전공" 카테고리**
- 복수전공이 없는 학생
- 결측치를 `"단일전공"` 또는 `"없음"`으로 대체
- `major type`과 일관성 유지

---

### 🟢 결측치 5% 이하 (일반적 대체)

#### completed_semester
- **completed_semester**: 28개 (3.7%)

**처리 방법**: ✅ **중앙값 또는 평균값 대체**
- 수치형 변수
- 결측치 비율이 낮아 중앙값/평균값으로 안전하게 대체 가능

#### major_field
- **major_field**: 23개 (3.1%)   -> 기타로 분류

**처리 방법**: ✅ **최빈값** 또는 **"기타" 카테고리**
- 범주형 변수
- 최빈값으로 대체하거나 "기타"로 분류

#### major type 
- **major type**: 22개 (2.9%)

**처리 방법**: ✅ **최빈값 대체**
- 범주형 변수
- 복수전공 여부이므로 최빈값 사용

#### major1_1
- **major1_1**: 20개 (2.7%) -> 없음으로 분류

**처리 방법**: ✅ **최빈값** 또는 **"미정" 카테고리**
- 범주형 변수
- 제1전공이므로 최빈값 또는 "미정"으로 대체

#### nationality
- **nationality**: 1개 (0.1%) -> 제거

**처리 방법**: ✅ **최빈값 대체**
- 결측치 1개만 존재
- 최빈값(내국인)으로 대체

---

## 2. 처리 우선순위

### 1단계: 삭제
```python
# 100% 결측치 컬럼 삭제
cols_to_drop = ['contest_award', 'idea_contest']
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)
```

### 2단계: 특수 카테고리 생성
```python
# 수강 관련 결측치 -> "미수강"
for col in ['class2', 'class3', 'class4']:
    train[col] = train[col].fillna(-1)  # -1 = 미수강
    test[col] = test[col].fillna(-1)

# 이전 기수 수강 -> "신규학생"
prev_cols = [col for col in train.columns if col.startswith('previous_class')]
for col in prev_cols:
    train[col] = train[col].fillna('신규학생')
    test[col] = test[col].fillna('신규학생')

# 대회 경험 -> "경험없음"
train['contest_participation'] = train['contest_participation'].fillna('경험없음')
test['contest_participation'] = test['contest_participation'].fillna('경험없음')

# 복수전공 -> "단일전공"
train['major1_2'] = train['major1_2'].fillna('단일전공')
test['major1_2'] = test['major1_2'].fillna('단일전공')
```

### 3단계: 일반적 대체
```python
# 수치형: 중앙값
median_val = train['completed_semester'].median()
train['completed_semester'] = train['completed_semester'].fillna(median_val)
test['completed_semester'] = test['completed_semester'].fillna(median_val)

# 범주형: 최빈값
for col in ['major type', 'major1_1', 'major_field', 'nationality']:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col] = test[col].fillna(mode_val)
```

---

## 3. 주의사항

### ⚠️ Train/Test 일관성
- Train과 Test에 **동일한 처리** 적용
- Train에서 계산한 통계값(중앙값, 최빈값)을 Test에도 사용

### ⚠️ 도메인 지식 활용
- 결측치가 **의미 있는 정보**일 수 있음
  - 예: `major1_2` 결측 = 단일전공
  - 예: `previous_class` 결측 = 신규학생
- 단순 삭제보다 **카테고리 생성**이 더 나을 수 있음

### ⚠️ 특징 엔지니어링 기회
- 결측치 여부 자체를 새로운 특징으로 활용
  ```python
  train['has_major2'] = (~train['major1_2'].isnull()).astype(int)
  train['is_returning_student'] = (~train['previous_class_3'].isnull()).astype(int)
  ```

---

## 4. 권장 처리 순서

1. ✅ **100% 결측치 컬럼 삭제**
2. ✅ **의미 있는 결측치 → 특수 카테고리**
3. ✅ **저결측치 수치형 → 중앙값/평균값**
4. ✅ **저결측치 범주형 → 최빈값**
5. ✅ **결측치 여부 특징 생성** (선택)
6. ✅ **최종 확인 및 검증**
