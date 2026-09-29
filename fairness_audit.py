import pandas as pd

# 1. Đọc dữ liệu kiểm thử và dự báo của XGBoost
test = pd.read_csv('data/test.csv')
pred = pd.read_csv('reports/pred_xgb.csv')

# 2. Ghép xác suất dự báo vào dữ liệu gốc
df = test[['SEX', 'AGE']].copy()
df['prob_default'] = pred['prob_default']

# 3. Phân nhóm và quyết định phê duyệt (Ngưỡng PD < 0.5 thì Duyệt)
df['AGE_group'] = df['AGE'].apply(lambda x: 'Trẻ tuổi (<=25)' if x <= 25 else 'Trưởng thành (>25)')
df['SEX_group'] = df['SEX'].map({1: 'Nam', 2: 'Nữ'})
df['Approved'] = (df['prob_default'] < 0.5).astype(int)

# 4. Tính toán theo Giới tính
sex_metrics = df.groupby('SEX_group')['Approved'].agg(['count', 'mean'])
rate_nam = sex_metrics.loc['Nam', 'mean']
rate_nu = sex_metrics.loc['Nữ', 'mean']
di_sex = rate_nam / rate_nu  # Nam là nhóm yếu thế (tỷ lệ duyệt thấp hơn)

# 5. Tính toán theo Độ tuổi
age_metrics = df.groupby('AGE_group')['Approved'].agg(['count', 'mean'])
rate_tre = age_metrics.loc['Trẻ tuổi (<=25)', 'mean']
rate_truong_thanh = age_metrics.loc['Trưởng thành (>25)', 'mean']
di_age = rate_tre / rate_truong_thanh  # Trẻ tuổi là nhóm yếu thế

# 6. In kết quả và Quy tắc 80% (Four-Fifths Rule)
print("\n=== KIỂM ĐỊNH THIÊN LỆCH THEO GIỚI TÍNH (SEX) ===")
print(sex_metrics.round(4).rename(columns={'count': 'Số hồ sơ', 'mean': 'Tỷ lệ duyệt'}))
print(f"-> Chỉ số Disparate Impact (DI): {di_sex:.2%}")
print(f"-> Đánh giá Quy tắc 80%: {'ĐẠT (Không thiên lệch)' if di_sex >= 0.8 else 'KHÔNG ĐẠT (Có thiên lệch)'}")

print("\n=== KIỂM ĐỊNH THIÊN LỆCH THEO ĐỘ TUỔI (AGE) ===")
print(age_metrics.round(4).rename(columns={'count': 'Số hồ sơ', 'mean': 'Tỷ lệ duyệt'}))
print(f"-> Chỉ số Disparate Impact (DI): {di_age:.2%}")
print(f"-> Đánh giá Quy tắc 80%: {'ĐẠT (Không thiên lệch)' if di_age >= 0.8 else 'KHÔNG ĐẠT (Có thiên lệch)'}")