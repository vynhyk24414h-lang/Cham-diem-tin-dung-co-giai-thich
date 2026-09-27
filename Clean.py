import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

print("Đang tải dữ liệu...")

df = pd.read_excel(
    "default of credit card clients.xls",
    header=1
)

# Xóa cột ID
df = df.drop(columns=["ID"])

# Đổi tên biến mục tiêu
df = df.rename(
    columns={"default payment next month": "default"}
)

print(f"Kích thước dữ liệu: {df.shape}")

#### LÀM SẠCH DỮ LIỆU

# 1. Kiểm tra dữ liệu thiếu
print("\n" + "="*50)
print("--- KIỂM TRA GIÁ TRỊ THIẾU ---")

missing = df.isnull().sum()

print(missing[missing > 0])
print(f"Tổng số giá trị thiếu: {missing.sum()}")

# 2. Kiểm tra nhãn không hợp lệ
print("\n" + "="*50)
print("--- KIỂM TRA NHÃN KHÔNG HỢP LỆ ---")

##### EDUCATION
valid_education = [1, 2, 3, 4]
invalid_education = df.loc[
    ~df["EDUCATION"].isin(valid_education),
    "EDUCATION"
]

print("\nGiá trị EDUCATION bất thường:")
print(invalid_education.value_counts().sort_index())

print(f"Tổng số giá trị EDUCATION bất thường: {len(invalid_education)}")


##### MARRIAGE
valid_marriage = [1, 2, 3]
invalid_marriage = df.loc[
    ~df["MARRIAGE"].isin(valid_marriage),
    "MARRIAGE"
]

print("\nGiá trị MARRIAGE bất thường:")
print(invalid_marriage.value_counts().sort_index())

print(f"Tổng số giá trị MARRIAGE bất thường: {len(invalid_marriage)}")


##### SEX
valid_sex = [1, 2]
invalid_sex = df.loc[
    ~df["SEX"].isin(valid_sex),
    "SEX"
]

print("\nGiá trị SEX bất thường:")
print(invalid_sex.value_counts().sort_index())

print(f"Tổng số giá trị SEX bất thường: {len(invalid_sex)}")


##### DEFAULT
valid_default = [0, 1]
invalid_default = df.loc[
    ~df["default"].isin(valid_default),
    "default"
]

print("\nGiá trị DEFAULT bất thường:")
print(invalid_default.value_counts().sort_index())

print(f"Tổng số giá trị DEFAULT bất thường: {len(invalid_default)}")

##### PAY*
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
valid_pay = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9]

invalid_pay = df[pay_cols].apply(
    lambda x: ~x.isin(valid_pay)
)

print("\nGiá trị PAY_* chưa được codebook định nghĩa:")   # Đây là tất cả giá trị chưa được định nghĩa của 6 cột PAY_*
print(df[pay_cols][invalid_pay].stack().value_counts().sort_index())

print(f"Tổng số giá trị PAY_* chưa được codebook định nghĩa: {invalid_pay.sum().sum()}")

# Các giá trị hợp lệ theo codebook
invalid_sex = df.loc[~df["SEX"].isin([1, 2])]
invalid_education = df.loc[~df["EDUCATION"].isin([1, 2, 3, 4])]
invalid_marriage = df.loc[~df["MARRIAGE"].isin([1, 2, 3])]
invalid_default = df.loc[~df["default"].isin([0, 1])]

print(f"SEX không hợp lệ: {len(invalid_sex)}")
print(f"EDUCATION không hợp lệ: {len(invalid_education)}")
print(f"MARRIAGE không hợp lệ: {len(invalid_marriage)}")
print(f"default không hợp lệ: {len(invalid_default)}")
print(f"PAY_* chưa được codebook định nghĩa: {invalid_pay.sum().sum()}")

# 3. XỬ LÝ NHÃN KHÔNG HỢP LỆ

print("\n" + "="*50)
print("--- XỬ LÝ NHÃN KHÔNG HỢP LỆ ---")

# EDUCATION: Các mã 0, 5, 6 không được định nghĩa trong codebook nên được gộp vào nhóm Others (4)
education_invalid_count = df["EDUCATION"].isin([0, 5, 6]).sum()

df.loc[
    df["EDUCATION"].isin([0, 5, 6]),
    "EDUCATION"
] = 4

print(
    f"Đã chuyển {education_invalid_count} "
    f"giá trị EDUCATION bất thường về 4 (Others)."
)


# MARRIAGE: Mã 0 không được định nghĩa trong codebook nên được gộp vào nhóm Others (3)
marriage_invalid_count = (df["MARRIAGE"] == 0).sum()

df.loc[
    df["MARRIAGE"] == 0,
    "MARRIAGE"
] = 3

print(
    f"Đã chuyển {marriage_invalid_count} "
    f"giá trị MARRIAGE bất thường về 3 (Others)."
)

# Kiểm tra lại sau khi xử lý
print("\nKiểm tra lại EDUCATION:")
print(df["EDUCATION"].value_counts().sort_index())

print("\nKiểm tra lại MARRIAGE:")
print(df["MARRIAGE"].value_counts().sort_index())

# 4. Kiểm tra tính hợp lý của biến Age và Limit Bal

print("\n" + "="*50)
print("--- KIỂM TRA AGE ---")

print(f"Tuổi nhỏ nhất: {df['AGE'].min()}")
print(f"Tuổi lớn nhất: {df['AGE'].max()}")

# Tuổi âm hoặc bằng 0 là không hợp lý
invalid_age = df[df["AGE"] <= 0]

print(f"Số giá trị AGE <= 0: {len(invalid_age)}")

if len(invalid_age) > 0:
    print(invalid_age["AGE"].value_counts().sort_index())


print("\n" + "="*50)
print("--- KIỂM TRA LIMIT_BAL ---")

print(f"LIMIT_BAL nhỏ nhất: {df['LIMIT_BAL'].min()}")
print(f"LIMIT_BAL lớn nhất: {df['LIMIT_BAL'].max()}")

# Hạn mức tín dụng không thể âm
invalid_limit = df[df["LIMIT_BAL"] < 0]

print(f"Số giá trị LIMIT_BAL < 0: {len(invalid_limit)}")

if len(invalid_limit) > 0:
    print(invalid_limit["LIMIT_BAL"].value_counts().sort_index())

# 5. Kiểm tra giá trị ngoại lai bằng IQR
print("\n" + "="*50)
print("--- KIỂM TRA NGOẠI LAI ---")

money_cols = [
    "LIMIT_BAL",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

outlier_result = []

for col in money_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = (
        (df[col] < lower) |
        (df[col] > upper)
    ).sum()

    outlier_result.append({
        "Variable": col,
        "Outlier_Count": outlier_count,
        "Outlier_%": round(
            outlier_count / len(df) * 100, 2
        )
    })

outlier_result = pd.DataFrame(outlier_result)

print(outlier_result)

print("\nLưu ý: Ngoại lai được phát hiện nhưng chưa xóa,")
print("vì các giá trị lớn có thể là dữ liệu tín dụng hợp lệ.")

##### TRAIN/TEST THEO STRATIFIED

from sklearn.model_selection import train_test_split

print("\n" + "="*50)
print("--- CHIA TRAIN / TEST ---")

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")


# 1.Kiểm tra tỷ lệ default
print("\nTỷ lệ default trong toàn bộ dữ liệu:")
print(y.value_counts(normalize=True))

print("\nTỷ lệ default trong TRAIN:")
print(y_train.value_counts(normalize=True))

print("\nTỷ lệ default trong TEST:")
print(y_test.value_counts(normalize=True))

# 2. Kiểm tra rò rỉ dữ liệu
print("\n" + "="*50)
print("--- KIỂM TRA RÒ RỈ DỮ LIỆU ---")

# Kiểm tra dòng trùng giữa TRAIN và TEST
overlap = X_train.index.intersection(X_test.index)

print(
    f"Số dòng xuất hiện ở cả TRAIN và TEST: "
    f"{len(overlap)}"
)

# Kiểm tra biến mục tiêu có nằm trong X không
print(
    f"default có trong X_train: "
    f"{'default' in X_train.columns}"
)

print(
    f"default có trong X_test: "
    f"{'default' in X_test.columns}"
)

# Kiểm tra tổng số dòng
print(f"\nSố dòng TRAIN: {len(X_train)}")
print(f"Số dòng TEST: {len(X_test)}")
print(
    f"Tổng TRAIN + TEST: "
    f"{len(X_train) + len(X_test)}"
)
print(f"Số dòng dữ liệu gốc: {len(df)}")

#### XUẤT DỮ LIỆU

print("\n" + "="*50)
print("--- XUẤT DỮ LIỆU ---")

# Lưu dữ liệu sau khi làm sạch
df.to_csv(
    "data/cleaned_data.csv",
    index=False
)

# Lưu tập Train
train_data = X_train.copy()
train_data["default"] = y_train

train_data.to_csv(
    "data/train.csv",
    index=False
)

# Lưu tập Test
test_data = X_test.copy()
test_data["default"] = y_test

test_data.to_csv(
    "data/test.csv",
    index=False
)

print("Đã xuất dữ liệu thành công:")
print("- data/cleaned_data.csv")
print("- data/train.csv")
print("- data/test.csv")