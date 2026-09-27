import pandas as pd
import numpy as np

print("Đang tải dữ liệu...")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")

# 1. PHÂN LOẠI BIẾN:

categorical_cols = ["SEX","EDUCATION","MARRIAGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

numerical_cols = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

# 2. CHIA BIN CHO BIẾN SỐ

print("\n" + "="*50)
print("--- CHIA BIN CHO BIẾN SỐ ---")

# Lưu các khoảng bin được tạo ra từ TRAIN. Sau này TEST phải sử dụng đúng các khoảng bin này, không được tự chia lại TEST.
train_bins = {}

for col in numerical_cols:
    _, bins = pd.qcut(
        train[col],
        q=10,
        retbins=True,
        duplicates="drop"
    )

    bins[0] = -np.inf
    bins[-1] = np.inf

    train_bins[col] = bins

    # Tạo biến bin cho TRAIN
    train[col + "_bin"] = pd.cut(
        train[col],
        bins=bins,
        include_lowest=True
    )

    # TEST sử dụng chính các khoảng bin được học từ TRAIN
    test[col + "_bin"] = pd.cut(
        test[col],
        bins=bins,
        include_lowest=True
    )

print("Đã chia bin cho các biến số.")


# 4. HÀM TÍNH WoE VÀ IV

def calculate_woe_iv(data, variable, target="default"):
    """
    Tính WoE và IV cho một biến.

    variable:
        Tên biến hoặc tên biến đã được chia bin.

    target:
        Biến mục tiêu, ở đây là default.
    """

    # Đếm số khách hàng không default và default trong từng nhóm của biến.
    grouped = data.groupby(
        variable,
        observed=False
    )[target].agg(
        ["count", "sum"]
    )

    # Số default trong từng nhóm
    grouped["bad"] = grouped["sum"]

    # Số không default trong từng nhóm
    grouped["good"] = grouped["count"] - grouped["bad"]

    # Tổng số good và bad trong toàn bộ TRAIN
    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    # Tỷ lệ good của từng nhóm
    grouped["dist_good"] = grouped["good"] / total_good

    # Tỷ lệ bad của từng nhóm
    grouped["dist_bad"] = grouped["bad"] / total_bad

    # Tránh trường hợp dist_good hoặc dist_bad = 0 khiến phép tính log bị lỗi.
    epsilon = 1e-10

    grouped["woe"] = np.log(
        (grouped["dist_good"] + epsilon) /
        (grouped["dist_bad"] + epsilon)
    )

    # IV contribution của từng nhóm
    grouped["iv"] = (
        (grouped["dist_good"] - grouped["dist_bad"])
        * grouped["woe"]
    )

    # IV của toàn bộ biến = tổng IV của các nhóm
    iv = grouped["iv"].sum()

    return grouped, iv

# 5. KIỂM TRA BIN CỦA CÁC BIẾN SỐ

print("\n" + "="*50)
print("--- KIỂM TRA BIN CỦA CÁC BIẾN SỐ ---")

for col in numerical_cols:

    bin_col = col + "_bin"

    # Đếm số quan sát, số good và số bad trong từng bin
    check = train.groupby(
        bin_col,
        observed=False
    )["default"].agg(
        count="count",
        bad="sum"
    )

    check["good"] = check["count"] - check["bad"]

    # Tính tỷ lệ bad trong từng bin
    check["bad_rate"] = (
        check["bad"] / check["count"]
    )

    print("\n" + "-"*60)
    print(f"Biến: {col}")
    print(check)

# 6. KIỂM TRA WoE CỦA CÁC BIẾN SỐ

print("\n" + "="*50)
print("--- KIỂM TRA WoE CỦA CÁC BIẾN SỐ ---")

for col in numerical_cols:

    group_col = col + "_bin"

    woe_table, iv = calculate_woe_iv(
        train,
        group_col,
        target="default"
    )

    print("\n" + "-"*60)
    print(f"WoE của {col}")

    print(
        woe_table[
            ["count", "good", "bad", "woe", "iv"]
        ]
    )

    print(f"IV = {iv:.6f}")

# 7. KIỂM TRA CÁC NHÓM NHỎ CỦA PAY_0 - PAY_6

print("\n" + "="*50)
print("--- KIỂM TRA NHÓM NHỎ CỦA CÁC BIẾN PAY ---")

pay_cols = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

# Ngưỡng tối thiểu để cảnh báo một nhóm có ít quan sát.
# 50 chỉ là ngưỡng kiểm tra ban đầu, không phải quy tắc bắt buộc.
min_count = 50

for col in pay_cols:

    print("\n" + "-"*50)
    print(f"Biến: {col}")

    # Đếm số quan sát trong từng nhóm
    group_count = train[col].value_counts().sort_index()

    print("\nSố quan sát trong từng nhóm:")
    print(group_count)

    # Xác định các nhóm có ít hơn 50 quan sát
    small_groups = group_count[
        group_count < min_count
    ]

    if len(small_groups) > 0:
        print("\nCác nhóm có ít hơn 50 quan sát:")
        print(small_groups)
    else:
        print("\nKhông có nhóm nào dưới 50 quan sát.")

# 8. GỘP CÁC NHÓM TRẢ CHẬM CAO CỦA PAY

print("\n" + "="*50)
print("--- GỘP NHÓM PAY TỪ 5 TRỞ LÊN ---")

pay_cols = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

# Tạo biến mới để giữ nguyên dữ liệu PAY ban đầu.
# Các mức 5, 6, 7, 8 được gộp thành một nhóm "5+".
for col in pay_cols:

    train[col + "_group"] = train[col].apply(
        lambda x: "5+" if x >= 5 else str(x)
    )

    test[col + "_group"] = test[col].apply(
        lambda x: "5+" if x >= 5 else str(x)
    )

    print(f"\n{col} sau khi gộp:")

    print(
        train[col + "_group"]
        .value_counts()
        .sort_index()
    )

# TÍNH WoE/IV SAU KHI GỘP PAY

print("\n" + "="*50)
print("--- WoE/IV SAU KHI GỘP PAY ---")

grouped_woe_tables = {}
grouped_iv_results = {}

for col in pay_cols:

    group_col = col + "_group"

    woe_table, iv = calculate_woe_iv(
        train,
        group_col,
        target="default"
    )

    grouped_woe_tables[col] = woe_table
    grouped_iv_results[col] = iv

    print(f"\n--- {col} ---")
    print(
        woe_table[
            ["count", "good", "bad", "woe", "iv"]
        ]
    )

    print(f"IV = {iv:.6f}")

# Chốt gộp PAY_0, PAY_2, PAY_5
for col in ["PAY_0", "PAY_2", "PAY_5"]:

    train[col + "_final"] = train[col].apply(
        lambda x: "5+" if x >= 5 else str(x)
    )

    test[col + "_final"] = test[col].apply(
        lambda x: "5+" if x >= 5 else str(x)
    )

# 9. Gộp PAY_3 và PAY_4: Gộp nhóm 1 vào nhóm 2

print("\n" + "="*50)
print("--- KIỂM TRA GỘP NHÓM 1 VÀO NHÓM 2 ---")

# PAY_3 và PAY_4 có nhóm 1 quá nhỏ. Ta thử gộp nhóm 1 vào nhóm 2.

for col in ["PAY_3", "PAY_4"]:

    group_col = col + "_group"

    train[group_col + "_merge"] = train[group_col].apply(
        lambda x: "2+" if x in ["1", "2"] else x
    )

    test[group_col + "_merge"] = test[group_col].apply(
        lambda x: "2+" if x in ["1", "2"] else x
    )

    print(f"\n--- {col} ---")

    check = train.groupby(
        group_col + "_merge",
        observed=False
    )["default"].agg(
        count="count",
        bad="sum"
    )

    check["good"] = check["count"] - check["bad"]

    print(check)

# TÍNH WoE/IV SAU KHI gộp PAY_3 và PAY_4

print("\n" + "="*50)
print("--- WoE/IV SAU KHI GỘP 1 VÀO 2 CỦA PAY_3 VÀ PAY_4 ---")

for col in ["PAY_3", "PAY_4"]:

    group_col = col + "_group_merge"

    woe_table, iv = calculate_woe_iv(
        train,
        group_col,
        target="default"
    )

    print(f"\n--- {col} ---")
    print(
        woe_table[
            ["count", "good", "bad", "woe", "iv"]
        ]
    )

    print(f"IV = {iv:.6f}")

# Chốt sau khi gộp PAY_3, PAY_4
for col in ["PAY_3", "PAY_4"]:

    train[col + "_final"] = train[col + "_group_merge"]
    test[col + "_final"] = test[col + "_group_merge"]


# 10. KIỂM TRA GỘP PAY_6: 4 VÀO 5+

print("\n" + "="*50)
print("--- KIỂM TRA GỘP PAY_6: 4 VÀO 5+ ---")

train["PAY_6_final"] = train["PAY_6_group"].apply(
    lambda x: "5+" if x in ["4", "5+"] else x
)

test["PAY_6_final"] = test["PAY_6_group"].apply(
    lambda x: "5+" if x in ["4", "5+"] else x
)

check = train.groupby(
    "PAY_6_final",
    observed=False
)["default"].agg(
    count="count",
    bad="sum"
)

check["good"] = check["count"] - check["bad"]

print(check)

# WoE/IV SAU KHI GỘP PAY_6

print("\n" + "="*50)
print("--- WoE/IV PAY_6 SAU KHI GỘP 4 VÀO 5+ ---")

woe_table, iv = calculate_woe_iv(
    train,
    "PAY_6_final",
    target="default"
)

print(
    woe_table[
        ["count", "good", "bad", "woe", "iv"]
    ]
)

print(f"IV = {iv:.6f}")

# 11. XÁC ĐỊNH BIN PAY DÙNG CHO WoE/IV CUỐI

print("\n" + "="*50)
print("--- BIẾN DÙNG CHO WoE/IV CUỐI ---")

woe_variables = {}

# Các biến categorical thông thường
for col in ["SEX", "EDUCATION", "MARRIAGE"]:
    woe_variables[col] = col

# PAY sử dụng bin cuối cùng
woe_variables["PAY_0"] = "PAY_0_final"
woe_variables["PAY_2"] = "PAY_2_final"
woe_variables["PAY_3"] = "PAY_3_final"
woe_variables["PAY_4"] = "PAY_4_final"
woe_variables["PAY_5"] = "PAY_5_final"
woe_variables["PAY_6"] = "PAY_6_final"

# Các biến numerical sử dụng bin hiện tại
for col in numerical_cols:
    woe_variables[col] = col + "_bin"

print("\nDanh sách biến:")
for original_col, woe_col in woe_variables.items():
    print(f"{original_col} -> {woe_col}")

# 12. TÍNH WoE VÀ IV CUỐI CHO TOÀN BỘ BIẾN

print("\n" + "="*50)
print("--- TÍNH WoE/IV CUỐI CHO TOÀN BỘ BIẾN ---")

final_woe_tables = {}
final_iv_results = {}

for original_col, woe_col in woe_variables.items():

    woe_table, iv = calculate_woe_iv(
        train,
        woe_col,
        target="default"
    )

    final_woe_tables[original_col] = woe_table
    final_iv_results[original_col] = iv


# Tạo bảng IV
iv_table = pd.DataFrame(
    list(final_iv_results.items()),
    columns=["Variable", "IV"]
)

# Sắp xếp IV từ cao xuống thấp
iv_table = iv_table.sort_values(
    by="IV",
    ascending=False
).reset_index(drop=True)

print("\nBẢNG IV CUỐI:")
print(iv_table.to_string(index=False))

# PHÂN LOẠI MỨC ĐỘ IV

def iv_strength(iv):

    if iv < 0.02:
        return "Rất yếu"
    elif iv < 0.10:
        return "Yếu"
    elif iv < 0.30:
        return "Trung bình"
    elif iv < 0.50:
        return "Mạnh"
    else:
        return "Rất mạnh"
    
iv_table["Strength"] = iv_table["IV"].apply(iv_strength)

print("\n" + "="*50)
print("--- PHÂN LOẠI IV ---")
print(iv_table.to_string(index=False))

# KIỂM TRA CHI TIẾT CÁC BIẾN CÓ IV CAO

print("\n" + "="*50)
print("--- KIỂM TRA CHI TIẾT CÁC BIẾN IV CAO ---")

for col in ["PAY_0", "PAY_2"]:

    print("\n" + "-"*60)
    print(f"Biến: {col}")

    print(
        final_woe_tables[col][
            ["count", "good", "bad", "dist_good", "dist_bad", "woe", "iv"]
        ].to_string()
    )


    # KIỂM TRA ĐỘ ỔN ĐỊNH BAD RATE GIỮA TRAIN VÀ TEST

print("\n" + "="*50)
print("--- KIỂM TRA BAD RATE TRAIN VS TEST ---")

for col in ["PAY_0_final", "PAY_2_final"]:

    print("\n" + "-"*60)
    print(f"Biến: {col}")

    # TRAIN
    train_check = train.groupby(
        col,
        observed=False
    )["default"].agg(
        count="count",
        bad="sum"
    )

    train_check["good"] = (
        train_check["count"] - train_check["bad"]
    )

    train_check["bad_rate"] = (
        train_check["bad"] / train_check["count"]
    )

    # TEST
    test_check = test.groupby(
        col,
        observed=False
    )["default"].agg(
        count="count",
        bad="sum"
    )

    test_check["good"] = (
        test_check["count"] - test_check["bad"]
    )

    test_check["bad_rate"] = (
        test_check["bad"] / test_check["count"]
    )

    # Đổi tên để phân biệt TRAIN và TEST
    train_check = train_check[
        ["count", "bad", "bad_rate"]
    ].rename(
        columns={
            "count": "train_count",
            "bad": "train_bad",
            "bad_rate": "train_bad_rate"
        }
    )

    test_check = test_check[
        ["count", "bad", "bad_rate"]
    ].rename(
        columns={
            "count": "test_count",
            "bad": "test_bad",
            "bad_rate": "test_bad_rate"
        }
    )

    comparison = train_check.join(
        test_check,
        how="outer"
    )

    print(comparison)

# KIỂM TRA NGƯỠNG IV = 0.02

print("\n" + "="*50)
print("--- KIỂM TRA NGƯỠNG IV = 0.02 ---")

selected_iv = iv_table[
    iv_table["IV"] >= 0.02
]

weak_iv = iv_table[
    iv_table["IV"] < 0.02
]

print(f"\nSố biến có IV >= 0.02: {len(selected_iv)}")
print(f"Số biến có IV < 0.02: {len(weak_iv)}")

print("\nBiến có IV >= 0.02:")
print(selected_iv.to_string(index=False))

print("\nBiến có IV < 0.02:")
print(weak_iv.to_string(index=False))

selected_variables = selected_iv["Variable"].tolist()

print("\nDanh sách biến được chọn:")
print(selected_variables)

# 13. CHUYỂN CÁC BIẾN ĐƯỢC CHỌN SANG WoE

print("\n" + "="*50)
print("--- TẠO BIẾN WoE ---")

train_woe = pd.DataFrame(index=train.index)
test_woe = pd.DataFrame(index=test.index)

for original_col in selected_variables:

    woe_col = woe_variables[original_col]

    # Lấy bảng WoE đã tính trên TRAIN
    woe_table = final_woe_tables[original_col]

    # Tạo dictionary: giá trị/bin -> WoE
    woe_mapping = woe_table["woe"].to_dict()

    # TRAIN
    train_woe[original_col] = train[woe_col].map(woe_mapping)

    # TEST
    test_woe[original_col] = test[woe_col].map(woe_mapping)

print("Đã tạo biến WoE cho các biến có IV >= 0.02.")

print("\nKích thước TRAIN WoE:")
print(train_woe.shape)

print("\nKích thước TEST WoE:")
print(test_woe.shape)

# KIỂM TRA WoE BỊ THIẾU

print("\n" + "="*50)
print("--- KIỂM TRA WoE BỊ THIẾU ---")

print("TRAIN:")
print(train_woe.isna().sum()[train_woe.isna().sum() > 0])

print("\nTEST:")
print(test_woe.isna().sum()[test_woe.isna().sum() > 0])

# 14. KIỂM TRA TƯƠNG QUAN TRÊN BIẾN WoE

print("\n" + "="*50)
print("--- KIỂM TRA TƯƠNG QUAN TRÊN WoE ---")

woe_correlation = train_woe.corr()

print("\nMa trận tương quan WoE:")
print(woe_correlation.round(2))

# TÌM CẶP BIẾN WoE CÓ TƯƠNG QUAN CAO

print("\n" + "="*50)
print("--- CÁC CẶP BIẾN WoE CÓ TƯƠNG QUAN CAO ---")

woe_correlation_pairs = []

for i in range(len(woe_correlation.columns)):
    for j in range(i + 1, len(woe_correlation.columns)):

        var1 = woe_correlation.columns[i]
        var2 = woe_correlation.columns[j]

        corr_value = woe_correlation.iloc[i, j]

        if abs(corr_value) >= 0.7:
            woe_correlation_pairs.append(
                [var1, var2, corr_value]
            )

if woe_correlation_pairs:

    woe_correlation_pairs_df = pd.DataFrame(
        woe_correlation_pairs,
        columns=["Variable_1", "Variable_2", "Correlation"]
    )

    woe_correlation_pairs_df = woe_correlation_pairs_df.sort_values(
        by="Correlation",
        key=lambda x: abs(x),
        ascending=False
    )

    print(
        woe_correlation_pairs_df.to_string(index=False)
    )

else:
    print("Không có cặp biến WoE nào có |Correlation| >= 0.7.")

# 15. XUẤT DỮ LIỆU WoE CHO CÁC MÔ HÌNH TIẾP THEO

print("\n" + "="*50)
print("--- XUẤT DỮ LIỆU WoE ---")

# Chỉ giữ các biến đã được chọn theo IV
woe_train = train_woe[selected_variables].copy()
woe_test = test_woe[selected_variables].copy()

# Thêm biến mục tiêu
woe_train["default"] = train["default"].values
woe_test["default"] = test["default"].values

# Xuất file
woe_train.to_csv(
    "data/woe_train.csv",
    index=False
)

woe_test.to_csv(
    "data/woe_test.csv",
    index=False
)

print("\nĐÃ XUẤT FILE WoE:")
print("woe_train.csv:", woe_train.shape)
print("woe_test.csv:", woe_test.shape)

print("\nCác biến trong file:")
print(woe_train.columns.tolist())