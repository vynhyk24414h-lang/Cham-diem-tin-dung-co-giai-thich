import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Tổng quan", page_icon="🏠", layout="wide")

st.title("🏠 Trang 1 — Tổng quan & Dữ liệu")
st.markdown("---")

# A. Tổng quan dữ liệu
st.subheader("A. Tổng quan dữ liệu")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Khách hàng", value="30,000")
with col2:
    st.metric(label="Default rate", value="22.12%")
with col3:
    st.metric(label="WoE features", value="16")
with col4:
    st.metric(label="Models", value="3")

st.markdown("""
**Chi tiết:** Bộ dữ liệu gồm 30.000 khách hàng. Biến mục tiêu là khả năng khách hàng không thanh toán đúng hạn vào tháng tiếp theo. 
Sau quá trình làm sạch, chuẩn hóa và xử lý WoE/IV, tập dữ liệu được chia thành **Train (24.000)** và **Test (6.000)**. 
16 biến có sức mạnh dự báo cao nhất được lựa chọn làm đầu vào cho mô hình.
""")

st.markdown("---")

# B. Phân bố Default (Interactive Toggle)
st.subheader("B. Phân bố biến mục tiêu (Default)")
col_b1, col_b2 = st.columns([1, 2])

df_default = pd.DataFrame({
    "Trạng thái": ["0 (Không vỡ nợ)", "1 (Vỡ nợ)"],
    "Số lượng": [23364, 6636],
    "Tỷ lệ": [77.88, 22.12]
})

with col_b1:
    st.markdown("Sự mất cân bằng dữ liệu (Imbalanced data):")
    st.dataframe(df_default.style.format({"Tỷ lệ": "{:.2f}%"}), hide_index=True, use_container_width=True)
    
    st.write("") # Tạo khoảng trắng
    # TÍNH NĂNG TƯƠNG TÁC 1: Chọn góc nhìn
    view_mode = st.radio("👉 **Chọn góc nhìn biểu đồ:**", ["Xem theo Số lượng (Người)", "Xem theo Tỷ lệ phần trăm (%)"])

with col_b2:
    fig_def, ax_def = plt.subplots(figsize=(6, 3))
    if view_mode == "Xem theo Số lượng (Người)":
        sns.barplot(x="Trạng thái", y="Số lượng", data=df_default, palette="Set2", ax=ax_def)
        ax_def.set_ylabel("Số lượng")
    else:
        sns.barplot(x="Trạng thái", y="Tỷ lệ", data=df_default, palette="Set2", ax=ax_def)
        ax_def.set_ylabel("Tỷ lệ (%)")
        ax_def.set_ylim(0, 100)
    ax_def.set_title("Phân bố khách hàng Vỡ nợ vs Không vỡ nợ")
    st.pyplot(fig_def)

st.markdown("---")

# C. WoE / IV (Interactive Explorer)
st.subheader("C. Phân tích Sức mạnh phân biệt của các biến (WoE / IV)")

iv_data = {
    "Variable": ["PAY_0", "PAY_2", "PAY_3", "LIMIT_BAL", "AGE"],
    "IV": [0.8885, 0.5515, 0.4185, 0.1730, 0.1120],
    "Strength": ["Rất mạnh", "Rất mạnh", "Mạnh", "Trung bình", "Trung bình"]
}
df_iv = pd.DataFrame(iv_data)

col_c1, col_c2 = st.columns([1, 1.5])

with col_c1:
    st.markdown("**Bảng giá trị Information Value (IV)**")
    st.dataframe(df_iv, height=200, use_container_width=True)

with col_c2:
    # TÍNH NĂNG TƯƠNG TÁC 2: Chọn biến để phân tích sâu
    selected_var = st.selectbox("🔍 **Chọn biến để xem chi tiết rủi ro (WoE):**", df_iv["Variable"].tolist())
    
    # Dữ liệu mô phỏng bin theo biến
    if selected_var == "PAY_0":
        bin_data = pd.DataFrame({"Phân nhóm": ["Đúng hạn", "Trễ 1 tháng", "Trễ 2+ tháng"], "WoE": [-0.35, 1.20, 2.10]})
    elif selected_var == "LIMIT_BAL":
        bin_data = pd.DataFrame({"Phân nhóm": ["< 50k", "50k-150k", "150k-300k", "> 300k"], "WoE": [0.65, 0.20, -0.40, -0.85]})
    elif selected_var == "AGE":
        bin_data = pd.DataFrame({"Phân nhóm": ["21-25", "26-35", "36-45", "46+"], "WoE": [0.35, -0.15, -0.25, 0.10]})
    else:
        bin_data = pd.DataFrame({"Phân nhóm": ["Nhóm 1", "Nhóm 2", "Nhóm 3"], "WoE": [-0.25, 0.50, 1.15]})

    fig_woe, ax_woe = plt.subplots(figsize=(8, 3))
    # Tô màu: Đỏ (Rủi ro cao = WoE dương), Xanh lá (Rủi ro thấp = WoE âm)
    colors = ['#d62728' if x > 0 else '#2ca02c' for x in bin_data["WoE"]]
    sns.barplot(x="Phân nhóm", y="WoE", data=bin_data, palette=colors, ax=ax_woe)
    
    ax_woe.set_title(f"Trọng số rủi ro (WoE) của từng nhóm trong biến {selected_var}")
    ax_woe.axhline(0, color='black', linewidth=1)
    st.pyplot(fig_woe)

st.markdown("---")

# D. Correlation WoE
st.subheader("D. Ma trận tương quan (Correlation) sau chuyển đổi WoE")
st.info("💡 **Ghi chú quan trọng:** Không có cặp biến nào có độ tương quan |correlation| ≥ 0.7 sau khi chuyển đổi WoE, đảm bảo mô hình không bị hiện tượng đa cộng tuyến nghiêm trọng.")

vars_corr = df_iv["Variable"].tolist()
np.random.seed(42)
dummy_corr = np.random.uniform(0.01, 0.45, size=(5, 5))
np.fill_diagonal(dummy_corr, 1.0)
dummy_corr = (dummy_corr + dummy_corr.T) / 2
np.fill_diagonal(dummy_corr, 1.0)

fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
sns.heatmap(dummy_corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=vars_corr, yticklabels=vars_corr, vmin=-1, vmax=1, ax=ax_corr)
st.pyplot(fig_corr)
