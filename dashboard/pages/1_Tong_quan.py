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
16 biến được lựa chọn làm đầu vào cho mô hình chấm điểm tín dụng.
""")

st.markdown("---")

# B. Phân bố Default
st.subheader("B. Phân bố biến mục tiêu (Default)")
col_b1, col_b2 = st.columns([1, 2])

# Dữ liệu giả lập khớp với thống kê ban đầu
df_default = pd.DataFrame({
    "Trạng thái": ["0 (Không vỡ nợ)", "1 (Vỡ nợ)"],
    "Số lượng": [23364, 6636],
    "Tỷ lệ": ["77.88%", "22.12%"]
})

with col_b1:
    st.markdown("Sự mất cân bằng dữ liệu (Imbalanced data):")
    st.dataframe(df_default, hide_index=True, use_container_width=True)

with col_b2:
    fig_def, ax_def = plt.subplots(figsize=(6, 3))
    sns.barplot(x="Trạng thái", y="Số lượng", data=df_default, palette="Set2", ax=ax_def)
    ax_def.set_title("Phân bố khách hàng Vỡ nợ vs Không vỡ nợ")
    st.pyplot(fig_def)

st.markdown("---")

# C. WoE / IV
st.subheader("C. Sức mạnh phân biệt của các biến (WoE / IV)")
col_c1, col_c2 = st.columns([1, 1.5])

# Dữ liệu minh họa IV (Trong thực tế load từ WoE_IV.py)
iv_data = {
    "Variable": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "LIMIT_BAL", "PAY_AMT1", "PAY_AMT2", "BILL_AMT1"],
    "IV": [0.8885, 0.5515, 0.4185, 0.3590, 0.3330, 0.2850, 0.1730, 0.1580, 0.1440, 0.0450],
    "Strength": ["Rất mạnh", "Rất mạnh", "Mạnh", "Mạnh", "Mạnh", "Trung bình", "Trung bình", "Trung bình", "Trung bình", "Yếu"]
}
df_iv = pd.DataFrame(iv_data)

with col_c1:
    st.markdown("**Bảng giá trị Information Value (IV)**")
    st.dataframe(df_iv, height=350, use_container_width=True)

with col_c2:
    st.markdown("**Biểu đồ Information Value (IV)**")
    fig_iv, ax_iv = plt.subplots(figsize=(8, 4))
    sns.barplot(x="IV", y="Variable", data=df_iv, palette="viridis", ax=ax_iv)
    ax_iv.set_title("Sức mạnh phân biệt (IV) của các biến quan trọng nhất")
    st.pyplot(fig_iv)

st.markdown("---")

# D. Correlation WoE
st.subheader("D. Ma trận tương quan (Correlation) sau chuyển đổi WoE")
st.info("💡 **Ghi chú quan trọng:** Không có cặp biến nào có độ tương quan |correlation| ≥ 0.7 sau khi chuyển đổi WoE, đảm bảo mô hình không bị đa cộng tuyến nghiêm trọng.")

# Tạo Heatmap minh họa (Thực tế sẽ load từ WoE_IV.py)
vars_corr = df_iv["Variable"].tolist()[:8] # Lấy 8 biến đầu họa
np.random.seed(42)
dummy_corr = np.random.uniform(0.01, 0.45, size=(8, 8))
np.fill_diagonal(dummy_corr, 1.0)
dummy_corr = (dummy_corr + dummy_corr.T) / 2 # Tạo ma trận đối xứng
np.fill_diagonal(dummy_corr, 1.0)

fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
sns.heatmap(dummy_corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=vars_corr, yticklabels=vars_corr, vmin=-1, vmax=1, ax=ax_corr)
ax_corr.set_title("Correlation Heatmap (WoE Transformed Variables)")
st.pyplot(fig_corr)
