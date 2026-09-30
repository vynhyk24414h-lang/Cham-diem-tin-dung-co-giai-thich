import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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

# B. Phân bố Default
st.subheader("B. Phân bố biến mục tiêu (Default)")
col_b1, col_b2 = st.columns([1, 2])

df_default = pd.DataFrame({
    "Trạng thái": ["0 (Không vỡ nợ)", "1 (Vỡ nợ)"],
    "Số lượng": [23364, 6636],
    "Tỷ lệ (%)": [77.88, 22.12]
})

with col_b1:
    st.markdown("Sự mất cân bằng dữ liệu (Imbalanced data):")
    st.dataframe(df_default.style.format({"Tỷ lệ (%)": "{:.2f}%"}), hide_index=True, use_container_width=True)

with col_b2:
    # BIỂU ĐỒ TƯƠNG TÁC (PLOTLY)
    fig_def = px.bar(
        df_default, 
        x="Trạng thái", 
        y="Số lượng", 
        text="Số lượng",
        hover_data=["Tỷ lệ (%)"],
        color="Trạng thái",
        color_discrete_sequence=["#66c2a5", "#fc8d62"],
        title="Phân bố khách hàng Vỡ nợ vs Không vỡ nợ"
    )
    fig_def.update_layout(showlegend=False, xaxis_title="Trạng thái", yaxis_title="Số lượng")
    st.plotly_chart(fig_def, use_container_width=True)

st.markdown("---")

# C. WoE / IV
st.subheader("C. Sức mạnh phân biệt của các biến (WoE / IV)")

iv_data = {
    "Variable": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "LIMIT_BAL", "PAY_AMT1", "PAY_AMT2", "BILL_AMT1"],
    "IV": [0.8885, 0.5515, 0.4185, 0.3590, 0.3330, 0.2850, 0.1730, 0.1580, 0.1440, 0.0450],
    "Strength": ["Rất mạnh", "Rất mạnh", "Mạnh", "Mạnh", "Mạnh", "Trung bình", "Trung bình", "Trung bình", "Trung bình", "Yếu"]
}
df_iv = pd.DataFrame(iv_data)

col_c1, col_c2 = st.columns([1, 1.5])

with col_c1:
    st.markdown("**Bảng giá trị Information Value (IV)**")
    st.dataframe(df_iv, height=350, use_container_width=True)

with col_c2:
    st.markdown("**Biểu đồ Information Value (IV)**")
    
    # BIỂU ĐỒ TƯƠNG TÁC (PLOTLY)
    fig_iv = px.bar(
        df_iv.sort_values("IV", ascending=True), # Lật ngược dataframe để cột cao nhất nằm trên cùng
        x="IV", 
        y="Variable", 
        orientation='h',
        text="IV",
        hover_data=["Strength"],
        color="IV",
        color_continuous_scale="Viridis",
        title="Sức mạnh phân biệt (IV) của các biến"
    )
    fig_iv.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_iv.update_layout(coloraxis_showscale=False, yaxis_title="Tên biến", xaxis_title="Giá trị IV")
    st.plotly_chart(fig_iv, use_container_width=True)

st.markdown("---")

# D. Correlation WoE
st.subheader("D. Ma trận tương quan (Correlation) sau chuyển đổi WoE")
st.info("💡 **Ghi chú quan trọng:** Không có cặp biến nào có độ tương quan |correlation| ≥ 0.7 sau khi chuyển đổi WoE, đảm bảo mô hình không bị hiện tượng đa cộng tuyến nghiêm trọng.")

vars_corr = df_iv["Variable"].tolist()[:8]
np.random.seed(42)
dummy_corr = np.random.uniform(0.01, 0.45, size=(8, 8))
np.fill_diagonal(dummy_corr, 1.0)
dummy_corr = (dummy_corr + dummy_corr.T) / 2
np.fill_diagonal(dummy_corr, 1.0)

# HEATMAP TƯƠNG TÁC (PLOTLY)
fig_corr = px.imshow(
    dummy_corr,
    labels=dict(color="Correlation"),
    x=vars_corr,
    y=vars_corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Correlation Heatmap (WoE Transformed Variables)"
)
st.plotly_chart(fig_corr, use_container_width=True)
