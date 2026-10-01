import streamlit as st


# ============================================================
# PAGE CONFIG CHUNG
# ============================================================

st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MÀU DÙNG CHUNG TOÀN BỘ DASHBOARD
# ============================================================

COLORS = {
    "navy": "#17324D",
    "navy_light": "#244B64",
    "primary": "#315B78",
    "teal": "#168C83",
    "gold": "#C39A3A",
    "red": "#8C3B3B",
    "green": "#2E7D32",
    "text": "#334155",
    "muted": "#64748B",
    "background": "#F8FAFC",
    "card": "#FFFFFF",
    "border": "#DCE5EF",
    "grid": "#E8EDF3",
}


# ============================================================
# GIAO DIỆN CHUNG
# Áp dụng cho cả MODEL / SHAP / FAIRNESS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       NỀN CHUNG
       ======================================================== */

    [data-testid="stAppViewContainer"] {{
        background: {COLORS["background"]};
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    .block-container {{
        max-width: 1600px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    .stApp {{
        color: {COLORS["text"]};
    }}

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {{
        color: {COLORS["text"]};
        line-height: 1.65;
    }}

    h1 {{
        color: {COLORS["navy"]} !important;
        font-size: 2rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem !important;
    }}

    h2 {{
        color: {COLORS["navy"]} !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }}

    h3 {{
        color: {COLORS["navy_light"]} !important;
        font-size: 1.15rem !important;
        font-weight: 650 !important;
    }}


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {{
        background: {COLORS["navy"]};
    }}

    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebarNav"] {{
        padding-top: 0.5rem;
    }}

    [data-testid="stSidebarNav"] a {{
        border-radius: 8px;
        margin: 3px 8px;
        padding: 8px 10px;
        transition: 0.2s ease;
    }}

    [data-testid="stSidebarNav"] a:hover {{
        background: rgba(255,255,255,0.08) !important;
    }}

    [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: rgba(255,255,255,0.15) !important;
    }}


    /* ========================================================
       METRIC CARD
       ======================================================== */

    div[data-testid="stMetric"] {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-left: 4px solid {COLORS["primary"]};
        border-radius: 10px;
        padding: 20px 18px;
        min-height: 125px;
        box-shadow: 0 3px 10px rgba(23, 50, 77, 0.05);
    }}

    div[data-testid="stMetricLabel"] p {{
        color: {COLORS["muted"]} !important;
        font-size: 0.9rem !important;
        font-weight: 550 !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {COLORS["navy"]} !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }}

    div[data-testid="stMetricDelta"] {{
        font-size: 0.85rem !important;
    }}


    /* ========================================================
       TAB
       ======================================================== */

    button[data-baseweb="tab"] {{
        color: {COLORS["muted"]};
        font-size: 0.95rem;
        font-weight: 600;
        padding: 12px 18px;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {COLORS["navy"]} !important;
        border-bottom-color: {COLORS["teal"]} !important;
    }}


    /* ========================================================
       EXPANDER
       ======================================================== */

    div[data-testid="stExpander"] {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 9px;
    }}


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {{
        background: {COLORS["card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 9px;
        overflow: hidden;
    }}


    /* ========================================================
       ALERT
       ======================================================== */

    div[data-testid="stAlert"] {{
        border-radius: 8px;
    }}


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {{
        border-color: {COLORS["border"]};
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR HEADER
# ============================================================

with st.sidebar:

    st.markdown(
        "### 💳 CREDIT SCORING"
    )

    st.caption(
        "Explainable AI Dashboard"
    )

# ============================================================
# NAVIGATION
# ============================================================

pages = {
    "TỔNG QUAN": [
        st.Page(
            "dashboard/home.py",
            title="Tổng quan",
            icon="💳",
            default=True,
        ),
    ],

    "PHÂN TÍCH": [
        st.Page(
            "dashboard/pages/2_Mo_hinh.py",
            title="Mô hình & Chấm điểm",
            icon="🤖",
        ),

        st.Page(
            "dashboard/pages/shap.py",
            title="SHAP — Giải thích mô hình",
            icon="🔎",
        ),

        st.Page(
            "dashboard/pages/fairness.py",
            title="Fairness — Công bằng mô hình",
            icon="⚖️",
        ),
    ],
}


# ============================================================
# CHẠY NAVIGATION
# ============================================================

pg = st.navigation(
    pages,
    position="sidebar",
    expanded=True,
)

pg.run()