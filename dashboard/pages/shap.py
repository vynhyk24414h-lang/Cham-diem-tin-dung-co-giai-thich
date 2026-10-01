from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="SHAP", page_icon="🔎", layout="wide")

# ============================================================
# ĐƯỜNG DẪN & HẰNG SỐ (khớp với shap_analysis.py)
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
SHAP_DIR = ROOT / "reports" / "shap"
TEST_PATH = ROOT / "data" / "test.csv"
XGB_PATH = ROOT / "models" / "xgb_model.pkl"

TARGET_COL = "default"
NOMINAL = ["SEX", "EDUCATION", "MARRIAGE"]
THRESHOLD = 0.5
MAX_ROWS = 6000
RANDOM_STATE = 42

# Màu chuẩn của thư viện shap (giống ảnh PNG)
SHAP_RED = "#FF0051"
SHAP_BLUE = "#008BFB"
DEP_BLUE = "#1E88E5"
GREY_TXT = "#888888"
MPL_BLUE, MPL_ORANGE = "#1f77b4", "#ff7f0e"
FONT_FAMILY = "DejaVu Sans, Arial, sans-serif"

st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 45px !important;
    font-weight: 900 !important;
    color: #FF4B4B;
}
div[data-testid="stMetricLabel"] {
    font-size: 20px !important;
    font-weight: bold !important;
    color: #31333F;
}
h2, h3 { font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DỮ LIỆU
# ============================================================
@st.cache_data
def read_csv(name: str):
    p = SHAP_DIR / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_model():
    return joblib.load(XGB_PATH)


@st.cache_data(show_spinner="Đang tính SHAP values...")
def compute_shap():
    """SHAP bằng pred_contribs của XGBoost (tương đương TreeExplainer)."""
    model = load_model()
    test = pd.read_csv(TEST_PATH)
    features = list(model.feature_names_in_)
    if len(test) > MAX_ROWS:
        test = test.loc[test.sample(MAX_ROWS, random_state=RANDOM_STATE).index]
    X = test[features]
    y = test[TARGET_COL].values
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    values, base = contrib[:, :-1], float(contrib[0, -1])
    prob = model.predict_proba(X)[:, 1]
    return X, y, values, base, prob


# ============================================================
# HÀM PHỤ VẼ BIỂU ĐỒ (giống phong cách matplotlib/shap)
# ============================================================
def fmt_val(x):
    try:
        x = float(x)
    except Exception:
        return str(x)
    out = f"{int(round(x))}" if (abs(x - round(x)) < 1e-9 or abs(x) >= 1000) else f"{x:.3g}"
    return out.replace("-", "−")


def fmt_sh(v):
    return f"{v:+.2f}".rstrip("0").rstrip(".").replace("-", "−")


def base_layout(title, height):
    return dict(
        height=height,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT_FAMILY, color="black", size=14),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        margin=dict(l=20, r=40, t=70, b=60),
    )


def AX_L(**kw):
    d = dict(showline=True, linecolor="#333", linewidth=1, ticks="outside",
             showgrid=False, zeroline=False)
    d.update(kw)
    return d


def split_top(order, top_n):
    """Giống max_display của shap: top_n-1 biến + 1 dòng 'Sum of k other features'."""
    if len(order) > top_n:
        return order[:top_n - 1], order[top_n - 1:]
    return order, np.array([], dtype=int)


def norm_color(fv):
    lo, hi = np.percentile(fv, [5, 95])
    if hi == lo:
        lo, hi = fv.min(), fv.max()
    return np.zeros_like(fv) if hi == lo else np.clip((fv - lo) / (hi - lo), 0, 1)


def beeswarm_offset(sv, rng, n_bins=40):
    hist, edges = np.histogram(sv, bins=n_bins)
    b = np.clip(np.digitize(sv, edges[1:-1]), 0, n_bins - 1)
    return rng.uniform(-1, 1, len(sv)) * (hist[b] / hist.max()) * 0.4


def show(fig):
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ---------- 1. Bar mean|SHAP| ----------
def bar_fig(X, values, top_n):
    mean_abs = np.abs(values).mean(0)
    order = np.argsort(-mean_abs)
    shown, rest = split_top(order, top_n)
    labels = [X.columns[j] for j in shown]
    vals = [mean_abs[j] for j in shown]
    if len(rest):
        labels.append(f"Sum of {len(rest)} other features")
        vals.append(mean_abs[rest].sum())
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", width=0.7, marker_color=SHAP_RED,
        text=[f"+{v:.2f}" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(color=SHAP_RED, size=14),
        hovertemplate="%{y}<br>mean|SHAP| = %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**base_layout("Mức độ quan trọng trung bình |SHAP| — XGBoost",
                                    max(450, 34 * len(labels) + 130)))
    fig.update_xaxes(**AX_L(title="mean(|SHAP value|)", range=[0, max(vals) * 1.18]))
    fig.update_yaxes(**AX_L(autorange="reversed", showgrid=True,
                     gridcolor="#D9D9D9", griddash="dot", ticks="",
                     tickfont=dict(size=14), automargin=True))
    return fig


# ---------- 2. Beeswarm ----------
def beeswarm_fig(X, values, top_n):
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(len(X), 2000)
    sel = rng.choice(len(X), n, replace=False)
    order = np.argsort(-np.abs(values).mean(0))
    shown, rest = split_top(order, top_n)

    rows = []   # (label, shap, norm, raw_text)
    for j in shown:
        fv = X.iloc[sel, j].values.astype(float)
        rows.append((X.columns[j], values[sel, j], norm_color(fv), fv))
    if len(rest):
        sv = values[sel][:, rest].sum(1)
        nm = np.mean([norm_color(X.iloc[sel, j].values.astype(float)) for j in rest], axis=0)
        rows.append((f"Sum of {len(rest)} other features", sv, nm, np.full(n, np.nan)))

    m = len(rows)
    xs, ys, cs, cd = [], [], [], []
    for r, (lab, sv, nm, raw) in enumerate(rows):
        xs.append(sv)
        ys.append((m - 1 - r) + beeswarm_offset(sv, rng))
        cs.append(nm)
        cd.append(np.column_stack([np.full(n, lab, dtype=object), raw]))
    fig = go.Figure(go.Scattergl(
        x=np.concatenate(xs), y=np.concatenate(ys), mode="markers",
        marker=dict(size=5, opacity=0.85, color=np.concatenate(cs),
                    colorscale=[[0, SHAP_BLUE], [1, SHAP_RED]], cmin=0, cmax=1,
                    colorbar=dict(title=dict(text="Feature value", side="right"),
                                  tickvals=[0, 1], ticktext=["Low", "High"],
                                  thickness=12, len=0.95, outlinewidth=0)),
        customdata=np.vstack(cd),
        hovertemplate="<b>%{customdata[0]}</b><br>Giá trị: %{customdata[1]}<br>SHAP: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#999", line_width=1.5)
    fig.update_layout(**base_layout("SHAP toàn cục — XGBoost (beeswarm)", max(500, 40 * m + 140)))
    fig.update_xaxes(**AX_L(title="SHAP value (impact on model output)"))
    fig.update_yaxes(**AX_L(tickmode="array", tickvals=list(range(m)),
                     ticktext=[r[0] for r in rows][::-1], showgrid=True,
                     gridcolor="#D9D9D9", griddash="dot", ticks="",
                     range=[-0.7, m - 0.3], tickfont=dict(size=14), automargin=True))
    return fig


# ---------- 3. Dependence ----------
def dependence_fig(X, values, feat):
    j = list(X.columns).index(feat)
    xv = X[feat].values.astype(float)
    sv = values[:, j]
    uniq = np.unique(xv)
    rng = np.random.default_rng(RANDOM_STATE)
    if len(uniq) <= 20:
        gap = np.min(np.diff(uniq)) if len(uniq) > 1 else 1.0
        xj = xv + rng.uniform(-0.12, 0.12, len(xv)) * gap
        hx = uniq
        hc = np.array([(xv == u).sum() for u in uniq])
        bw = 0.4 * gap
    else:
        xj = xv
        hc, edges = np.histogram(xv, bins=20)
        hx = (edges[:-1] + edges[1:]) / 2
        bw = (edges[1] - edges[0]) * 0.9

    ymin, ymax = sv.min(), sv.max()
    pad = 0.05 * (ymax - ymin)
    y0, y1 = ymin - pad, ymax + pad
    hh = hc / hc.max() * 0.45 * (y1 - y0)

    fig = go.Figure()
    fig.add_bar(x=hx, y=hh, base=y0, width=bw, marker_color="#E6E6E6",
                hoverinfo="skip", showlegend=False)
    fig.add_scattergl(x=xj, y=sv, mode="markers", showlegend=False,
                      marker=dict(size=6, color=DEP_BLUE, opacity=0.9),
                      customdata=xv,
                      hovertemplate=f"{feat} = %{{customdata}}<br>SHAP = %{{y:.3f}}<extra></extra>")
    fig.update_layout(**base_layout(f"Quan hệ giá trị {feat} và SHAP", 520))
    fig.update_xaxes(**AX_L(title=feat))
    fig.update_yaxes(**AX_L(title=dict(text=f"SHAP value for {feat}", standoff=12),
                                     range=[y0, y1], automargin=True))
    return fig


# ---------- 4. Waterfall ----------
def waterfall_fig(X, values, base, i, title, max_display=10):
    sv = values[i]
    order = np.argsort(-np.abs(sv))
    shown, rest = split_top(order, max_display)
    rows = [(X.columns[j], fmt_val(X.iloc[i, j]), float(sv[j])) for j in shown]
    if len(rest):
        rows.append((f"{len(rest)} other features", "", float(sv[rest].sum())))
    n = len(rows)

    starts, ends, cur = [], [], base          # tích lũy từ dưới lên
    for r in reversed(rows):
        starts.append(cur)
        cur += r[2]
        ends.append(cur)
    starts, ends, fx = starts[::-1], ends[::-1], cur

    allx = starts + ends + [base, fx]
    lo, hi = min(allx), max(allx)
    xr = hi - lo
    pad = 0.12 * xr
    head_max = 0.035 * xr

    shapes, anns = [], []
    for r, (name, val, v) in enumerate(rows):
        y = n - 1 - r
        s, e = starts[r], ends[r]
        h = 0.36
        hd = min(head_max, abs(v) * 0.7)
        sgn = 1 if v >= 0 else -1
        color = SHAP_RED if v >= 0 else SHAP_BLUE
        path = (f"M {s},{y - h} L {e - sgn * hd},{y - h} L {e},{y} "
                f"L {e - sgn * hd},{y + h} L {s},{y + h} Z")
        shapes.append(dict(type="path", path=path, fillcolor=color,
                           line=dict(color=color, width=0), layer="above"))
        txt = fmt_sh(v)
        if abs(v) >= 0.13 * xr:
            anns.append(dict(x=(s + e) / 2, y=y, text=txt, showarrow=False,
                             font=dict(color="white", size=14)))
        else:
            anns.append(dict(x=e, y=y, text=txt, showarrow=False,
                             xanchor="left" if v >= 0 else "right",
                             xshift=6 if v >= 0 else -6,
                             font=dict(color=color, size=14)))
    for xv_ in (base, fx):
        shapes.append(dict(type="line", x0=xv_, x1=xv_, y0=-0.6, y1=n - 0.4,
                           line=dict(color="#BBBBBB", width=1, dash="dot"), layer="below"))
    anns.append(dict(x=fx, y=n - 0.45, yanchor="bottom", showarrow=False,
                     text=f"f(x) = {fx:.3f}".replace("-", "−"), font=dict(color=GREY_TXT, size=13)))
    anns.append(dict(x=base, y=-0.55, yanchor="top", showarrow=False,
                     text=f"E[f(X)] = {base:.3f}", font=dict(color=GREY_TXT, size=13)))

    ticktext = [(f"<span style='color:{GREY_TXT}'>{val} = </span>{name}" if val else name)
                for name, val, _ in rows]
    fig = go.Figure()
    lay = base_layout(title, 60 * n + 170)
    lay["margin"] = dict(l=20, r=40, t=80, b=80)
    fig.update_layout(**lay, shapes=shapes, annotations=anns)
    fig.update_xaxes(**AX_L(range=[lo - pad, hi + pad], showline=True))
    fig.update_yaxes(tickmode="array", tickvals=[n - 1 - r for r in range(n)],
                     ticktext=ticktext, range=[-1.1, n + 0.1], showgrid=True,
                     gridcolor="#E3E3E3", griddash="dot", showline=False, zeroline=False,
                     ticks="", tickfont=dict(size=14), automargin=True)
    fig.add_scatter(x=[lo], y=[0], mode="markers", marker_opacity=0, showlegend=False,
                    hoverinfo="skip")
    return fig


# ============================================================
# TIÊU ĐỀ + NẠP DỮ LIỆU
# ============================================================
st.title("🔎 SHAP: Giải thích mô hình XGBoost")
st.markdown("---")

imp = read_csv("shap_importance_xgb.csv")
vs = read_csv("shap_vs_scorecard.csv")
cases_csv = read_csv("shap_local_cases.csv")

model_ok = XGB_PATH.exists() and TEST_PATH.exists()
if model_ok:
    X, y, values, base, prob = compute_shap()
    if imp is None:
        imp = (pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(0)})
               .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
        imp["share_pct"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
        imp["rank"] = imp.index + 1

if imp is None:
    st.error("Chưa có dữ liệu SHAP. Hãy chạy `python shap_analysis.py` trước.")
    st.stop()

# ============================================================
# A. KPI
# ============================================================
st.markdown("## **A. Tổng quan kết quả SHAP**")

rho = None
if vs is not None and len(vs) > 2:
    rho = vs[["shap_share_pct", "logit_share_pct"]].corr(method="spearman").iloc[0, 1]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("📊 Số biến XGBoost", f"{len(imp)}")
with c2:
    st.metric("🥇 Biến quan trọng nhất", imp.loc[0, "feature"])
with c3:
    st.metric("🎯 Đóng góp của biến #1", f"{imp.loc[0, 'share_pct']:.1f}%")
with c4:
    st.metric("🔗 Spearman SHAP vs Logit", f"{rho:.2f}" if rho is not None else "N/A")

st.write("")
st.markdown(f"""
<div style='font-size: 18px; line-height: 1.7; background-color: #D6E6FA; padding: 20px; border-radius: 10px;'>
<b>Cách đọc:</b> SHAP &gt; 0 nghĩa là biến <b>đẩy xác suất vỡ nợ (PD) lên</b> (màu đỏ), SHAP &lt; 0 là <b>kéo xuống</b> (màu xanh).
Đơn vị là log-odds của XGBoost. Với mỗi khách hàng: <i>base value + ΣSHAP = log-odds dự báo</i>.
Biến <b>{imp.loc[0, 'feature']}</b> chiếm {imp.loc[0, 'share_pct']:.1f}% tổng mức độ ảnh hưởng của mô hình.
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ============================================================
# B. SHAP TOÀN CỤC
# ============================================================
st.markdown("## **B. SHAP toàn cục (Global)**")

if not model_ok:
    st.warning("Thiếu `models/xgb_model.pkl` hoặc `data/test.csv` — hiển thị ảnh PNG thay cho biểu đồ động.")
    for f in ("shap_summary.png", "shap_bar.png", "shap_dependence_top1.png"):
        if (SHAP_DIR / f).exists():
            st.image(str(SHAP_DIR / f), use_container_width=True)
else:
    top_n = st.slider("Số dòng hiển thị (biến + dòng 'Sum of other features')", 5, min(25, len(imp)),
                      min(15, len(imp)))
    tab_bee, tab_bar, tab_dep = st.tabs(["🐝 Beeswarm", "📊 Mean |SHAP|", "📈 Dependence"])

    with tab_bee:
        show(beeswarm_fig(X, values, top_n))
        st.caption("Mỗi chấm = 1 khách hàng (mẫu ngẫu nhiên tối đa 2.000 khách). "
                   "Sang phải = tăng rủi ro vỡ nợ. Đỏ = giá trị biến cao, xanh = thấp.")

    with tab_bar:
        col_t, col_g = st.columns([1, 2])
        with col_t:
            st.markdown("<b style='font-size:18px;'>Bảng xếp hạng mean |SHAP|</b>", unsafe_allow_html=True)
            st.dataframe(imp[["rank", "feature", "mean_abs_shap", "share_pct"]]
                         .style.format({"mean_abs_shap": "{:.4f}", "share_pct": "{:.2f}%"}),
                         height=520, hide_index=True, use_container_width=True)
        with col_g:
            show(bar_fig(X, values, top_n))

    with tab_dep:
        feat = st.selectbox("Chọn biến", list(imp["feature"]), index=0)
        show(dependence_fig(X, values, feat))

st.markdown("<br><br>", unsafe_allow_html=True)

# ============================================================
# C. SHAP CỤC BỘ
# ============================================================
st.markdown("## **C. SHAP cục bộ (Local) — vì sao một khách hàng bị chấm như vậy?**")

if not model_ok:
    for n_ in ("high_risk", "low_risk", "borderline"):
        p = SHAP_DIR / f"shap_local_{n_}.png"
        if p.exists():
            st.image(str(p), use_container_width=True)
else:
    pos_i, neg_i = np.where(y == 1)[0], np.where(y == 0)[0]
    cases = {
        "high_risk": int(pos_i[np.argmax(prob[pos_i])]),
        "low_risk": int(neg_i[np.argmin(prob[neg_i])]),
        "borderline": int(np.argmin(np.abs(prob - THRESHOLD))),
    }
    names = {"high_risk": "🔴 Rủi ro cao", "low_risk": "🟢 Rủi ro thấp",
             "borderline": "🟡 Sát ngưỡng", "custom": "✏️ Khách hàng khác"}
    tabs = st.tabs(list(names.values()))

    def render_case(i, label):
        k1, k2, k3 = st.columns(3)
        k1.metric("PD dự báo", f"{prob[i]:.3f}")
        k2.metric("Nhãn thật", "Vỡ nợ" if y[i] == 1 else "Không vỡ nợ")
        k3.metric("Base value (log-odds)", f"{base:.3f}")
        show(waterfall_fig(X, values, base, i,
                           f"Khách hàng {label} — PD dự báo = {prob[i]:.3f}, nhãn thật = {int(y[i])}"))

    for tab, key in zip(tabs[:3], cases):
        with tab:
            render_case(cases[key], key)
    with tabs[3]:
        idx_c = int(st.number_input("Vị trí khách hàng trong mẫu (0 → n-1)", 0, len(X) - 1, 0, step=1))
        render_case(idx_c, f"#{idx_c}")
    st.caption("Đọc từ dưới lên: bắt đầu từ E[f(X)] (log-odds trung bình), mỗi mũi tên là đóng góp của một biến, "
               "kết thúc ở f(x). Đỏ = tăng rủi ro, xanh = giảm rủi ro.")

if cases_csv is not None:
    with st.expander("📋 Top 5 biến đóng góp của 3 khách hàng minh họa (shap_local_cases.csv)"):
        st.dataframe(cases_csv, hide_index=True, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ============================================================
# D. ĐỐI CHIẾU VỚI LOGISTIC SCORECARD
# ============================================================
st.markdown("## **D. Đối chiếu SHAP (XGBoost) với Logistic Scorecard**")

if vs is None:
    st.info("Chưa có `shap_vs_scorecard.csv`. Chạy `python shap_analysis.py` để tạo.")
else:
    top5_x = set(vs.nsmallest(5, "shap_rank")["feature"])
    top5_l = set(vs.nsmallest(5, "logit_rank")["feature"])
    dir_df = vs[vs["cung_chieu"].isin(["Có", "Không"])]
    n_same = int((dir_df["cung_chieu"] == "Có").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔗 Biến chung", f"{len(vs)}")
    m2.metric("📐 Spearman thứ hạng", f"{rho:.2f}" if rho is not None else "N/A")
    m3.metric("🏆 Trùng top-5", f"{len(top5_x & top5_l)}/5")
    m4.metric("↔️ Cùng chiều", f"{n_same}/{len(dir_df)}")
    st.write("")

    d = vs.head(15)
    fig = go.Figure()
    fig.add_bar(y=d["feature"], x=d["shap_share_pct"], orientation="h",
                name="SHAP (XGBoost)", marker_color=MPL_BLUE)
    fig.add_bar(y=d["feature"], x=d["logit_share_pct"], orientation="h",
                name="Logistic Scorecard", marker_color=MPL_ORANGE)
    fig.update_layout(**base_layout("Đối chiếu mức độ quan trọng: SHAP (XGBoost) vs Logistic Scorecard",
                                    max(500, 46 * len(d) + 140)),
                      barmode="group", bargap=0.2, bargroupgap=0,
                      legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom",
                                  bordercolor="#CCCCCC", borderwidth=1, bgcolor="white"))
    fig.update_xaxes(showline=True, mirror=True, linecolor="#333", ticks="outside", showgrid=False,
                     title="Tỷ trọng trong tổng |đóng góp| của các biến chung (%)")
    fig.update_yaxes(showline=True, mirror=True, linecolor="#333", ticks="outside",
                     autorange="reversed", automargin=True)
    show(fig)

    st.markdown("<b style='font-size:18px;'>Bảng đối chiếu chi tiết</b>", unsafe_allow_html=True)
    st.dataframe(vs[["feature", "beta_logit", "shap_share_pct", "logit_share_pct",
                     "shap_rank", "logit_rank", "cung_chieu"]]
                 .style.format({"beta_logit": "{:+.4f}", "shap_share_pct": "{:.2f}%",
                                "logit_share_pct": "{:.2f}%"}),
                 hide_index=True, use_container_width=True)

    st.info("💡 **Ghi chú:** Hai mô hình có thang log-odds khác nhau (XGBoost dùng scale_pos_weight) "
            "nên so sánh **tỷ trọng %**, thứ hạng và chiều tác động thay vì độ lớn tuyệt đối. "
            "Biến danh mục (SEX, EDUCATION, MARRIAGE) không có thứ tự thật nên so theo từng nhóm.")