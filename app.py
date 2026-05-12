import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle

def download_chart(fig, key):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(
        label="Download Chart",
        data=buf,
        file_name="chart.png",
        mime="image/png",
        key=key
    )

@st.cache_data
def load_data(file_bytes):          
    df = pd.read_csv(io.BytesIO(file_bytes), engine="pyarrow")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
        )
    return df

@st.cache_data
def make_num_plots(data_list, col_name, mean_val, median_val):
    clean = pd.Series(data_list)
    fig1, ax = plt.subplots(figsize=(5, 3))
    fig1.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.hist(clean, bins=30, color="#3b82f6", alpha=0.7, edgecolor="none")
    ax2 = ax.twinx()
    clean.plot.kde(ax=ax2, color="#f472b6", linewidth=1.5)
    ax2.set_ylabel("")
    ax2.tick_params(left=False, right=False, labelleft=False, labelright=False)
    ax2.set_facecolor("#0d1117")
    for s in ax2.spines.values(): s.set_visible(False)
    ax.axvline(mean_val,   color="#f59e0b", linewidth=1.2, linestyle="--", label="Mean")
    ax.axvline(median_val, color="#22c55e", linewidth=1.2, linestyle="--", label="Median")
    ax.set_xlabel(col_name, color="#64748b", fontsize=8)
    ax.set_ylabel("Count",  color="#64748b", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)
    for s in ax.spines.values(): s.set_visible(False)
    ax.legend(fontsize=7, labelcolor="#94a3b8", facecolor="#0d1117", edgecolor="#1e293b")
    plt.tight_layout()
    return fig1

@st.cache_data
def make_box_plot(data_list, col_name):
    fig2, ax = plt.subplots(figsize=(5, 3))
    fig2.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.boxplot(data_list, vert=False, patch_artist=True, widths=0.5,
               boxprops    =dict(facecolor="#1e3a5f", color="#3b82f6"),
               medianprops =dict(color="#f472b6", linewidth=2),
               whiskerprops=dict(color="#3b82f6"),
               capprops    =dict(color="#3b82f6"),
               flierprops  =dict(marker="o", color="#f59e0b", markersize=3, alpha=0.5))
    ax.set_xlabel(col_name, color="#64748b", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)
    ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.xaxis.grid(True, color="#151c28", linewidth=0.5)
    plt.tight_layout()
    return fig2


# ================= PAGE CONFIG =================
st.set_page_config(page_title="ML predicter", layout="wide", page_icon="🚀")

# ================= STYLING =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: #080b10;
    color: #cbd5e1;
}

/* ── Hide clutter ── */
#MainMenu, footer, .stDeployButton { display: none; }

/* ── Hero header ── */
.hero-wrap {
    padding: 2.5rem 0 1.8rem 0;
    border-bottom: 1px solid #151c28;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    background: linear-gradient(110deg, #e2e8f0 30%, #64748b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}

.hero-accent {
    color: #3b82f6;
    -webkit-text-fill-color: #3b82f6;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #334155;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 1.5px dashed #1e293b !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #334155 !important;
}

/* ── Info / Error banners ── */
.banner {
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.83rem;
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
}

.banner-ok {
    background: #051810;
    border: 1px solid #14532d;
    border-left: 3px solid #22c55e;
    color: #86efac;
}

.banner-warn {
    background: #16100a;
    border: 1px solid #78350f;
    border-left: 3px solid #f59e0b;
    color: #fcd34d;
}

.banner-err {
    background: #150a0a;
    border: 1px solid #7f1d1d;
    border-left: 3px solid #ef4444;
    color: #fca5a5;
}

/* ── Stat cards ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1.4rem 0;
}

.stat-card {
    background: #0d1117;
    border: 1px solid #151c28;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
}

.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #334155;
    margin-bottom: 0.5rem;
}

.stat-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}

.stat-value.accent { color: #3b82f6; }
.stat-value.warn   { color: #f59e0b; }
.stat-value.ok     { color: #22c55e; }

/* ── Section label ── */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #64748b;
    border-bottom: 1px solid #0f1520;
    padding-bottom: 0.45rem;
    margin: 1.8rem 0 1rem 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border: 1px solid #151c28;
    border-radius: 10px;
    padding: 4px;
    gap: 3px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    color: #334155;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    padding: 0.4rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: #151c28 !important;
    color: #3b82f6 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #151c28 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

[data-testid="stDataFrame"] table {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Subheaders ── */
h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    letter-spacing: -0.01em;
    margin-top: 1.5rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080b10; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ================= HERO HEADER =================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title"><span class="hero-accent">ML</span>Predictor</div>
    <div class="hero-sub">Upload CSV &nbsp;→&nbsp; Select Model &nbsp;→&nbsp; Train &nbsp;→&nbsp; Evaluate</div>
</div>
""", unsafe_allow_html=True)

# ================= FILE UPLOAD =================
st.markdown('<div class="sec-label">Dataset</div>', unsafe_allow_html=True)

file = st.file_uploader(
    "Drop your CSV file here — max 100 MB",
    type=["csv"],
    label_visibility="collapsed"
)

if file is not None:
    if file.size > 100 * 1024 * 1024:
        st.markdown("""
        <div class="banner banner-err">
            ✕ &nbsp;<span>File too large — please upload a file under <strong>100 MB</strong>.</span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if "file_name" not in st.session_state or st.session_state.file_name != file.name:
        st.session_state.file_name = file.name
        st.session_state.original_df = load_data(file.read())
        st.session_state.df = st.session_state.original_df.copy()
        st.session_state["tab3_history"] = []
        st.session_state["enc_history"]  = []

    if "df" not in st.session_state:
        st.markdown('<div class="banner banner-warn">⚠ Data not initialized — please re-upload.</div>', unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df

    size_mb = file.size / (1024 * 1024)
    st.markdown(f"""
    <div class="banner banner-ok">
        ✓ &nbsp;<span><strong>{file.name}</strong> loaded &nbsp;·&nbsp; {size_mb:.2f} MB</span>
    </div>
    """, unsafe_allow_html=True)

    if len(df) > 200000:
        st.markdown('<div class="banner banner-warn">⚠ Large dataset — some operations may be slow.</div>', unsafe_allow_html=True)

    missing_total = int(df.isnull().sum().sum())
    dup_total     = int(df.duplicated().sum())

    miss_cls = "warn" if missing_total > 0 else "ok"
    dup_cls  = "warn" if dup_total > 0 else "ok"

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-label">Rows</div>
            <div class="stat-value accent">{df.shape[0]:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Columns</div>
            <div class="stat-value accent">{df.shape[1]}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Missing Values</div>
            <div class="stat-value {miss_cls}">{missing_total:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Duplicate Rows</div>
            <div class="stat-value {dup_cls}">{dup_total:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ================= TABS =================
    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5, eda_tab6 = st.tabs([
        "Overview",
        "Column Analyser",
        "Fill Missing Values",
        "Encoding",
        "Feature Importance",
        "Model Training"
    ])


    # ================= TAB 1 — OVERVIEW =================
    with eda_tab1:
 
        # ── helper: reusable section label (consistent with .sec-label CSS) ──
        def sec(title):
            st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)
 
        def divider():
            st.markdown("<hr style='border:none;border-top:1px solid #0f1520;margin:1.8rem 0;'>",
                        unsafe_allow_html=True)
 
        # ── pre-compute column groups once ──
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        object_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
        bool_cols    = df.select_dtypes(include=["bool"]).columns.tolist()
 
        # ── pre-compute outlier info ONCE (used for both Health Score + table) ──
        @st.cache_data
        def compute_outliers(df, num_cols):
            _df = df.copy()
            rows = []
            for col in num_cols:
                if col not in _df.columns:
                    continue  # 🔥 IMPORTANT FIX

                s = _df[col].dropna()
                if s.empty:
                    continue

                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                count = int(((s < lower) | (s > upper)).sum())

                rows.append({
                    "Column": col,
                    "Lower Bound": round(float(lower), 3),
                    "Upper Bound": round(float(upper), 3),
                    "Outlier Count": count,
                    "Outlier %": round((count / len(s)) * 100, 2) if len(s) else 0,
                })
            return rows
 
        if numeric_cols:
            outlier_rows = compute_outliers(df, numeric_cols)
        else:
            outlier_rows = []
 
        # ── 1. Dataset Preview ──
        sec("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        divider()
 
        # ── 2. Column Info ──
        sec("Column Info")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p style="font-size:0.78rem;color:#64748b;margin-bottom:0.3rem;">Column Names</p>',
                        unsafe_allow_html=True)
            st.write(df.columns.tolist())
        with c2:
            st.markdown('<p style="font-size:0.78rem;color:#64748b;margin-bottom:0.3rem;">Data Types</p>',
                        unsafe_allow_html=True)
            st.write(df.dtypes)
        divider()
 
        # ── 3. Statistical Summary ──
        sec("Statistical Summary")
        if numeric_cols:
            st.markdown('<p style="font-size:0.75rem;color:#3b82f6;margin-bottom:0.3rem;">▸ Numerical Columns</p>',
                        unsafe_allow_html=True)
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        if object_cols:
            st.markdown('<p style="font-size:0.75rem;color:#22c55e;margin-bottom:0.3rem;">▸ Categorical Columns</p>',
                        unsafe_allow_html=True)
            st.dataframe(df[object_cols].describe(), use_container_width=True)
        if bool_cols:
            st.markdown('<p style="font-size:0.75rem;color:#f59e0b;margin-bottom:0.3rem;">▸ Boolean Columns</p>',
                        unsafe_allow_html=True)
            st.dataframe(df[bool_cols].describe(), use_container_width=True)
        divider()
 
        # ── 4. Correlation Table ──
        if len(numeric_cols) > 1:
            sec("Correlation Matrix")
            corr = df[numeric_cols].corr().round(3)
 
            # Style the correlation dataframe with a heatmap-like background
            def color_corr(val):
                if pd.isna(val):
                    return "background-color: #0d1117; color: #334155;"
                abs_val = abs(val)
                if abs_val > 0.8:
                    bg = "#0c2340" if val > 0 else "#2d0a0a"
                    fg = "#3b82f6" if val > 0 else "#ef4444"
                elif abs_val > 0.5:
                    bg = "#0a1a2e" if val > 0 else "#1f0808"
                    fg = "#60a5fa" if val > 0 else "#f87171"
                elif abs_val > 0.2:
                    bg = "#0d1117"
                    fg = "#94a3b8"
                else:
                    bg = "#0d1117"
                    fg = "#334155"
                return f"background-color:{bg}; color:{fg}; font-weight:500;"
 
            styled_corr = corr.style.applymap(color_corr)
            st.dataframe(styled_corr, use_container_width=True)
            divider()
 
        # ── 5. Missing Values ──
        sec("Missing Values")
        missing         = df.isnull().sum()
        missing_pct     = (missing / len(df)) * 100
        missing_df      = pd.DataFrame({
            "Missing Values": missing,
            "Percentage (%)": missing_pct.round(2)
        })
        missing_filtered = missing_df[missing_df["Missing Values"] > 0]
        if missing_filtered.empty:
            st.markdown("""
            <div class="banner banner-ok">
                ✓ &nbsp;<span>No missing values found in this dataset.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.dataframe(missing_filtered, use_container_width=True)
        divider()
 
        # ── 6. Data Health Score ──
        sec("Data Health Score")
 
        total_cells  = df.shape[0] * df.shape[1]
        missing_pct_val = (df.isnull().sum().sum() / total_cells) * 100
        dup_pct_val     = (df.duplicated().sum() / len(df)) * 100
        outlier_total   = sum(r["Outlier Count"] for r in outlier_rows)
        outlier_pct_val = (outlier_total / len(df)) * 100 if len(df) > 0 else 0
 
        missing_score = max(0, 100 - missing_pct_val * 2)
        dup_score     = max(0, 100 - dup_pct_val * 3)
        outlier_score = max(0, 100 - outlier_pct_val * 1.5)
        health_score  = round(missing_score * 0.4 + dup_score * 0.3 + outlier_score * 0.3, 1)
 
        if health_score >= 80:
            score_color, score_label = "#22c55e", "Excellent"
        elif health_score >= 60:
            score_color, score_label = "#f59e0b", "Fair"
        else:
            score_color, score_label = "#ef4444", "Poor"
 
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #151c28;border-radius:12px;
                    padding:1.4rem 1.6rem;margin-bottom:1rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#334155;
                        text-transform:uppercase;letter-spacing:0.16em;margin-bottom:0.5rem;">
                Overall Health Score
            </div>
            <div style="font-size:3rem;font-weight:800;color:{score_color};line-height:1;">
                {health_score}
                <span style="font-size:1rem;color:#334155;">/100</span>
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                        color:{score_color};margin-top:0.3rem;">{score_label}</div>
            <div style="margin-top:0.9rem;background:#151c28;border-radius:6px;
                        height:5px;overflow:hidden;">
                <div style="width:{health_score}%;height:100%;background:{score_color};
                            border-radius:6px;transition:width 0.6s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Missing Score",   f"{round(missing_score,1)}/100",
                   f"{missing_pct_val:.1f}% missing")
        sc2.metric("Duplicate Score", f"{round(dup_score,1)}/100",
                   f"{dup_pct_val:.1f}% duplicates")
        sc3.metric("Outlier Score",   f"{round(outlier_score,1)}/100",
                   f"{outlier_pct_val:.1f}% outliers")
 
        divider()
 
        # ── 7. Outlier Detection Table ──
        sec("Outlier Detection (IQR Method)")
        if not numeric_cols:
            st.markdown("""
            <div class="banner banner-warn">
                ⚠ &nbsp;<span>No numeric columns found for outlier detection.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            outlier_df = (
                pd.DataFrame(outlier_rows)
                .sort_values("Outlier Count", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(outlier_df, use_container_width=True)



    # ================= TAB 2 — COLUMN ANALYSER =================
    with eda_tab2:
 
        def sec(title):
            st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)
 
        def divider():
            st.markdown(
                "<hr style='border:none;border-top:1px solid #0f1520;margin:1.4rem 0;'>",
                unsafe_allow_html=True,
            )
 
        # ── Column selector ──
        column = st.selectbox("Select Column", df.columns, key="ml_col_analyzer")
        divider()
 
        # ── Top metrics (all column types) ──
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Data Type",      str(df[column].dtype))
        colB.metric("Missing Values", int(df[column].isnull().sum()))
        colC.metric("Missing %",      f"{(df[column].isnull().sum() / len(df)) * 100:.2f}%")
        colD.metric("Unique Values",  int(df[column].nunique()))
        divider()
 
        # ══════════════════════════════════════════
        #  NUMERIC BRANCH
        # ══════════════════════════════════════════
        if pd.api.types.is_numeric_dtype(df[column]):
 
            series     = df[column].dropna()
            mean_val   = series.mean()
            median_val = series.median()
            std_val    = series.std()
            min_val    = series.min()
            max_val    = series.max()
            skew_val   = series.skew()
            kurt_val   = series.kurt()
 
            # ── Core stats ──
            sec("Descriptive Statistics")
            colE, colF, colG = st.columns(3)
            colE.metric("Mean",    round(mean_val, 4))
            colF.metric("Median",  round(median_val, 4))
            colG.metric("Std Dev", round(std_val, 4))
 
            colH, colI, colJ = st.columns(3)
            colH.metric("Min",      round(min_val, 4))
            colI.metric("Max",      round(max_val, 4))
            colJ.metric("Skewness", round(skew_val, 4))
 
            # ── Percentile table ──
            percentiles = series.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).round(4)
            pct_df = pd.DataFrame({
                "Percentile": ["P5", "P25 (Q1)", "P50 (Median)", "P75 (Q3)", "P95"],
                "Value":      percentiles.values
            })
            st.dataframe(pct_df, use_container_width=True, hide_index=True)
            divider()
 
            # ── Distribution Interpretation ──
            sec("Distribution Interpretation")
 
            if skew_val > 1:
                skew_label, skew_color = "Highly Right Skewed",         "#ef4444"
            elif skew_val > 0.5:
                skew_label, skew_color = "Moderately Right Skewed",     "#f59e0b"
            elif skew_val < -1:
                skew_label, skew_color = "Highly Left Skewed",          "#ef4444"
            elif skew_val < -0.5:
                skew_label, skew_color = "Moderately Left Skewed",      "#f59e0b"
            else:
                skew_label, skew_color = "Approximately Symmetric",     "#22c55e"
 
            if kurt_val > 3:
                kurt_label = "Leptokurtic — heavy tails, sharp peak (outliers likely)"
                kurt_color = "#ef4444"
            elif kurt_val < -1:
                kurt_label = "Platykurtic — light tails, flat distribution"
                kurt_color = "#f59e0b"
            else:
                kurt_label = "Mesokurtic — normal-like tails"
                kurt_color = "#22c55e"
 
            interp_l, interp_r = st.columns(2)
            interp_l.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1e293b;border-left:3px solid {skew_color};
                        border-radius:10px;padding:0.85rem 1rem;">
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#64748b;
                            text-transform:uppercase;letter-spacing:0.2em;margin-bottom:0.4rem;">
                    Skewness · {round(skew_val, 3)}
                </div>
                <div style="font-size:0.78rem;color:{skew_color};">{skew_label}</div>
            </div>
            """, unsafe_allow_html=True)
 
            interp_r.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1e293b;border-left:3px solid {kurt_color};
                        border-radius:10px;padding:0.85rem 1rem;">
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#64748b;
                            text-transform:uppercase;letter-spacing:0.2em;margin-bottom:0.4rem;">
                    Kurtosis · {round(kurt_val, 3)}
                </div>
                <div style="font-size:0.78rem;color:{kurt_color};">{kurt_label}</div>
            </div>
            """, unsafe_allow_html=True)
            divider()
 
            # ── Charts ──
            sec("Visualisations")
            plot_series = (
                series
                if len(series) <= 5000
                else series.sample(5000, random_state=42)
            )

            plot_series = plot_series.dropna()

            if plot_series.nunique() < 2:
                st.markdown("""
                <div class="banner banner-warn">
                    ⚠ &nbsp;<span>Not enough variation for KDE/distribution plotting.</span>
                </div>
                """, unsafe_allow_html=True)

            else:
                chart_l, chart_r = st.columns(2)
                with chart_l:
                    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.63rem;'
                                'text-transform:uppercase;letter-spacing:0.2em;color:#64748b;'
                                'margin-bottom:0.5rem;">Distribution + KDE</p>',
                                unsafe_allow_html=True)
                    fig1 = make_num_plots(plot_series.tolist(), column, mean_val, median_val)
                    st.pyplot(fig1)
                    download_chart(fig1, key="ml_hist_download")
                    plt.close(fig1)
    
                with chart_r:
                    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.63rem;'
                                'text-transform:uppercase;letter-spacing:0.2em;color:#64748b;'
                                'margin-bottom:0.5rem;">Box Plot</p>',
                                unsafe_allow_html=True)
                    fig2 = make_box_plot(plot_series.tolist(), column)
                    st.pyplot(fig2)
                    download_chart(fig2, key="ml_box_download")
                    plt.close(fig2)
 
            divider()
 
            # ── Outlier Detection (IQR) ──
            sec("Outlier Detection (IQR Method)")
 
            q1, q3   = series.quantile(0.25), series.quantile(0.75)
            iqr      = q3 - q1
            lower_b  = q1 - 1.5 * iqr          # renamed → no clash with Tab 1
            upper_b  = q3 + 1.5 * iqr
            out_mask = (df[column] < lower_b) | (df[column] > upper_b)
            out_df   = df[out_mask]             # renamed from outlier_rows → out_df
 
            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Lower Bound",  round(lower_b, 3))
            oc2.metric("Upper Bound",  round(upper_b, 3))
            oc3.metric("Outlier Rows", len(out_df))
 
            if out_df.empty:
                st.markdown("""
                <div class="banner banner-ok">
                    ✓ &nbsp;<span>No outliers found in this column.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                preview_n = min(50, len(out_df))
                with st.expander(f"Preview {preview_n} Outlier Rows"):
                    st.dataframe(out_df.head(50), use_container_width=True)
                    if len(out_df) > 50:
                        st.caption(f"Showing 50 of {len(out_df)} outlier rows.")
 
        # ══════════════════════════════════════════
        #  CATEGORICAL BRANCH
        # ══════════════════════════════════════════
        else:
            vc = df[column].value_counts()
 
            mode_val      = df[column].mode()
            most_frequent = mode_val[0] if not mode_val.empty else "N/A"
            top_count     = int(vc.iloc[0]) if not vc.empty else 0
 
            colE, colF = st.columns(2)
            colE.metric("Most Frequent Value", str(most_frequent))
            colF.metric("Top Value Count",     top_count)
            divider()
 
            # ── Value Distribution table ──
            sec("Value Distribution")
            vc_table = pd.DataFrame({
                "Value"      : vc.index,
                "Count"      : vc.values,
                "Percentage" : (vc.values / len(df) * 100).round(2)
            })
            vc_table["Percentage"] = vc_table["Percentage"].astype(str) + " %"
            st.dataframe(vc_table, use_container_width=True, hide_index=True)
            divider()
 
            # ── Charts ──
            MAX_BARS  = 20
            vc_plot   = vc.head(MAX_BARS)
            truncated = len(vc) > MAX_BARS
 
            sec(f"Top {MAX_BARS} Values" if truncated else "Value Counts")
            chart_l, chart_r = st.columns(2)
 
            with chart_l:
                st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.63rem;'
                            'text-transform:uppercase;letter-spacing:0.2em;color:#64748b;'
                            'margin-bottom:0.5rem;">Bar Chart</p>',
                            unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, max(3, len(vc_plot) * 0.35)))
                fig.patch.set_facecolor("#080b10")
                ax.set_facecolor("#080b10")
                bar_colors = ["#3b82f6" if i == 0 else "#1e293b"
                              for i in range(len(vc_plot))]
                ax.barh(vc_plot.index[::-1], vc_plot.values[::-1],
                        color=bar_colors[::-1], edgecolor="none")
                ax.set_xlabel("Count", color="#64748b", fontsize=8)
                ax.tick_params(colors="#64748b", labelsize=7)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.xaxis.grid(True, color="#151c28", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                download_chart(fig, key="ml_bar_download")
                plt.close(fig)
 
            with chart_r:
                if len(vc) <= 10:
                    pie_label = "Distribution"
                    pie_data  = vc
                else:
                    pie_label = "Top 10 Share"
                    top10     = vc.head(10)
                    other_sum = vc.iloc[10:].sum()
                    pie_data  = pd.concat([top10, pd.Series({"Other": other_sum})])
 
                st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:0.63rem;'
                            f'text-transform:uppercase;letter-spacing:0.2em;color:#64748b;'
                            f'margin-bottom:0.5rem;">{pie_label}</p>',
                            unsafe_allow_html=True)
 
                pie_colors = ["#3b82f6","#60a5fa","#22c55e","#f59e0b",
                              "#f472b6","#a78bfa","#38bdf8","#4ade80",
                              "#fb923c","#f87171","#94a3b8"]
 
                fig, ax = plt.subplots(figsize=(4, 4))
                fig.patch.set_facecolor("#080b10")
                ax.set_facecolor("#080b10")
                wedges, texts, autotexts = ax.pie(
                    pie_data.values,
                    labels=pie_data.index,
                    colors=pie_colors[:len(pie_data)],
                    autopct="%1.1f%%",
                    startangle=140,
                    textprops={"color": "#64748b", "fontsize": 7},
                    wedgeprops={"edgecolor": "#080b10", "linewidth": 1.5},
                )
                for at in autotexts:
                    at.set_color("#e2e8f0")
                    at.set_fontsize(7)
                plt.tight_layout()
                st.pyplot(fig)
                download_chart(fig, key="ml_pie_download")
                plt.close(fig)



    # ================= TAB 3 — FILL MISSING VALUES =================
    with eda_tab3:
 
        def sec(title):
            st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)
 
        def divider():
            st.markdown(
                "<hr style='border:none;border-top:1px solid #0f1520;margin:1.6rem 0;'>",
                unsafe_allow_html=True,
            )
 
        def banner(msg, kind="ok"):
            icons = {"ok": "✓", "warn": "⚠", "err": "✕"}
            st.markdown(
                f'<div class="banner banner-{kind}">'
                f'{icons[kind]} &nbsp;<span>{msg}</span></div>',
                unsafe_allow_html=True,
            )
 
        # always pull latest from session state
        df = st.session_state.df
 
        # ── 1. Missing Values Table ──
        sec("Missing Values Table")
 
        missing     = df.isnull().sum()
        missing_col = missing[missing > 0]
 
        if missing_col.empty:
            banner("No missing values in the current dataset.", "ok")
        else:
            missing_pct = (missing_col / len(df) * 100).round(2)
            missing_tbl = pd.DataFrame({
                "Column"        : missing_col.index,
                "Missing Count" : missing_col.values,
                "Missing %"     : missing_pct.values,
            }).sort_values("Missing Count", ascending=False).reset_index(drop=True)
 
            st.caption(f"{len(missing_tbl)} column(s) with missing values")
            st.dataframe(missing_tbl, use_container_width=True, hide_index=True)
            divider()
 
            # ── 2. Fill Missing Values ──
            sec("Fill Missing Values")
 
            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox(
                    "Select Column",
                    missing_tbl["Column"].tolist(),
                    key="ml_col_select",
                )
 
            is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])
            method_options = (
                ["Mean", "Median", "Mode", "Custom Value"]
                if is_numeric
                else ["Mode", "Custom Value"]
            )
 
            with col2:
                method = st.selectbox(
                    "Fill Method",
                    method_options,
                    key="ml_method_select",
                )
 
            # live preview of fill value
            preview_val = None
            if method == "Mean":
                preview_val = round(df[selected_col].mean(), 4)
            elif method == "Median":
                preview_val = round(df[selected_col].median(), 4)
            elif method == "Mode":
                m = df[selected_col].mode()
                preview_val = m[0] if not m.empty else None
 
            if method != "Custom Value" and preview_val is not None:
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #151c28;border-left:3px solid #3b82f6;'
                    f'border-radius:8px;padding:0.6rem 1rem;font-family:\'DM Mono\',monospace;'
                    f'font-size:0.75rem;color:#60a5fa;margin:0.5rem 0 0.8rem 0;">'
                    f'Fill preview → <strong>{preview_val}</strong></div>',
                    unsafe_allow_html=True,
                )
 
            custom_value = None
            if method == "Custom Value":
                custom_value = st.text_input(
                    "Enter custom fill value",
                    placeholder='e.g. 0 or "Unknown"',
                    key="ml_custom_fill_val",
                )
 
            if st.button("Fill Missing", key="ml_fill_btn"):
                value = None
                if method == "Mean":
                    value = df[selected_col].mean()
                elif method == "Median":
                    value = df[selected_col].median()
                elif method == "Mode":
                    m = df[selected_col].mode()
                    value = m[0] if not m.empty else None
                else:
                    if not custom_value:
                        banner("Please enter a custom value before filling.", "warn")
                    else:
                        if is_numeric:
                            try:
                                value = float(custom_value)
                            except ValueError:
                                banner("Numeric column — please enter a valid number.", "err")
                        else:
                            value = custom_value
 
                if value is not None:
                    st.session_state.setdefault("tab3_history", []).append(
                        st.session_state.df.copy()
                    )
                    st.session_state.df[selected_col] = (
                        st.session_state.df[selected_col].fillna(value)
                    )
                    st.session_state["last_action"] = (
                        f"<strong>{selected_col}</strong> filled using "
                        f"<strong>{method}</strong> → <code>{round(value, 4) if isinstance(value, float) else value}</code>"
                    )
                    st.rerun()
 
        # ── last action toast (shown after rerun) ──
        if "last_action" in st.session_state:
            st.markdown(
                f'<div class="banner banner-ok">✓ &nbsp;{st.session_state.pop("last_action")}</div>',
                unsafe_allow_html=True,
            )
 
        divider()
 
        # ── 3. Rename Column ──
        sec("Rename a Column")
        df = st.session_state.df
 
        rn1, rn2, rn3 = st.columns([2, 2, 1])
        with rn1:
            rename_col = st.selectbox(
                "Select Column",
                df.columns.tolist(),
                key="ml_rename_col",
            )
        with rn2:
            new_name = st.text_input(
                "New Name",
                placeholder="Enter new column name",
                key="ml_rename_new",
            )
        with rn3:
            st.markdown("<br>", unsafe_allow_html=True)
            rename_btn = st.button("Rename", key="ml_rename_btn")
 
        if rename_btn:
            stripped = new_name.strip()
            if not stripped:
                banner("New column name cannot be empty.", "err")
            elif stripped in df.columns and stripped != rename_col:
                banner(f"Column <strong>{stripped}</strong> already exists.", "err")
            else:
                st.session_state.setdefault("tab3_history", []).append(
                    st.session_state.df.copy()
                )
                st.session_state.df = st.session_state.df.rename(
                    columns={rename_col: stripped}
                )
                st.session_state["last_action"] = (
                    f"<strong>{rename_col}</strong> renamed to <strong>{stripped}</strong>"
                )
                st.rerun()
 
        divider()
 
        # ── 4. Delete Column ──
        sec("Delete a Column")
        df = st.session_state.df
 
        dc1, dc2 = st.columns([3, 1])
        with dc1:
            del_col = st.selectbox(
                "Select Column to Delete",
                df.columns.tolist(),
                key="ml_fill_del_col",
            )
        with dc2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Delete", key="ml_fill_del_btn"):
                st.session_state.setdefault("tab3_history", []).append(
                    st.session_state.df.copy()
                )
                st.session_state.df = st.session_state.df.drop(columns=[del_col])
                st.session_state["last_action"] = (
                    f"Column <strong>{del_col}</strong> deleted"
                )
                st.rerun()
 
        divider()
 
        # ── 5. Undo / Reset ──
        sec("Undo / Reset")
        history = st.session_state.get("tab3_history", [])
 
        undo_col, reset_col = st.columns(2)
        with undo_col:
            if st.button(
                f"↩  Undo  ({len(history)} step{'s' if len(history) != 1 else ''} available)",
                key="ml_undo_btn",
                disabled=len(history) == 0,
            ):
                st.session_state.df = st.session_state["tab3_history"].pop()
                st.session_state["last_action"] = "Last action undone"
                st.rerun()
 
        with reset_col:
            if st.button("↺  Reset to Original", key="ml_fill_reset_btn"):
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state["tab3_history"] = []   # sirf Tab 3 ki history clear
                st.session_state["last_action"]  = "Dataset reset to original"
                st.rerun()
 
        divider()
 
        # ── 6. Download Cleaned Dataset ──
        sec("Download Cleaned Dataset")
 
        rows, cols = st.session_state.df.shape
        remaining_missing = int(st.session_state.df.isnull().sum().sum())
 
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #151c28;border-radius:10px;'
            f'padding:0.85rem 1.2rem;margin-bottom:0.9rem;font-family:\'DM Mono\',monospace;'
            f'font-size:0.72rem;color:#64748b;">'
            f'<span style="color:#e2e8f0;">{rows:,}</span> rows &nbsp;·&nbsp; '
            f'<span style="color:#e2e8f0;">{cols}</span> columns &nbsp;·&nbsp; '
            f'<span style="color:{"#ef4444" if remaining_missing else "#22c55e"};">'
            f'{remaining_missing:,} missing value{"s" if remaining_missing != 1 else ""} remaining'
            f'</span></div>',
            unsafe_allow_html=True,
        )
 
        csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download Cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_data.csv",
            mime="text/csv",
            key="ml_download_csv",
        )



    # ================= TAB 4 — ENCODING =================
    with eda_tab4:
 
        def sec(title):
            st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)
 
        def divider():
            st.markdown(
                "<hr style='border:none;border-top:1px solid #0f1520;margin:1.6rem 0;'>",
                unsafe_allow_html=True,
            )
 
        def banner(msg, kind="ok"):
            icons = {"ok": "✓", "warn": "⚠", "err": "✕"}
            st.markdown(
                f'<div class="banner banner-{kind}">'
                f'{icons[kind]} &nbsp;<span>{msg}</span></div>',
                unsafe_allow_html=True,
            )
 
        # ── always pull latest ──
        df       = st.session_state.df
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
 
        # ── enc_history is SEPARATE from df_history (Tab 3) ──
        if "enc_history" not in st.session_state:
            st.session_state["enc_history"] = []
 
        if not cat_cols:
            banner("No categorical columns found — all columns are already encoded.", "ok")
        else:
            # ── 1. Categorical Columns Info ──
            sec("Categorical Columns")
 
            cat_info = pd.DataFrame({
                "Column"        : cat_cols,
                "Unique Values" : [df[col].nunique() for col in cat_cols],
                "Missing Values": [int(df[col].isnull().sum()) for col in cat_cols],
            }).sort_values("Unique Values", ascending=False).reset_index(drop=True)
 
            st.dataframe(cat_info, use_container_width=True, hide_index=True)
            divider()
 
            # ── 2. Apply Encoding ──
            sec("Apply Encoding")
 
            col1, col2 = st.columns(2)
            with col1:
                selected_col = st.selectbox("Select Column", cat_cols, key="enc_col")
            with col2:
                method = st.selectbox(
                    "Encoding Method",
                    ["Label Encoding", "One Hot Encoding", "Manual (Ordinal)"],
                    key="enc_method",
                )
 
            unique_count = df[selected_col].nunique()
            has_missing  = bool(df[selected_col].isnull().sum() > 0)
            ohe_blocked  = method == "One Hot Encoding" and unique_count > 25
 
            # info card
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #151c28;border-radius:8px;'
                f'padding:0.65rem 1rem;font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                f'color:#64748b;margin:0.4rem 0 0.7rem 0;">'
                f'<span style="color:#e2e8f0;">{unique_count}</span> unique values in '
                f'<span style="color:#3b82f6;">{selected_col}</span></div>',
                unsafe_allow_html=True,
            )
 
            if has_missing:
                banner(
                    f"<strong>{selected_col}</strong> has missing values — "
                    f"fill them in the <strong>Fill Missing Values</strong> tab first.",
                    "err",
                )
            if ohe_blocked:
                banner(
                    f"<strong>{selected_col}</strong> has {unique_count} unique values "
                    f"(max 25 for OHE). Use Label Encoding instead.",
                    "err",
                )
 
            # ── Manual ordinal input ──
            order = ""
            if method == "Manual (Ordinal)":
                unique_vals = df[selected_col].dropna().unique().tolist()
                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #151c28;border-radius:8px;'
                    f'padding:0.6rem 1rem;font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                    f'color:#64748b;margin-bottom:0.5rem;">'
                    f'Available values: <span style="color:#f59e0b;">'
                    f'{", ".join(str(v) for v in unique_vals)}</span></div>',
                    unsafe_allow_html=True,
                )
                order = st.text_input(
                    "Enter order — lowest → highest, comma separated",
                    placeholder="e.g. poor, fair, good, excellent",
                    key="ordinal_input",
                )
 
            # ── Apply button ──
            is_disabled = has_missing or ohe_blocked
            if st.button("Apply Encoding", key="enc_btn", disabled=is_disabled):
                working_df = st.session_state.df.copy()
                series = working_df[selected_col]
                encoded_ok = False
 
                if method == "Label Encoding":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    working_df[selected_col] = le.fit_transform(series.astype(str))

                    msg = (
                        f"Label Encoding applied on "
                        f"<strong>{selected_col}</strong>"
                    )

                    encoded_ok = True
 
                elif method == "One Hot Encoding":
                    dummies = pd.get_dummies(
                            series,
                            prefix=selected_col,
                            dtype=np.int8
                    )
                    working_df = pd.concat(
                        [working_df.drop(columns=[selected_col]), dummies], axis=1
                    )
                    msg        = (f"One Hot Encoding applied on "
                                  f"<strong>{selected_col}</strong> "
                                  f"→ {dummies.shape[1]} new columns created")
                    encoded_ok = True
 
                else:  # Manual (Ordinal)
                    if not order.strip():
                        banner("Please enter the ordinal order before applying.", "warn")
                    else:
                        values = [x.strip() for x in order.split(",")]
                        actual_values = (
                            working_df[selected_col]
                            .dropna()
                            .unique()
                            .tolist()
                        )

                        invalid = set(values) - set(actual_values)
                        missing_vals = set(actual_values) - set(values)

                        if invalid:
                            banner(
                                f"Invalid value(s): <strong>{', '.join(invalid)}</strong> "
                                f"— check spelling and try again.",
                                "err",
                            )
                        elif missing_vals:
                            banner(
                                f"Missing category(s): <strong>{', '.join(missing_vals)}</strong>",
                                "err",
                            )
                        else:
                            mapping = {val: i for i, val in enumerate(values)}

                            working_df[selected_col] = (
                                working_df[selected_col]
                                .map(mapping)
                            )
                            msg = (
                                f"Manual Ordinal Encoding applied on "
                                f"<strong>{selected_col}</strong>"
                            )
                            encoded_ok = True
 
                if encoded_ok:
                    st.session_state["enc_history"].append(st.session_state.df.copy())
                    st.session_state.df          = working_df
                    st.session_state["enc_msg"]  = msg
                    st.rerun()
 
        # ── success / info banner (after rerun) ──
        if "enc_msg" in st.session_state:
            st.markdown(
                f'<div class="banner banner-ok">✓ &nbsp;'
                f'{st.session_state.pop("enc_msg")}</div>',
                unsafe_allow_html=True,
            )
 
        divider()
 
        # ── 3. Undo / Reset  (enc_history only — does NOT touch Tab 3 history) ──
        sec("Undo / Reset")
 
        enc_history = st.session_state.get("enc_history", [])
        undo_c, reset_c = st.columns(2)
 
        with undo_c:
            if st.button(
                f"↩  Undo  ({len(enc_history)} step{'s' if len(enc_history) != 1 else ''} available)",
                key="enc_undo_btn",
                disabled=len(enc_history) == 0,
            ):
                st.session_state.df = st.session_state["enc_history"].pop()
                st.session_state["enc_msg"] = "Last encoding undone"
                st.rerun()
 
        with reset_c:
            if st.button("↺  Reset Encoding", key="enc_reset_btn"):
                if st.session_state["enc_history"]:
                    st.session_state.df = st.session_state["enc_history"][0].copy()
                else:
                    st.session_state.df = st.session_state.original_df.copy()
                st.session_state["enc_history"] = [] 
                st.session_state["enc_msg"]     = "Encoding reset"
                st.rerun()
 
        divider()
 
        # ── 4. Download ──
        sec("Download Encoded Dataset")
 
        rows, cols = st.session_state.df.shape
        cat_remaining = st.session_state.df.select_dtypes(include="object").shape[1]
 
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #151c28;border-radius:10px;'
            f'padding:0.85rem 1.2rem;margin-bottom:0.9rem;font-family:\'DM Mono\',monospace;'
            f'font-size:0.72rem;color:#64748b;">'
            f'<span style="color:#e2e8f0;">{rows:,}</span> rows &nbsp;·&nbsp; '
            f'<span style="color:#e2e8f0;">{cols}</span> columns &nbsp;·&nbsp; '
            f'<span style="color:{"#f59e0b" if cat_remaining else "#22c55e"};">'
            f'{cat_remaining} categorical column{"s" if cat_remaining != 1 else ""} remaining'
            f'</span></div>',
            unsafe_allow_html=True,
        )
 
        csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇  Download Encoded CSV",
            data=csv_bytes,
            file_name="encoded_data.csv",
            mime="text/csv",
            key="enc_download",
        )
 


    # ================= TAB 5 — FEATURE IMPORTANCE =================
    with eda_tab5:
        def sec(title):
            st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)

        def divider():
            st.markdown(
                "<hr style='border:none;border-top:1px solid #151c28;margin:1.6rem 0;'>",
                unsafe_allow_html=True,
            )

        def banner(msg, kind="ok"):
            icons = {"ok": "✓", "warn": "⚠", "err": "✕"}
            st.markdown(
                f'<div class="banner banner-{kind}">'
                f'{icons[kind]} &nbsp;<span>{msg}</span></div>',
                unsafe_allow_html=True,
            )

        df = st.session_state.df

        non_numeric = df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()

        if non_numeric:
            banner(
                f"Please encode these columns first in the <strong>Encoding</strong> tab: "
                f"<code>{'</code>, <code>'.join(non_numeric)}</code>",
                "warn",
            )

        else:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()

            if len(num_cols) < 2:
                banner("At least 2 numeric columns are required to run Feature Importance.", "warn")

            else:
                # ── 1. Configuration ──
                sec("Configuration")

                cfg1, cfg2 = st.columns(2)
                with cfg1:
                    target_col = st.selectbox(
                        "Select Target Column (Y)", num_cols, key="fi_target"
                    )
                feature_cols = [c for c in num_cols if c != target_col]

                MAX_ROWS = 20_000
                is_large = len(df) > MAX_ROWS

                with cfg2:
                    n_estimators = st.slider(
                        "Number of Trees", min_value=50, max_value=500,
                        value=100, step=50, key="fi_n_est"
                    )

                target_unique = df[target_col].nunique()
                is_classifier = target_unique <= 10
                model_label   = "Classifier" if is_classifier else "Regressor"
                model_color   = "#f59e0b" if is_classifier else "#3b82f6"

                st.markdown(
                    f'<div style="background:#0d1117;border:1px solid #151c28;border-radius:8px;'
                    f'padding:0.65rem 1rem;font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                    f'color:#64748b;margin:0.5rem 0 0.4rem 0;">'
                    f'Auto-detected model &nbsp;→&nbsp; '
                    f'<span style="color:{model_color};font-weight:600;">'
                    f'Random Forest {model_label}</span>'
                    f'&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'<span style="color:#e2e8f0;">{len(feature_cols)}</span> features'
                    f'&nbsp;&nbsp;·&nbsp;&nbsp;'
                    f'<span style="color:#e2e8f0;">{target_unique}</span> unique target values'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if is_large:
                    banner(
                        f"Large dataset ({len(df):,} rows) — Feature Importance will use "
                        f"{MAX_ROWS:,} random samples for speed.",
                        "warn",
                    )

                divider()

                # ── 2. Run ──
                sec("Run Feature Importance")

                if st.button("▶  Run Feature Importance", key="fi_btn"):
                    data = df[feature_cols + [target_col]].dropna()

                    if data.shape[0] < 10:
                        banner(
                            "Not enough clean rows after dropping missing values (need ≥ 10).",
                            "err",
                        )
                    else:
                        if len(data) > MAX_ROWS:
                            data = data.sample(MAX_ROWS, random_state=42)

                        X = data[feature_cols]
                        y = data[target_col]

                        with st.spinner("Training Random Forest… this may take a moment."):
                            if is_classifier:
                                from sklearn.ensemble import RandomForestClassifier
                                model = RandomForestClassifier(
                                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                                )
                            else:
                                from sklearn.ensemble import RandomForestRegressor
                                model = RandomForestRegressor(
                                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                                )
                            model.fit(X, y)

                        importance_df = pd.DataFrame({
                            "Feature"         : feature_cols,
                            "Importance Score": np.round(model.feature_importances_, 4),
                            "Rank"            : range(1, len(feature_cols) + 1),
                        }).sort_values("Importance Score", ascending=False).reset_index(drop=True)
                        importance_df["Rank"]         = range(1, len(importance_df) + 1)
                        importance_df["Cumulative %"] = (
                            importance_df["Importance Score"].cumsum() * 100
                        ).round(2)

                        st.session_state["importance_df"]  = importance_df
                        st.session_state["fi_target_used"] = target_col
                        st.session_state["fi_model_label"] = model_label
                        st.session_state["fi_rows_used"]   = len(data)

                # ── 3. Results ──
                if "importance_df" in st.session_state:
                    importance_df = st.session_state["importance_df"]
                    divider()

                    sec(f"Results — Target: {st.session_state['fi_target_used']}  "
                        f"({st.session_state['fi_model_label']})")

                    top_n     = min(3, len(importance_df))
                    card_cols = st.columns(top_n)
                    medal     = ["🥇", "🥈", "🥉"]

                    for i, col in enumerate(card_cols):
                        row = importance_df.iloc[i]
                        col.markdown(
                            f'<div style="background:#0d1117;border:1px solid #151c28;'
                            f'border-top:3px solid #3b82f6;border-radius:10px;'
                            f'padding:0.9rem 1rem;text-align:center;">'
                            f'<div style="font-size:1.3rem;">{medal[i]}</div>'
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                            f'color:#64748b;text-transform:uppercase;letter-spacing:0.1em;'
                            f'margin:0.3rem 0 0.2rem 0;">#{int(row["Rank"])}</div>'
                            f'<div style="font-size:0.85rem;font-weight:600;color:#e2e8f0;'
                            f'word-break:break-all;">{row["Feature"]}</div>'
                            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.78rem;'
                            f'color:#3b82f6;margin-top:0.3rem;">{row["Importance Score"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    row_height = 35
                    header_h   = 38
                    table_h    = min(600, header_h + len(importance_df) * row_height)
                    st.dataframe(
                        importance_df,
                        use_container_width=True,
                        hide_index=True,
                        height=table_h,
                    )

                    rows_used = st.session_state.get("fi_rows_used", "?")
                    st.caption(f"Trained on {rows_used:,} rows · {n_estimators} trees")

                    divider()

                    # ── 4. Chart ──
                    sec("Importance Chart")

                    fig, ax = plt.subplots(
                        figsize=(8, max(3, len(importance_df) * 0.42))
                    )
                    fig.patch.set_facecolor("#0d1117")
                    ax.set_facecolor("#0d1117")

                    n          = len(importance_df)
                    alphas     = np.linspace(1.0, 0.35, n)
                    bar_colors = [(59/255, 130/255, 246/255, a) for a in alphas]

                    bars = ax.barh(
                        importance_df["Feature"][::-1],
                        importance_df["Importance Score"][::-1],
                        color=bar_colors[::-1],
                        edgecolor="none",
                    )

                    for bar, val in zip(bars, importance_df["Importance Score"][::-1]):
                        ax.text(
                            bar.get_width() + 0.001,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.4f}",
                            va="center", ha="left",
                            color="#64748b", fontsize=7,
                        )

                    ax.set_xlabel("Importance Score", color="#64748b", fontsize=8)
                    ax.tick_params(colors="#64748b", labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.xaxis.grid(True, color="#151c28", linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    download_chart(fig, key="fi_chart_download")
                    plt.close(fig)

                    divider()

                    # ── 5. Download ──
                    sec("Download")

                    dl1, dl2 = st.columns(2)
                    with dl1:
                        csv_data = st.session_state.df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇  Download Dataset CSV",
                            data=csv_data,
                            file_name="feature_selected_data.csv",
                            mime="text/csv",
                            key="fi_dl_dataset",
                        )
                    with dl2:
                        imp_csv = importance_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇  Download Importance Scores CSV",
                            data=imp_csv,
                            file_name="feature_importance.csv",
                            mime="text/csv",
                            key="fi_dl_importance",
                        )



    # ================= TAB 6 — MODEL TRAINING =================
    with eda_tab6:
 
        def sec(title):
            st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)
 
        def divider():
            st.markdown(
                "<hr style='border:none;border-top:1px solid #0f1520;margin:1.6rem 0;'>",
                unsafe_allow_html=True,
            )
 
        def banner(msg, kind="ok"):
            icons = {"ok": "✓", "warn": "⚠", "err": "✕"}
            st.markdown(
                f'<div class="banner banner-{kind}">'
                f'{icons[kind]} &nbsp;<span>{msg}</span></div>',
                unsafe_allow_html=True,
            )
 
        df = st.session_state.df
 
        # ── Guards ──
        if df.isnull().sum().sum() > 0:
            banner(
                "Dataset has missing values — handle them in the "
                "<strong>Fill Missing Values</strong> tab first.",
                "err",
            )
            st.stop()
 
        non_numeric = df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        if non_numeric:
            banner(
                f"Non-numeric columns found: "
                f"<code>{'</code>, <code>'.join(non_numeric)}</code> — "
                f"encode them in the <strong>Encoding</strong> tab first.",
                "err",
            )
            st.stop()
 
        # bool → int
        df = df.replace({True: 1, False: 0})
 
        MAX_ROWS = 20_000
 
        # ── 1. Target + Task Detection ──
        sec("Target Column")
 
        target = st.selectbox("Select Target (Y)", df.columns, key="train_target")
        X = df.drop(columns=[target])
        y = df[target]
 
        task_type     = "Classification" if (y.dtype == object or y.nunique() <= 15) else "Regression"
        task_color    = "#f59e0b" if task_type == "Classification" else "#3b82f6"
        is_large      = len(X) > MAX_ROWS
        is_highdim    = X.shape[1] > 100
 
        st.markdown(
            f'<div style="background:#0d1117;border:1px solid #151c28;border-radius:8px;'
            f'padding:0.7rem 1.1rem;font-family:\'DM Mono\',monospace;font-size:0.72rem;'
            f'color:#64748b;margin:0.5rem 0;">'
            f'Task &nbsp;→&nbsp; <span style="color:{task_color};font-weight:600;">{task_type}</span>'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'<span style="color:#e2e8f0;">{X.shape[1]}</span> features'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'<span style="color:#e2e8f0;">{len(X):,}</span> rows'
            f'&nbsp;&nbsp;·&nbsp;&nbsp;'
            f'<span style="color:#e2e8f0;">{y.nunique()}</span> unique target values'
            f'</div>',
            unsafe_allow_html=True,
        )
 
        if is_large:
            banner(
                f"Large dataset ({len(X):,} rows) — training will use "
                f"{MAX_ROWS:,} random samples.",
                "warn",
            )
        if is_highdim:
            banner(
                f"High-dimensional data ({X.shape[1]} features) — "
                f"PCA will auto-apply (90% variance retained).",
                "warn",
            )
 
        divider()
 
        # ── 2. Train / Test Split ──
        sec("Train / Test Split")
        test_size = st.slider(
            "Test Set Size", min_value=0.10, max_value=0.40,
            value=0.20, step=0.05, key="train_test_size",
            help="Fraction of data held out for evaluation"
        )
        train_rows = int(min(len(X), MAX_ROWS) * (1 - test_size))
        test_rows  = int(min(len(X), MAX_ROWS) * test_size)
        st.caption(f"~{train_rows:,} training rows  ·  ~{test_rows:,} test rows")
 
        divider()
 
        # ── 3. Model Selection ──
        sec("Select Model")
 
        if task_type == "Regression":
            model_list = ["Linear Regression", "KNN", "SVM", "Decision Tree", "Random Forest"]
        else:
            model_list = ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest"]
 
        model_name = st.selectbox("Model", model_list, key="model_select")
 
        # ── 4. Hyperparameter Tuning ──
        hp = {}
        hyper_models = ["Decision Tree", "Random Forest", "SVM"]
 
        if model_name in hyper_models:
            enable_tuning = st.toggle(
                "Enable Hyperparameter Tuning",
                value=False,
                key="hp_toggle",
            )
 
            if enable_tuning:
                st.markdown("<br>", unsafe_allow_html=True)
 
                if model_name == "Decision Tree":
                    c1, c2, c3 = st.columns(3)
                    hp["max_depth"] = c1.slider(
                        "Max Depth", 1, 20, 5,
                        help="Higher = more complex, risk of overfitting",
                    )
                    hp["min_samples_split"] = c2.slider(
                        "Min Samples Split", 2, 20, 5,
                        help="Min samples needed to split a node",
                    )
                    hp["min_samples_leaf"] = c3.slider(
                        "Min Samples Leaf", 1, 20, 2,
                        help="Min samples at a leaf node",
                    )
                    st.caption("Tip: Max Depth 3–8 is usually best. Too high → overfitting.")
 
                elif model_name == "Random Forest":
                    c1, c2, c3, c4 = st.columns(4)
                    hp["n_estimators"] = c1.slider(
                        "N Estimators", 50, 500, 150, step=50,
                        help="More trees = better but slower",
                    )
                    hp["max_depth"] = c2.slider(
                        "Max Depth", 1, 20, 7,
                        help="Max depth of each tree",
                    )
                    hp["min_samples_split"] = c3.slider(
                        "Min Samples Split", 2, 20, 5,
                    )
                    hp["min_samples_leaf"] = c4.slider(
                        "Min Samples Leaf", 1, 20, 2,
                    )
                    st.caption("Tip: Start with 100–200 trees. Max Depth 5–10 works well.")
 
                elif model_name == "SVM":
                    c1, c2, c3 = st.columns(3)
                    hp["C"] = c1.select_slider(
                        "C (Regularization)",
                        options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                        value=1.0,
                        help="Higher C = less regularization",
                    )
                    hp["kernel"] = c2.selectbox(
                        "Kernel", ["rbf", "linear", "poly", "sigmoid"],
                        help="rbf works best for most cases",
                    )
                    hp["gamma"] = c3.selectbox(
                        "Gamma", ["scale", "auto"],
                        help="scale = 1/(n_features * X.var())",
                    )
                    st.caption("Tip: rbf kernel with C=1.0 is a safe start.")
 
            else:
                # defaults
                if model_name == "Decision Tree":
                    hp = {"max_depth": 5, "min_samples_split": 5, "min_samples_leaf": 2}
                elif model_name == "Random Forest":
                    hp = {"n_estimators": 150, "max_depth": 7,
                          "min_samples_split": 5, "min_samples_leaf": 2}
                elif model_name == "SVM":
                    hp = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
 
        divider()
 
        # ── 5. Train Button ──
        sec("Train")
 
        if st.button("▶  Train Model", key="train_btn"):
            X_s = X.copy()
            y_s = y.copy()
 
            if len(X_s) > MAX_ROWS:
                X_s = X_s.sample(MAX_ROWS, random_state=42)
                y_s = y_s.loc[X_s.index]
 
            with st.spinner("Training… please wait."):
                from sklearn.model_selection import train_test_split, cross_val_score
                from sklearn.preprocessing   import StandardScaler
                from sklearn.decomposition   import PCA
                import time
 
                start = time.time()
 
                X_train, X_test, y_train, y_test = train_test_split(
                    X_s, y_s, test_size=test_size, random_state=42
                )
 
                # scaling
                needs_scale = model_name in [
                    "KNN", "SVM", "Linear Regression", "Logistic Regression"
                ]
                scaler = None
                if needs_scale or is_highdim:
                    scaler  = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test  = scaler.transform(X_test)
                else:
                    X_train = X_train.values
                    X_test  = X_test.values
 
                # PCA
                pca = None
                pca_n_after = None
                if is_highdim:
                    pca         = PCA(n_components=0.90, random_state=42)
                    X_train     = pca.fit_transform(X_train)
                    X_test      = pca.transform(X_test)
                    pca_n_after = X_train.shape[1]
 
                # model init
                if model_name == "Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
 
                elif model_name == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
 
                elif model_name == "KNN":
                    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                    model = (
                        KNeighborsClassifier(n_neighbors=5, weights="distance")
                        if task_type == "Classification"
                        else KNeighborsRegressor(n_neighbors=5, weights="distance")
                    )
 
                elif model_name == "SVM":
                    from sklearn.svm import SVC, SVR
                    model = (
                        SVC(kernel=hp["kernel"], C=hp["C"], gamma=hp["gamma"], probability=True)
                        if task_type == "Classification"
                        else SVR(kernel=hp["kernel"], C=hp["C"], gamma=hp["gamma"])
                    )
 
                elif model_name == "Decision Tree":
                    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                    model = (
                        DecisionTreeClassifier(
                            max_depth=hp["max_depth"],
                            min_samples_split=hp["min_samples_split"],
                            min_samples_leaf=hp["min_samples_leaf"],
                            random_state=42,
                        )
                        if task_type == "Classification"
                        else DecisionTreeRegressor(
                            max_depth=hp["max_depth"],
                            min_samples_split=hp["min_samples_split"],
                            min_samples_leaf=hp["min_samples_leaf"],
                            random_state=42,
                        )
                    )
 
                else:  # Random Forest
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    model = (
                        RandomForestClassifier(
                            n_estimators=hp.get("n_estimators", 150),
                            max_depth=hp.get("max_depth", 7),
                            min_samples_split=hp.get("min_samples_split", 5),
                            min_samples_leaf=hp.get("min_samples_leaf", 2),
                            random_state=42, n_jobs=-1,
                        )
                        if task_type == "Classification"
                        else RandomForestRegressor(
                            n_estimators=hp.get("n_estimators", 150),
                            max_depth=hp.get("max_depth", 7),
                            min_samples_split=hp.get("min_samples_split", 5),
                            min_samples_leaf=hp.get("min_samples_leaf", 2),
                            random_state=42, n_jobs=-1,
                        )
                    )
 
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
 
                # 5-fold cross-val score on training set
                cv_scoring = "accuracy" if task_type == "Classification" else "r2"
                cv_scores  = cross_val_score(
                    model, X_train, y_train, cv=5, scoring=cv_scoring, n_jobs=-1
                )
 
                elapsed = time.time() - start
 
            # save everything
            st.session_state.update({
                "trained_model"      : model,
                "trained_preds"      : preds.tolist(),
                "trained_y_test"     : y_test.tolist(),
                "trained_task"       : task_type,
                "trained_model_name" : model_name,
                "trained_target"     : target,
                "trained_features"   : X.columns.tolist(),
                "trained_pca_n"      : pca_n_after,
                "trained_hp"         : hp,
                "train_time"         : elapsed,
                "train_cv_scores"    : cv_scores.tolist(),
                "train_cv_metric"    : cv_scoring,
                "scaler"             : scaler,
                "pca"                : pca,
            })
 
        # ── 6. Results ──
        if "trained_model" in st.session_state:
            preds      = np.array(st.session_state["trained_preds"])
            y_test_arr = np.array(st.session_state["trained_y_test"])
            task       = st.session_state["trained_task"]
            mdl_name   = st.session_state["trained_model_name"]
            pca_n      = st.session_state["trained_pca_n"]
            used_hp    = st.session_state.get("trained_hp", {})
            cv_scores  = np.array(st.session_state.get("train_cv_scores", []))
            cv_metric  = st.session_state.get("train_cv_metric", "")
 
            divider()
            sec(f"Results — {mdl_name}")
 
            if pca_n:
                banner(f"PCA applied → reduced to <strong>{pca_n}</strong> components (90% variance retained).", "warn")
 
            if used_hp:
                hp_str = " &nbsp;·&nbsp; ".join(
                    [f'<span style="color:#e2e8f0;">{k}</span>: '
                     f'<span style="color:#3b82f6;">{v}</span>'
                     for k, v in used_hp.items()]
                )
                st.markdown(
                    f'<div style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                    f'color:#64748b;margin-bottom:0.8rem;">Params — {hp_str}</div>',
                    unsafe_allow_html=True,
                )
 
            # ── Metrics ──
            if task == "Regression":
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                r2   = r2_score(y_test_arr, preds)
                mse  = mean_squared_error(y_test_arr, preds)
                rmse = np.sqrt(mse)
                mae  = mean_absolute_error(y_test_arr, preds)
 
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", round(r2,   4))
                m2.metric("MAE",      round(mae,  4))
                m3.metric("MSE",      round(mse,  4))
                m4.metric("RMSE",     round(rmse, 4))
 
                # Actual vs Predicted chart
                divider()
                sec("Actual vs Predicted")
                sample_n  = min(200, len(y_test_arr))
                idx       = np.random.choice(len(y_test_arr), sample_n, replace=False)
                fig, ax   = plt.subplots(figsize=(7, 3.5))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#0d1117")
                ax.scatter(y_test_arr[idx], preds[idx],
                           color="#3b82f6", alpha=0.55, s=18, edgecolors="none")
                lims = [min(y_test_arr.min(), preds.min()),
                        max(y_test_arr.max(), preds.max())]
                ax.plot(lims, lims, color="#f59e0b", linewidth=1.2,
                        linestyle="--", label="Perfect fit")
                ax.set_xlabel("Actual",    color="#64748b", fontsize=8)
                ax.set_ylabel("Predicted", color="#64748b", fontsize=8)
                ax.tick_params(colors="#64748b", labelsize=7)
                ax.legend(fontsize=7, labelcolor="#94a3b8",
                          facecolor="#0d1117", edgecolor="#1e293b")
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.grid(True, color="#151c28", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                download_chart(fig, key="train_avp_download")
                plt.close(fig)
 
            else:
                from sklearn.metrics import (
                    accuracy_score, confusion_matrix, classification_report
                )
                acc = accuracy_score(y_test_arr, preds)
                st.metric("Test Accuracy", f"{round(acc * 100, 2)}%")
 
                divider()
                sec("Classification Report")
                report    = classification_report(
                    y_test_arr, preds, output_dict=True, zero_division=0
                )
                report_df = pd.DataFrame(report).transpose().round(3)
                row_h     = 35
                tbl_h     = min(500, 38 + len(report_df) * row_h)
                st.dataframe(report_df, use_container_width=True, height=tbl_h)
 
                divider()
                sec("Confusion Matrix")
                cm       = confusion_matrix(y_test_arr, preds)
                n_cls    = cm.shape[0]
                fig_size = max(3, min(7, n_cls * 1.2))
                fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#0d1117")
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=0.5, linecolor="#161b22",
                    annot_kws={"size": 9},
                )
                ax.set_xlabel("Predicted", color="#64748b", fontsize=8)
                ax.set_ylabel("Actual",    color="#64748b", fontsize=8)
                ax.tick_params(colors="#64748b", labelsize=7)
                for s in ax.spines.values():
                    s.set_visible(False)
                plt.tight_layout()
                _, mid, _ = st.columns([1, 2, 1])
                with mid:
                    st.pyplot(fig)
                    download_chart(fig, key="train_cm_download")
                plt.close(fig)
 
            # ── Cross-Validation ──
            if len(cv_scores) > 0:
                divider()
                sec("5-Fold Cross-Validation")
                cv_label = "Accuracy" if cv_metric == "accuracy" else "R²"
 
                cv1, cv2, cv3 = st.columns(3)
                cv1.metric(f"Mean {cv_label}",   round(cv_scores.mean(), 4))
                cv2.metric(f"Std Dev",            round(cv_scores.std(),  4))
                cv3.metric(f"Min / Max",
                           f"{round(cv_scores.min(),3)} / {round(cv_scores.max(),3)}")
 
                # small fold chart
                fig, ax = plt.subplots(figsize=(5, 2.2))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#0d1117")
                fold_colors = ["#3b82f6" if s >= cv_scores.mean() else "#334155"
                               for s in cv_scores]
                ax.bar(
                    [f"Fold {i+1}" for i in range(len(cv_scores))],
                    cv_scores,
                    color=fold_colors, edgecolor="none",
                )
                ax.axhline(cv_scores.mean(), color="#f59e0b",
                           linewidth=1.2, linestyle="--", label="Mean")
                ax.set_ylabel(cv_label, color="#64748b", fontsize=7)
                ax.tick_params(colors="#64748b", labelsize=7)
                ax.legend(fontsize=7, labelcolor="#94a3b8",
                          facecolor="#0d1117", edgecolor="#1e293b")
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.yaxis.grid(True, color="#151c28", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
 
            # ── Training summary ──
            divider()
            banner(
                f"<strong>{mdl_name}</strong> trained successfully &nbsp;·&nbsp; "
                f"⏱ {round(st.session_state['train_time'], 2)}s",
                "ok",
            )
 
            # ── Download Model ──
            divider()
            sec("Download Model")
 
            model_bundle = {
                "model"   : st.session_state["trained_model"],
                "scaler"  : st.session_state.get("scaler"),
                "pca"     : st.session_state.get("pca"),
                "features": st.session_state["trained_features"],
                "target"  : st.session_state["trained_target"],
                "task"    : st.session_state["trained_task"],
                "params"  : st.session_state.get("trained_hp", {}),
            }
            buf = io.BytesIO()
            pickle.dump(model_bundle, buf)
            buf.seek(0)
 
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇  Download Model (.pkl)",
                    data=buf,
                    file_name=f"{mdl_name.replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream",
                    key="train_dl_model",
                )
            with dl2:
                csv_out = st.session_state.df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇  Download Dataset CSV",
                    data=csv_out,
                    file_name="final_dataset.csv",
                    mime="text/csv",
                    key="train_dl_csv",
                )
