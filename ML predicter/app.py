import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle

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

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, engine="pyarrow")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
        )
    return df

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
    color: #1e293b;
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
        st.session_state.original_df = load_data(file)
        st.session_state.df = st.session_state.original_df.copy()

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


# ---------- OVERVIEW ----------
    with eda_tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Rows:", df.shape[0])
            st.write("Columns:", df.shape[1])
            st.write("Column Names:")
            st.write(df.columns.tolist())

        with col2:
            st.write("Data Types:")
            st.write(df.dtypes)

        st.subheader("Statistical Summary")
        st.write(df.describe(include="all"))

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100

        missing_df = pd.DataFrame({
            "Missing Values": missing,
            "Percentage (%)": missing_percent.round(2)
        })

        st.dataframe(missing_df[missing_df["Missing Values"] > 0], use_container_width=True)

        st.subheader("Duplicate Rows")
        st.write("Total Duplicate Rows:", df.duplicated().sum())


# ---------- COLUMN ANALYZER ----------
    with eda_tab2:
        column = st.selectbox("Select Column", df.columns, key="col_analyzer")

        missing_count   = df[column].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        unique_values   = df[column].nunique()

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Data Type",      str(df[column].dtype))
        colB.metric("Missing Values", int(missing_count))
        colC.metric("Missing %",      f"{missing_percent:.2f}%")
        colD.metric("Unique Values",  int(unique_values))

        # ── NUMERIC ──
        if np.issubdtype(df[column].dtype, np.number):

            mean_val   = df[column].mean()
            median_val = df[column].median()
            std_val    = df[column].std()
            min_val    = df[column].min()
            max_val    = df[column].max()
            skew_val   = df[column].skew()

            colE, colF, colG = st.columns(3)
            colE.metric("Mean",     round(mean_val, 2))
            colF.metric("Median",   round(median_val, 2))
            colG.metric("Std Dev",  round(std_val, 2))

            colH, colI, colJ = st.columns(3)
            colH.metric("Min",      round(min_val, 2))
            colI.metric("Max",      round(max_val, 2))
            colJ.metric("Skewness", round(skew_val, 2))

            chart_l, chart_r = st.columns(2)
            clean = df[column].dropna().tolist()

            with chart_l:
                st.markdown("##### Distribution")
                fig1 = make_num_plots(clean, column, mean_val, median_val)
                st.pyplot(fig1)
                plt.close()

            with chart_r:
                st.markdown("##### Box Plot")
                fig2 = make_box_plot(clean, column)
                st.pyplot(fig2)
                plt.close()

        # ── CATEGORICAL ──
        else:
            mode_val = df[column].mode()
            if not mode_val.empty:
                most_frequent = mode_val[0]
                top_count     = df[column].value_counts().iloc[0]
            else:
                most_frequent = "N/A"
                top_count     = 0

            colE, colF = st.columns(2)
            colE.metric("Most Frequent Value", str(most_frequent))
            colF.metric("Top Value Count",     int(top_count))

            vc = df[column].value_counts()
            MAX_BARS  = 20
            truncated = len(vc) > MAX_BARS
            vc_plot   = vc.head(MAX_BARS)

            chart_l, chart_r = st.columns(2)

            # Bar chart
            with chart_l:
                title = f"##### Top {MAX_BARS} Values" if truncated else "##### Value Counts"
                st.markdown(title)
                fig, ax = plt.subplots(figsize=(5, max(3, len(vc_plot) * 0.35)))
                fig.patch.set_facecolor("#0d1117")
                ax.set_facecolor("#0d1117")
                colors = ["#3b82f6" if i == 0 else "#1e3a5f" for i in range(len(vc_plot))]
                ax.barh(vc_plot.index[::-1], vc_plot.values[::-1],
                        color=colors[::-1], edgecolor="none")
                ax.set_xlabel("Count", color="#64748b", fontsize=8)
                ax.tick_params(colors="#64748b", labelsize=7)
                for s in ax.spines.values(): s.set_visible(False)
                ax.xaxis.grid(True, color="#151c28", linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Pie chart
            with chart_r:
                if len(vc) <= 10:
                    st.markdown("##### Distribution")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    fig.patch.set_facecolor("#0d1117")
                    pie_colors = ["#3b82f6","#6366f1","#8b5cf6","#a78bfa",
                                  "#f472b6","#fb7185","#f59e0b","#22c55e",
                                  "#14b8a6","#0ea5e9"]
                    wedges, texts, autotexts = ax.pie(
                        vc.values, labels=vc.index,
                        colors=pie_colors[:len(vc)],
                        autopct="%1.1f%%", startangle=140,
                        textprops={"color": "#94a3b8", "fontsize": 7},
                        wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5}
                    )
                    for at in autotexts:
                        at.set_color("#e2e8f0")
                        at.set_fontsize(7)
                    ax.set_facecolor("#0d1117")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.markdown("##### Top 10 Share")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    fig.patch.set_facecolor("#0d1117")
                    top10  = vc.head(10)
                    other  = vc.iloc[10:].sum()
                    pie_vc = pd.concat([top10, pd.Series({"Other": other})])
                    pie_colors = ["#3b82f6","#6366f1","#8b5cf6","#a78bfa",
                                  "#f472b6","#fb7185","#f59e0b","#22c55e",
                                  "#14b8a6","#0ea5e9","#475569"]
                    wedges, texts, autotexts = ax.pie(
                        pie_vc.values, labels=pie_vc.index,
                        colors=pie_colors[:len(pie_vc)],
                        autopct="%1.1f%%", startangle=140,
                        textprops={"color": "#94a3b8", "fontsize": 7},
                        wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5}
                    )
                    for at in autotexts:
                        at.set_color("#e2e8f0")
                        at.set_fontsize(7)
                    ax.set_facecolor("#0d1117")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()


# -------- FILL MISSING VALUES --------
    with eda_tab3:
        st.subheader("Missing Values Table")

        df = st.session_state.df

        # -------- Missing calculation --------
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            st.success("No missing values in dataset 🎉")
        else:
            missing_df = pd.DataFrame({
                "Column": missing.index,
                "Missing Values": missing.values
            }).sort_values(by="Missing Values", ascending=False)

            st.caption("Only columns with missing values are shown below")
            st.dataframe(missing_df, use_container_width=True)

            # -------- Fill section --------
            st.markdown("### Fill Missing Values")
            col1, col2 = st.columns(2)

            with col1:
                selected_col = st.selectbox(
                    "Select Column",
                    missing_df["Column"].tolist(),
                    key="col_select"
                )
            with col2:
                method = st.selectbox(
                    "Select Method",
                    ["Mean", "Median", "Mode"],
                    key="method_select"
                )

            # -------- Fill button --------
            if st.button("Fill Missing", key="fill_btn"):
                if method in ["Mean", "Median"] and not np.issubdtype(df[selected_col].dtype, np.number):
                    st.error("Mean/Median can only be applied to numeric columns ❌")
                else:
                    if method == "Mean":
                        value = df[selected_col].mean()
                    elif method == "Median":
                        value = df[selected_col].median()
                    else:
                        mode_val = df[selected_col].mode()
                        value = mode_val[0] if not mode_val.empty else None

                    if value is None:
                        st.warning("No valid value found to fill ❌")
                    else:
                        df[selected_col] = df[selected_col].fillna(value)
                        st.session_state.df = df
                        st.session_state["last_action"] = f"'{selected_col}' filled using {method}"
                        st.rerun()

        # -------- Show success after rerun --------
        if "last_action" in st.session_state:
            st.success(st.session_state["last_action"] + " ✅")
            del st.session_state["last_action"]

        # -------- Delete Column --------
        st.markdown("### Delete a Column")
        df = st.session_state.df  
        col1, col2 = st.columns([3, 1])
        
        with col1:
            del_col = st.selectbox(
                "Select column to delete",
                df.columns.tolist(),
                key="fill_del_col"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Delete Column", key="fill_del_btn"):
                if "df_history" not in st.session_state:
                    st.session_state["df_history"] = []
                st.session_state["df_history"].append(st.session_state.df.copy())
                st.session_state.df = st.session_state.df.drop(columns=[del_col])
                st.session_state["last_action"] = f"'{del_col}' column deleted"
                st.rerun()

        # -------- Reset --------
        if st.button("🔄 Reset to Original Dataset", key="fill_reset_btn"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state["df_history"] = []
            st.session_state["last_action"] = "Dataset reset to original"
            st.rerun()

        # -------- Download cleaned data --------
        st.markdown("### Download Cleaned Dataset")
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        ) 


#---------- ENCODING ----------
    with eda_tab4:
        df = st.session_state.df
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if not cat_cols:
            st.success("No categorical columns found 🎉")
        else:
            st.markdown("### Categorical Columns")
            cat_info = pd.DataFrame({
                "Column": cat_cols,
                "Unique Values": [df[col].nunique() for col in cat_cols]
            }).sort_values(by="Unique Values", ascending=False)
            st.dataframe(cat_info, use_container_width=True)

            st.markdown("### Apply Encoding")
            col1, col2 = st.columns(2)

            with col1:
                selected_col = st.selectbox("Select Column", cat_cols, key="enc_col")

            with col2:
                method = st.selectbox(
                    "Encoding Method",
                    ["Label Encoding", "One Hot Encoding", "Manual (Ordinal)"],
                    key="enc_method"
                )

            unique_count = df[selected_col].nunique()
            st.info(f"Unique values in '{selected_col}': {unique_count}")

            # -------- Warnings --------
            has_missing = bool(df[selected_col].isnull().sum() > 0)
            one_hot_blocked = method == "One Hot Encoding" and unique_count > 25

            if has_missing:
                st.error(f"❌ '{selected_col}' has missing values. Fill them first in 'Fill Missing Values' tab.")
            if one_hot_blocked:
                st.error(f"❌ '{selected_col}' has {unique_count} unique values (max 25). Use Label Encoding instead.")

            # -------- Manual Input --------
            order = ""
            if method == "Manual (Ordinal)":
                order = st.text_input(
                    "Enter order (lowest → highest, comma separated)",
                    placeholder="poor, fair, excellent",
                    key="ordinal_input"
                )

            # -------- Apply button --------
            is_disabled = has_missing or one_hot_blocked
            if st.button("Apply Encoding", key="enc_btn", disabled=is_disabled):
                df = st.session_state.df.copy()
                df[selected_col] = df[selected_col].astype(str)

                if method == "Label Encoding":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df[selected_col] = le.fit_transform(df[selected_col])
                    st.session_state["enc_msg"] = f"Label Encoding applied on '{selected_col}'"
                elif method == "One Hot Encoding":
                    dummies = pd.get_dummies(df[selected_col], prefix=selected_col)
                    df = pd.concat([df.drop(columns=[selected_col]), dummies], axis=1)
                    st.session_state["enc_msg"] = f"One Hot Encoding applied on '{selected_col}'"
                else:
                    if not order.strip():
                        st.error("Please enter order ❌")
                    else:
                        values = [x.strip() for x in order.split(",")]
                        actual_values = df[selected_col].dropna().unique().tolist()
                        if not set(values).issubset(set(actual_values)):
                            st.error(f"Invalid values ❌ — valid values are: {actual_values}")
                        else:
                            mapping = {val: i for i, val in enumerate(values)}
                            df[selected_col] = df[selected_col].map(mapping)
                            st.session_state["enc_msg"] = f"Manual Encoding applied on '{selected_col}'"

                if "enc_msg" in st.session_state:
                    if "df_history" not in st.session_state:
                        st.session_state["df_history"] = []
                    st.session_state["df_history"].append(st.session_state.df.copy())

                st.session_state.df = df 
                st.rerun()   

        # -------- Success message --------
        if "enc_msg" in st.session_state:
            st.success(st.session_state["enc_msg"] + " ✅")
            del st.session_state["enc_msg"]

        # -------- Undo / Reset --------
        st.markdown("### Undo / Reset")
        undo_col, reset_col = st.columns(2)

        with undo_col:
            history = st.session_state.get("df_history", [])
            if st.button(
                f"↩ Undo Last Encoding ({len(history)} steps)",
                key="undo_btn",
                disabled=len(history) == 0
            ):
                st.session_state.df = history.pop()
                st.session_state["df_history"] = history
                st.session_state["enc_msg"] = "Last encoding undone"
                st.rerun()

        with reset_col:
            if st.button("🔄 Reset to Original Dataset", key="reset_btn"):
                st.session_state.df = st.session_state.original_df.copy()
                st.session_state["df_history"] = []
                st.session_state["enc_msg"] = "Dataset reset to original"
                st.rerun()

        st.caption(f"Current dataset shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")

        # -------- Download --------
        st.markdown("### Download Encoded Dataset")
        csv = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="encoded_data.csv",
            mime="text/csv"
        )       


# ================= FEATURE IMPORTANCE TAB =================
    with eda_tab5:
        df = st.session_state.df
 
        # -------- Check non-numeric columns --------
        non_numeric = df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        if non_numeric:
            st.warning(f"⚠ Please encode these columns first: {non_numeric}")
        else:
            # -------- Numeric columns --------
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
 
            if len(num_cols) < 2:
                st.warning("⚠ At least 2 numeric columns required.")
            else:
                # -------- Target selection --------
                target_col   = st.selectbox("Select Target Column (Y)", num_cols, key="fi_target")
                feature_cols = [c for c in num_cols if c != target_col]
 
                # -------- Sampling warning --------
                MAX_ROWS = 20000
                if len(df) > MAX_ROWS:
                    st.warning(f"⚡ Large dataset ({len(df):,} rows) — Feature Importance will use {MAX_ROWS:,} random samples.")
 
                # -------- Run --------
                if st.button("Run Feature Importance", key="fi_btn"):
                    data = df[feature_cols + [target_col]].dropna()
 
                    if data.shape[0] < 10:
                        st.error("❌ Not enough clean data after removing missing values.")
                    else:
                        # ---- Sampling ----
                        if len(data) > MAX_ROWS:
                            data = data.sample(MAX_ROWS, random_state=42)
 
                        X = data[feature_cols]
                        y = data[target_col]
 
                        with st.spinner("Calculating Feature Importance... ⏳"):
                            # ---- Classifier ya Regressor auto detect ----
                            if y.nunique() <= 10:
                                from sklearn.ensemble import RandomForestClassifier
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                            else:
                                from sklearn.ensemble import RandomForestRegressor
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
 
                            model.fit(X, y)
                            importances = model.feature_importances_
 
                        importance_df = pd.DataFrame({
                            "Feature"          : feature_cols,
                            "Importance Score" : np.round(importances, 4)
                        }).sort_values("Importance Score", ascending=False).reset_index(drop=True)
 
                        st.session_state["importance_df"]  = importance_df
                        st.session_state["fi_target_used"] = target_col
 
                # -------- Show Results --------
                if "importance_df" in st.session_state:
                    importance_df = st.session_state["importance_df"]
 
                    st.markdown(f"### Results — Target: `{st.session_state['fi_target_used']}`")
                    st.dataframe(importance_df, use_container_width=True)
 
                    # -------- Chart --------
                    st.markdown("### Importance Chart")
                    fig, ax = plt.subplots(figsize=(8, max(3, len(importance_df) * 0.4)))
                    fig.patch.set_facecolor("#0d1117")
                    ax.set_facecolor("#0d1117")
 
                    ax.barh(
                        importance_df["Feature"][::-1],
                        importance_df["Importance Score"][::-1],
                        color="#3b82f6"
                    )
 
                    ax.set_xlabel("Importance Score", color="#64748b", fontsize=9)
                    ax.tick_params(colors="#64748b", labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.xaxis.grid(True, color="#151c28", linewidth=0.5)
 
                    st.pyplot(fig)
                    plt.close()
 
                    # -------- Download --------
                    st.markdown("### Download")
                    dl1, dl2 = st.columns(2)
 
                    with dl1:
                        csv_data = st.session_state.df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Dataset CSV",
                            data=csv_data,
                            file_name="feature_selected_data.csv",
                            mime="text/csv"
                        )
                    with dl2:
                        imp_csv = importance_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Importance Scores CSV",
                            data=imp_csv,
                            file_name="feature_importance.csv",
                            mime="text/csv"
                        )


# ================= MODEL TRAINING TAB =================
    with eda_tab6:
        df = st.session_state.df
        has_error = False

        if df.isnull().sum().sum() > 0:
            st.error("❌ Dataset contains missing values. Please handle them first.")
            has_error = True

        if not has_error:
            non_numeric = df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
            if non_numeric:
                st.error(f"❌ Non-numeric columns found: {non_numeric}. Please encode them first.")
                has_error = True

        if not has_error:
            # -------- Convert bool to int --------
            df = df.replace({True: 1, False: 0})
    
            # -------- Select Target --------
            st.markdown("### Select Target Column")
            target = st.selectbox("Target (Y)", df.columns, key="train_target")
    
            X = df.drop(columns=[target])
            y = df[target]
    
            # -------- Detect Task --------
            task_type = "Classification" if (y.dtype == object or y.nunique() <= 15) else "Regression"
            st.info(f"Detected Task: **{task_type}** &nbsp;|&nbsp; Features: **{X.shape[1]}** &nbsp;|&nbsp; Rows: **{len(X):,}**")
    
            # -------- Sampling / PCA warnings OUTSIDE spinner --------
            MAX_ROWS = 20000
            if len(X) > MAX_ROWS:
                st.warning(f"⚡ Large dataset detected ({len(X):,} rows). Training will use {MAX_ROWS:,} random samples.")
    
            if X.shape[1] > 100:
                st.warning(f"🧬 High dimensions detected ({X.shape[1]} features). PCA will auto-apply (90% variance).")
    
            # -------- Model Selection --------
            st.markdown("### Select Model")
            if task_type == "Regression":
                model_list = ["Linear Regression", "KNN", "SVM", "Decision Tree", "Random Forest"]
            else:
                model_list = ["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest"]
    
            model_name = st.selectbox("Model", model_list, key="model_select")
    
            # -------- Train Button --------
            if st.button("🚀 Train Model", key="train_btn"):
                X_sample = X.copy()
                y_sample = y.copy()
    
                if len(X_sample) > MAX_ROWS:
                    X_sample = X_sample.sample(MAX_ROWS, random_state=42)
                    y_sample = y_sample.loc[X_sample.index]
    
                with st.spinner("Training model... please wait"):
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing  import StandardScaler
                    from sklearn.decomposition  import PCA

                    import time
                    start_time = time.time()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_sample, y_sample, test_size=0.2, random_state=42
                    )
    
                    # ── Scaling ──
                    scaler = None
                    needs_scale = model_name in ["KNN", "SVM", "Linear Regression", "Logistic Regression"]
    
                    if needs_scale or X_train.shape[1] > 100:
                        scaler  = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test  = scaler.transform(X_test)
                    else:
                        X_train = X_train.values
                        X_test  = X_test.values
    
                    # ── PCA ──
                    pca         = None
                    pca_n_after = None
    
                    if X_train.shape[1] > 100:
                        pca         = PCA(n_components=0.90, random_state=42)
                        X_train     = pca.fit_transform(X_train)
                        X_test      = pca.transform(X_test)
                        pca_n_after = X_train.shape[1]
    
                    # ── Model Init ──
                    if model_name == "Linear Regression":
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
    
                    elif model_name == "Logistic Regression":
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    
                    elif model_name == "KNN":
                        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                        model = (KNeighborsClassifier(n_neighbors=5, weights="distance")
                                if task_type == "Classification"
                                else KNeighborsRegressor(n_neighbors=5, weights="distance"))
    
                    elif model_name == "SVM":
                        from sklearn.svm import SVC, SVR
                        model = (SVC(kernel="rbf", C=1.0, gamma="scale")
                                if task_type == "Classification"
                                else SVR(kernel="rbf", C=1.0, gamma="scale"))
    
                    elif model_name == "Decision Tree":
                        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                        model = (DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
                                if task_type == "Classification"
                                else DecisionTreeRegressor(max_depth=5, min_samples_split=5, min_samples_leaf=2))
    
                    else:  # Random Forest
                        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                        model = (RandomForestClassifier(n_estimators=150, max_depth=7,
                                                        min_samples_split=5, min_samples_leaf=2,
                                                        random_state=42, n_jobs=-1)
                                if task_type == "Classification"
                                else RandomForestRegressor(n_estimators=150, max_depth=7,
                                                            min_samples_split=5, min_samples_leaf=2,
                                                            random_state=42, n_jobs=-1))
    
                    # ── Train ──
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    end_time = time.time()
                    train_time = end_time - start_time
                    st.session_state["train_time"] = train_time
                    
    
                    # ── Save to session_state ──
                    st.session_state["trained_model"]      = model
                    st.session_state["trained_preds"]      = preds.tolist()
                    st.session_state["trained_y_test"]     = y_test.tolist()
                    st.session_state["trained_task"]       = task_type
                    st.session_state["trained_model_name"] = model_name
                    st.session_state["trained_target"]     = target
                    st.session_state["trained_features"]   = X.columns.tolist()
                    st.session_state["trained_pca_n"]      = pca_n_after
                    st.session_state["scaler"]             = scaler
                    st.session_state["pca"]                = pca

            # -------- Results --------
            if "trained_model" in st.session_state:
                preds      = np.array(st.session_state["trained_preds"])
                y_test_arr = np.array(st.session_state["trained_y_test"])
                task       = st.session_state["trained_task"]
                mdl_name   = st.session_state["trained_model_name"]
                pca_n      = st.session_state["trained_pca_n"]
 
                if pca_n:
                    st.info(f"PCA applied → reduced to {pca_n} features (90% variance retained)")
                st.markdown("### Model Performance")
 
                if task == "Regression":
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    r2   = r2_score(y_test_arr, preds)
                    mse  = mean_squared_error(y_test_arr, preds)
                    rmse = np.sqrt(mse)
                    mae  = mean_absolute_error(y_test_arr, preds)
 
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("R² Score", round(r2, 4))
                    col2.metric("MAE",      round(mae, 4))
                    col3.metric("MSE",      round(mse, 4))
                    col4.metric("RMSE",     round(rmse, 4))
                else:
                    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                                 classification_report)
                    acc = accuracy_score(y_test_arr, preds)
                    st.metric("Accuracy", f"{round(acc * 100, 2)}%")
 
                    # Classification Report
                    st.markdown("### Classification Report")
                    report = classification_report(y_test_arr, preds, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose().round(2)
                    st.dataframe(report_df, use_container_width=True)
 
                    # Confusion Matrix
                    st.markdown("### Confusion Matrix")
                    cm = confusion_matrix(y_test_arr, preds)
                    n_classes = cm.shape[0]
                    fig_size  = max(3, min(6, n_classes * 1.2))  
                    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                    fig.patch.set_facecolor("#0d1117")
                    ax.set_facecolor("#0d1117")
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                linewidths=0.5, linecolor="#151c28",
                                annot_kws={"size": 10})
                    ax.set_xlabel("Predicted", color="#64748b", fontsize=9)
                    ax.set_ylabel("Actual",    color="#64748b", fontsize=9)
                    ax.tick_params(colors="#64748b", labelsize=8)
                    for s in ax.spines.values(): s.set_visible(False)
                    plt.tight_layout()
                    
                    _, mid, _ = st.columns([1, 2, 1])
                    with mid:
                        st.pyplot(fig)
                    plt.close()
 
                st.success(f"✅ {mdl_name} trained successfully!")
                if "train_time" in st.session_state:
                    st.info(f"⏱ Training Time: {round(st.session_state['train_time'], 2)} seconds")
 
                # -------- Download Model .pkl --------
                model_bundle = {
                    "model"   : st.session_state["trained_model"],
                    "scaler"  : st.session_state.get("scaler"),
                    "pca"     : st.session_state.get("pca"),
                    "features": st.session_state["trained_features"],
                    "target"  : st.session_state["trained_target"],
                    "task"    : st.session_state["trained_task"],
                }
 
                buffer = io.BytesIO()
                pickle.dump(model_bundle, buffer)
                buffer.seek(0)
 
                st.download_button(
                    label="⬇ Download Model (.pkl)",
                    data=buffer,
                    file_name=f"{mdl_name.replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream"
                )

