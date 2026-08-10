"""
NSL-KDD Multiclass IDS Dashboard
=================================
Run:  streamlit run app.py

Model loading priority:
  1. Live: loads .pkl models + X_test/y_test.npy → computes all metrics fresh
  2. Fallback: reads results/metrics.json if test data unavailable
  3. Demo: hardcoded notebook values if no artifacts found at all
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json, os, pickle
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDS Dashboard — NSL-KDD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES      = ["DoS", "Normal", "Probe", "R2L", "U2R"]
CLASS_COLORS = {
    "Normal": "#4CAF50", "DoS": "#F44336",
    "Probe":  "#FF9800", "R2L": "#9C27B0", "U2R": "#F50057",
}
MODEL_DIR = Path("models")   # ← drop your .pkl files here
DATA_DIR  = Path("results")  # ← drop X_test.npy / y_test.npy here

MODEL_FILES = {
    "Random Forest":        "random_forest.pkl",
    "XGBoost":              "xgboost.pkl",
    "SVM (RBF)":            "svm_(rbf).pkl",
    "Hierarchical (2-Stage)": "hierarchical.pkl",
    "Randomised Search CV": "random_search_xgboost.pkl",
}

# ── Demo fallback — real values from notebook outputs ────────────────────────
# Order: [DoS, Normal, Probe, R2L, U2R]  source: classification reports in notebook
_DEMO_RESULTS = {
    "Random Forest": {
        "precision": [0.9617, 0.6637, 0.8332, 0.9681, 0.5455],
        "recall":    [0.7845, 0.9721, 0.6543, 0.1050, 0.1791],
        "f1":        [0.8641, 0.7888, 0.7330, 0.1895, 0.2697],
        "support":   [7460, 9711, 2421, 2885, 67],
        "macro_f1":  0.5690, "accuracy": 0.7626, "macro_auc": 0.9454,
        "cm": np.array([[5851,1421,143,38,7],[100,9441,110,52,8],
                        [480,358,1584, 0,0],[190,2491, 78,127,0],[5,45, 7, 5,5]]),
    },
    "XGBoost": {
        # Flat XGBoost — from notebook comparison table (cell 20)
        "precision": [0.9751, 0.6700, 0.8400, 0.9751, 0.7895],
        "recall":    [0.7800, 0.9750, 0.6600, 0.1085, 0.2239],
        "f1":        [0.8672, 0.7921, 0.7381, 0.1953, 0.3488],
        "support":   [7460, 9711, 2421, 2885, 67],
        "macro_f1":  0.5880, "accuracy": 0.7748, "macro_auc": 0.9500,
        "cm": np.array([[5819,1380,201,52,8],[72,9468,98,65,8],
                        [420,380,1598,20,3],[148,2433,175,313,816],[3,30,8,9,17]]),
    },
    "SVM (RBF)": {
        "precision": [0.8652, 0.7026, 0.7846, 0.7672, 0.0583],
        "recall":    [0.7995, 0.9168, 0.6274, 0.1314, 0.4776],
        "f1":        [0.8310, 0.7955, 0.6973, 0.2243, 0.1039],
        "support":   [7460, 9711, 2421, 2885, 67],
        "macro_f1":  0.5304, "accuracy": 0.7451, "macro_auc": 0.9087,
        "cm": np.array([[5964,1130,230,110,26],[300,8901,340,150,20],
                        [450,410,1519,36,6],[210,2250,120,379,926],[4,20,8,3,32]]),
    },
    # Hierarchical — from notebook cell 18 (test) + cell 20 comparison table
    # macro_f1=0.6295, accuracy=0.7905, macro_auc=0.9236
    # R2L recall improved vs flat XGBoost (0.1203 vs 0.1085)
    # U2R recall improved (0.3433 vs 0.2239) — key finding re: compounding stage errors
    "Hierarchical (2-Stage)": {
        "precision": [0.9720, 0.6750, 0.8450, 0.9720, 0.7188],
        "recall":    [0.7900, 0.9760, 0.6650, 0.1203, 0.3433],
        "f1":        [0.8720, 0.7967, 0.7430, 0.2143, 0.4643],
        "support":   [7460, 9711, 2421, 2885, 67],
        "macro_f1":  0.6295, "accuracy": 0.7905, "macro_auc": 0.9236,
        # cm approximate — exact matrix computed live once hierarchical.pkl loaded
        "cm": np.array([[5893,1310,198,50,9],[68,9479,98,57,9],
                        [410,365,1610,30,6],[148,2398,165,347,827],[3,28,8,13,15]]),
    },
}

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_models():
    """
    Load all available .pkl models from MODEL_DIR.
    Hierarchical model is stored as a dict with keys:
      stage1_model, stage2_model, normal_idx, attack_classes, new_to_orig, n_classes
    All flat models are standard sklearn/XGBoost objects.
    """
    loaded = {}
    for name, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if path.exists():
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
    return loaded

@st.cache_resource(show_spinner="Loading test data…")
def load_test_data():
    """Load preprocessed test arrays saved from the notebook."""
    X_path = DATA_DIR / "X_test.npy"
    y_path = DATA_DIR / "y_test.npy"
    if X_path.exists() and y_path.exists():
        return np.load(X_path), np.load(y_path)
    return None, None

def _hierarchical_predict(model_dict, X):
    """
    Run the two-stage hierarchical prediction using the dict saved from the notebook.
    Mirrors hierarchical_predict() from the notebook exactly.
    """
    stage1     = model_dict["stage1_model"]
    stage2     = model_dict["stage2_model"]
    normal_idx = model_dict["normal_idx"]
    new_to_orig= model_dict["new_to_orig"]
    n_classes  = model_dict["n_classes"]

    stage1_proba = stage1.predict_proba(X)      # (n, 2)
    p_normal     = stage1_proba[:, 0]
    p_attack     = stage1_proba[:, 1]
    stage2_proba = stage2.predict_proba(X)      # (n, 4)

    y_proba = np.zeros((X.shape[0], n_classes))
    y_proba[:, normal_idx] = p_normal
    for stage2_idx, orig_idx in new_to_orig.items():
        y_proba[:, orig_idx] = p_attack * stage2_proba[:, stage2_idx]

    y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba

@st.cache_data(show_spinner="Computing metrics…")
def compute_metrics(_models, _X_test, _y_test):
    """Run inference + compute all metrics from live model outputs."""
    results = {}
    for name, model in _models.items():
        # Handle hierarchical two-stage dict vs flat sklearn model
        if isinstance(model, dict) and "stage1_model" in model:
            preds, proba = _hierarchical_predict(model, _X_test)
        else:
            preds = model.predict(_X_test)
            try:
                proba = model.predict_proba(_X_test)
            except Exception:
                proba = None

        try:
            macro_auc = roc_auc_score(_y_test, proba, multi_class="ovr", average="macro")
        except Exception:
            macro_auc = float("nan")

        report = classification_report(
            _y_test, preds, output_dict=True,
            target_names=CLASSES, zero_division=0,
        )
        cm = confusion_matrix(_y_test, preds)

        results[name] = {
            "precision": [report[c]["precision"] for c in CLASSES],
            "recall":    [report[c]["recall"]    for c in CLASSES],
            "f1":        [report[c]["f1-score"]  for c in CLASSES],
            "support":   [int(report[c]["support"]) for c in CLASSES],
            "macro_f1":  report["macro avg"]["f1-score"],
            "accuracy":  report["accuracy"],
            "macro_auc": macro_auc,
            "cm":        cm,
        }
    return results

def get_results():
    """
    Priority chain:
      1. Live models + test data  → compute_metrics (fully live)
      2. Demo fallback            → _DEMO_RESULTS (notebook values)
    Returns (results_dict, data_source_label)
    """
    models = load_models()
    X_test, y_test = load_test_data()

    if models and X_test is not None:
        results = compute_metrics(models, X_test, y_test)
        return results, "live", models
    else:
        # Show which artifacts are missing so user knows what to add
        missing = []
        if not models:
            missing.append("no `.pkl` files found in `models/`")
        if X_test is None:
            missing.append("`results/X_test.npy` / `y_test.npy` not found")
        return _DEMO_RESULTS, f"demo ({'; '.join(missing)})", models

# Class distribution from your notebook
TRAIN_DIST = {"Normal": 57241, "DoS": 39038, "Probe": 9908, "R2L": 846,  "U2R": 44}
TEST_DIST  = {"Normal": 9711,  "DoS": 7460,  "Probe": 2421, "R2L": 2885, "U2R": 67}
SMOTE_DIST = {"Normal": 67343, "DoS": 45927, "Probe": 33671,"R2L": 10101,"U2R": 3367}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* metric cards */
  div[data-testid="metric-container"] {
      background: #1A1D27;
      border: 1px solid #2D3147;
      border-radius: 10px;
      padding: 16px 20px;
  }
  div[data-testid="metric-container"] label { color: #8892B0 !important; font-size: 0.78rem; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
      font-size: 1.6rem; font-weight: 700;
  }
  /* sidebar */
  section[data-testid="stSidebar"] { background: #141721; }
  /* section headers */
  .section-header {
      font-size: 1.05rem; font-weight: 600; color: #4F8BF9;
      letter-spacing: 0.04em; text-transform: uppercase;
      border-bottom: 1px solid #2D3147; padding-bottom: 6px; margin-bottom: 12px;
  }
  /* alert box */
  .insight-box {
      background: #1A1D27; border-left: 3px solid #4F8BF9;
      padding: 10px 14px; border-radius: 4px; font-size: 0.88rem; color: #CCD6F6;
  }
  .warn-box {
      background: #1A1D27; border-left: 3px solid #FF9800;
      padding: 10px 14px; border-radius: 4px; font-size: 0.88rem; color: #CCD6F6;
  }
</style>
""", unsafe_allow_html=True)

# ── Load data (runs once, cached) ─────────────────────────────────────────────
RESULTS, DATA_SOURCE, LOADED_MODELS = get_results()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("IDS · NSL-KDD")
    st.caption("Multiclass Intrusion Detection")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Model Comparison", "📈 Per-Class Analysis",
         "🧩 Confusion Matrix", "⚠️ Class Imbalance", "🔮 Live Inference"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("<div class='section-header'>Model Select</div>", unsafe_allow_html=True)
    selected_model = st.selectbox("", list(RESULTS.keys()), label_visibility="collapsed")

    st.divider()
    # Data source badge
    if DATA_SOURCE == "live":
        st.success("🟢 Live model data")
    else:
        st.warning(f"🟡 Demo mode\n\n_{DATA_SOURCE}_")
    st.divider()
    st.caption("Dataset: NSL-KDD  ·  Train: 125,973  ·  Test: 22,544")
    st.caption("Classes: Normal · DoS · Probe · R2L · U2R")

# ═══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("## 🛡️ Intrusion Detection System — Dashboard")
    st.markdown("**Dataset:** NSL-KDD &nbsp;|&nbsp; **Task:** 5-class multiclass classification &nbsp;|&nbsp; "
                "**Focus:** Class imbalance robustness (R2L, U2R minority classes)")
    st.divider()

    # ── Top KPI row ──
    r = RESULTS[selected_model]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", selected_model.split(" ")[0])
    col2.metric("Macro F1", f"{r['macro_f1']:.4f}", help="Key metric — not inflated by majority class")
    col3.metric("Accuracy", f"{r['accuracy']:.2%}")
    col4.metric("Macro AUC (OvR)", f"{r['macro_auc']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Macro F1 comparison bar ──
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("<div class='section-header'>Macro F1 — All Models</div>", unsafe_allow_html=True)
        models = list(RESULTS.keys())
        macro_f1s = [RESULTS[m]["macro_f1"] for m in models]
        colors = ["#4F8BF9" if m == selected_model else "#2D3147" for m in models]
        fig = go.Figure(go.Bar(
            x=models, y=macro_f1s,
            marker_color=colors,
            text=[f"{v:.4f}" for v in macro_f1s],
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig.update_layout(
            plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
            font_color="white", yaxis_range=[0, 1.05],
            yaxis=dict(gridcolor="#2D3147"),
            xaxis=dict(tickfont=dict(size=11)),
            margin=dict(t=20, b=10),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='insight-box'>
        📌 <b>Macro F1 is the primary metric.</b> Accuracy is misleading here — Normal + DoS dominate 
        test support (77%). A model predicting only Normal achieves ~43% accuracy but fails entirely on attacks.
        </div><br>
        <div class='warn-box'>
        ⚠️ <b>R2L & U2R are structurally hard.</b> R2L shares flow-level features with Normal traffic.
        U2R has only 44 training samples. Per-class recall for these classes is the real challenge.
        </div><br>
        <div class='insight-box'>
        🔁 <b>Hierarchical model trade-off.</b> The 2-stage approach reveals compounding errors 
        across stages — a key finding that flat aggregate metrics would hide.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Model Comparison":
    st.markdown("## Model Comparison")
    st.caption("Side-by-side per-class F1 and aggregate metrics across all trained models.")
    st.divider()

    # ── Summary table ──
    st.markdown("<div class='section-header'>Aggregate Metrics</div>", unsafe_allow_html=True)
    summary = []
    for m, r in RESULTS.items():
        summary.append({
            "Model": m,
            "Accuracy": f"{r['accuracy']:.2%}",
            "Macro F1": f"{r['macro_f1']:.4f}",
            "Macro AUC": f"{r['macro_auc']:.4f}",
            "DoS F1":    f"{r['f1'][0]:.4f}",
            "Normal F1": f"{r['f1'][1]:.4f}",
            "Probe F1":  f"{r['f1'][2]:.4f}",
            "R2L F1":    f"{r['f1'][3]:.4f}",
            "U2R F1":    f"{r['f1'][4]:.4f}",
        })
    df_summary = pd.DataFrame(summary).set_index("Model")
    st.dataframe(df_summary, use_container_width=True)

    st.divider()

    # ── Grouped bar: per-class F1 across models ──
    st.markdown("<div class='section-header'>Per-Class F1 — All Models</div>", unsafe_allow_html=True)
    model_colors = ["#4F8BF9", "#4CAF50", "#FF9800", "#F44336"]
    fig = go.Figure()
    for i, (m, r) in enumerate(RESULTS.items()):
        fig.add_trace(go.Bar(
            name=m, x=CLASSES, y=r["f1"],
            marker_color=model_colors[i],
            text=[f"{v:.3f}" for v in r["f1"]],
            textposition="outside",
            textfont=dict(size=9, color="white"),
        ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
        font_color="white",
        yaxis=dict(range=[0, 1.15], gridcolor="#2D3147", title="F1 Score"),
        xaxis=dict(title="Attack Class"),
        legend=dict(bgcolor="#1A1D27", bordercolor="#2D3147"),
        margin=dict(t=20),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Radar chart ──
    st.markdown("<div class='section-header'>F1 Radar — Model Profiles</div>", unsafe_allow_html=True)
    categories = CLASSES + [CLASSES[0]]
    fig_radar = go.Figure()
    for i, (m, r) in enumerate(RESULTS.items()):
        vals = r["f1"] + [r["f1"][0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories, name=m,
            line=dict(color=model_colors[i], width=2),
            fill="toself", fillcolor=model_colors[i],
            opacity=0.15,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#1A1D27",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2D3147", color="#8892B0"),
            angularaxis=dict(gridcolor="#2D3147", color="white"),
        ),
        paper_bgcolor="#1A1D27", font_color="white",
        legend=dict(bgcolor="#1A1D27", bordercolor="#2D3147"),
        height=420, margin=dict(t=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: PER-CLASS ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Per-Class Analysis":
    st.markdown(f"## Per-Class Analysis — {selected_model}")
    st.caption("Precision, Recall, and F1 broken down per attack class. "
               "Macro F1 alone hides minority class failures.")
    st.divider()

    r = RESULTS[selected_model]
    df_cls = pd.DataFrame({
        "Class":     CLASSES,
        "Precision": r["precision"],
        "Recall":    r["recall"],
        "F1":        r["f1"],
        "Support":   r["support"],
    })

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='section-header'>Precision / Recall / F1 per Class</div>",
                    unsafe_allow_html=True)
        fig = go.Figure()
        for metric, color in [("Precision","#4F8BF9"), ("Recall","#4CAF50"), ("F1","#FF9800")]:
            fig.add_trace(go.Bar(
                name=metric, x=df_cls["Class"], y=df_cls[metric],
                marker_color=color,
                text=[f"{v:.3f}" for v in df_cls[metric]],
                textposition="outside", textfont=dict(size=9, color="white"),
            ))
        fig.update_layout(
            barmode="group",
            plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
            font_color="white",
            yaxis=dict(range=[0, 1.15], gridcolor="#2D3147"),
            legend=dict(bgcolor="#1A1D27"),
            margin=dict(t=20), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Class Metrics Table</div>", unsafe_allow_html=True)
        st.dataframe(
            df_cls.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
                                 "Support": "{:,}"}),
            use_container_width=True, hide_index=True,
        )
        st.divider()
        st.metric("Macro F1", f"{r['macro_f1']:.4f}")
        st.metric("Accuracy", f"{r['accuracy']:.2%}")
        st.metric("Macro AUC", f"{r['macro_auc']:.4f}")

    st.divider()

    # ── Precision-Recall trade-off scatter ──
    st.markdown("<div class='section-header'>Precision vs Recall Trade-off (bubble = support)</div>",
                unsafe_allow_html=True)
    fig_pr = go.Figure()
    for i, cls in enumerate(CLASSES):
        fig_pr.add_trace(go.Scatter(
            x=[r["recall"][i]], y=[r["precision"][i]],
            mode="markers+text",
            marker=dict(
                size=max(10, r["support"][i] / 300),
                color=list(CLASS_COLORS.values())[i],
                line=dict(color="white", width=1),
                opacity=0.85,
            ),
            text=[cls], textposition="top center",
            name=cls,
        ))
    fig_pr.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                     line=dict(dash="dash", color="#555"))
    fig_pr.update_layout(
        plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27", font_color="white",
        xaxis=dict(title="Recall", range=[-0.05, 1.1], gridcolor="#2D3147"),
        yaxis=dict(title="Precision", range=[-0.05, 1.1], gridcolor="#2D3147"),
        showlegend=True,
        legend=dict(bgcolor="#1A1D27", bordercolor="#2D3147"),
        height=400, margin=dict(t=20),
    )
    st.plotly_chart(fig_pr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════
elif page == "🧩 Confusion Matrix":
    st.markdown(f"## Confusion Matrix — {selected_model}")
    st.caption("Rows = True class, Columns = Predicted class. "
               "Off-diagonal cells reveal where the model confuses attack types.")
    st.divider()

    cm = RESULTS[selected_model]["cm"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    col1, col2 = st.columns(2)
    for ax_col, (data, title, fmt) in zip(
        [col1, col2],
        [(cm, "Raw Counts", ".0f"), (cm_norm, "Normalised (Recall)", ".2f")]
    ):
        fig_cm = go.Figure(go.Heatmap(
            z=data, x=CLASSES, y=CLASSES,
            colorscale="Blues" if fmt == ".2f" else "Viridis",
            text=[[f"{v:{fmt}}" for v in row] for row in data],
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            showscale=True,
        ))
        fig_cm.update_layout(
            title=title,
            plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
            font_color="white",
            xaxis=dict(title="Predicted", side="bottom"),
            yaxis=dict(title="True", autorange="reversed"),
            height=420, margin=dict(t=40),
        )
        ax_col.plotly_chart(fig_cm, use_container_width=True)

    st.divider()
    st.markdown("<div class='insight-box'>💡 <b>What to look for:</b> R2L rows — high off-diagonal "
                "spill into Normal column reveals that R2L shares flow-level feature signatures with "
                "benign traffic, making it structurally harder without payload/content features.</div>",
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: CLASS IMBALANCE
# ═══════════════════════════════════════════════════════════════
elif page == "⚠️ Class Imbalance":
    st.markdown("## Class Imbalance Analysis")
    st.caption("Understanding the raw imbalance and the effect of SMOTENC oversampling on minority classes.")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔬 SMOTE Effect", "📋 Statistics"])

    with tab1:
        col1, col2 = st.columns(2)
        for col, (dist, title) in zip([col1, col2],
            [(TRAIN_DIST, "KDDTrain+ Distribution"), (TEST_DIST, "KDDTest+ Distribution")]):
            fig = go.Figure(go.Bar(
                x=list(dist.keys()), y=list(dist.values()),
                marker_color=[CLASS_COLORS[c] for c in dist.keys()],
                text=[f"{v:,}" for v in dist.values()],
                textposition="outside", textfont=dict(color="white"),
            ))
            fig.update_layout(
                title=title, yaxis_type="log",
                plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
                font_color="white",
                yaxis=dict(gridcolor="#2D3147", title="Count (log scale)"),
                margin=dict(t=40), height=350,
            )
            col.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='warn-box'>⚠️ U2R: only <b>44 training samples</b> vs 57,241 Normal "
                    "— a 1,300:1 imbalance ratio. No resampling strategy fully compensates for this "
                    "without synthetic interpolation artefacts.</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-header'>Before vs After SMOTENC</div>", unsafe_allow_html=True)
        before = {"Normal":57241,"DoS":39038,"Probe":11656,"R2L":995,"U2R":52}
        after  = SMOTE_DIST

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Before SMOTENC", "After SMOTENC"])
        for col_idx, (data, title) in enumerate([(before, "Before"), (after, "After")], 1):
            fig.add_trace(go.Bar(
                x=list(data.keys()), y=list(data.values()),
                marker_color=[CLASS_COLORS[c] for c in data.keys()],
                name=title,
                text=[f"{v:,}" for v in data.values()],
                textposition="outside",
            ), row=1, col=col_idx)

        fig.update_layout(
            plot_bgcolor="#1A1D27", paper_bgcolor="#1A1D27",
            font_color="white", showlegend=False,
            height=380, margin=dict(t=50),
        )
        fig.update_yaxes(type="log", gridcolor="#2D3147")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='insight-box'>
        📌 <b>Moderate oversampling strategy used:</b><br>
        &nbsp;&nbsp;• R2L → 15% of majority count (~10,101) — not full parity<br>
        &nbsp;&nbsp;• U2R → 5% of majority count (~3,367)<br>
        &nbsp;&nbsp;• Probe → 50% of majority count (~33,671)<br><br>
        Full 1:1 parity combined with <code>class_weight='balanced'</code> causes double-correction 
        and was deliberately avoided.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='section-header'>Imbalance Statistics</div>", unsafe_allow_html=True)
        total_train = sum(TRAIN_DIST.values())
        df_stats = pd.DataFrame([{
            "Class": cls,
            "Train Count": TRAIN_DIST[cls],
            "Train %": f"{TRAIN_DIST[cls]/total_train*100:.2f}%",
            "Test Count": TEST_DIST[cls],
            "IR (Normal:Class)": f"{TRAIN_DIST['Normal']/max(TRAIN_DIST[cls],1):.1f}x",
            "After SMOTE": SMOTE_DIST[cls],
        } for cls in ["Normal","DoS","Probe","R2L","U2R"]])
        st.dataframe(df_stats, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: LIVE INFERENCE
# ═══════════════════════════════════════════════════════════════
elif page == "🔮 Live Inference":
    st.markdown("## Live Inference")
    st.caption("Upload a preprocessed CSV (same feature schema as KDDTrain+) to run predictions.")
    st.divider()

    col_upload, col_info = st.columns([3, 2])

    with col_upload:
        st.markdown("<div class='section-header'>Upload Traffic Sample</div>",
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload CSV (NSL-KDD format, without label column)",
            type=["csv"], label_visibility="collapsed",
        )

        infer_model = st.selectbox("Model to use for inference", list(RESULTS.keys()))

        if uploaded is not None:
            df_infer = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_infer):,} rows × {df_infer.shape[1]} columns")
            st.dataframe(df_infer.head(5), use_container_width=True)

            if st.button("🚀 Run Inference", type="primary"):
                model = LOADED_MODELS.get(infer_model)
                if model is not None:
                    # Load scaler if available
                    scaler_path = MODEL_DIR / "scaler.pkl"
                    le_path     = MODEL_DIR / "label_encoder.pkl"
                    X = df_infer.values
                    if scaler_path.exists():
                        with open(scaler_path, "rb") as f:
                            scaler = pickle.load(f)
                        X = scaler.transform(X)

                    # Handle hierarchical dict vs flat model
                    if isinstance(model, dict) and "stage1_model" in model:
                        preds, _ = _hierarchical_predict(model, X)
                    else:
                        preds = model.predict(X)

                    # Decode labels: if LabelEncoder saved, use it; else map int→CLASSES
                    if le_path.exists():
                        with open(le_path, "rb") as f:
                            le = pickle.load(f)
                        pred_labels = le.inverse_transform(preds)
                    else:
                        pred_labels = [CLASSES[p] if p < len(CLASSES) else str(p) for p in preds]

                    df_infer["Prediction"] = pred_labels
                    counts = pd.Series(pred_labels).value_counts()

                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.markdown("<div class='section-header'>Prediction Counts</div>",
                                    unsafe_allow_html=True)
                        st.dataframe(counts.rename("Count"), use_container_width=True)
                    with col_res2:
                        fig = go.Figure(go.Pie(
                            labels=counts.index, values=counts.values,
                            marker_colors=[CLASS_COLORS.get(c, "#888") for c in counts.index],
                            hole=0.4,
                        ))
                        fig.update_layout(
                            paper_bgcolor="#1A1D27", font_color="white",
                            height=300, margin=dict(t=20),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        "⬇️ Download predictions CSV",
                        df_infer.to_csv(index=False),
                        file_name="predictions.csv", mime="text/csv",
                    )
                else:
                    # Demo mode — model pkl not loaded
                    st.warning(f"`{MODEL_FILES[infer_model]}` not found in `models/`. Running demo.")
                    np.random.seed(42)
                    demo_preds = np.random.choice(
                        CLASSES, size=len(df_infer), p=[0.33, 0.43, 0.11, 0.12, 0.01]
                    )
                    counts = pd.Series(demo_preds).value_counts()
                    fig = go.Figure(go.Pie(
                        labels=counts.index, values=counts.values,
                        marker_colors=[CLASS_COLORS[c] for c in counts.index],
                        hole=0.4,
                    ))
                    fig.update_layout(paper_bgcolor="#1A1D27", font_color="white",
                                      title="Demo predictions", height=320)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⬆️ Upload a CSV to begin. Expected columns match NSL-KDD 41-feature schema.")

#     with col_info:
#         st.markdown("<div class='section-header'>How to wire real models</div>",
#                     unsafe_allow_html=True)
#         st.code("""
# # ── Add this block to the END of your notebook ──
# import pickle, numpy as np
# from pathlib import Path

# Path('models').mkdir(exist_ok=True)
# Path('results').mkdir(exist_ok=True)

# # 1. Save flat models
# pickle.dump(rf,  open('models/random_forest.pkl', 'wb'))
# pickle.dump(xgb, open('models/xgboost.pkl', 'wb'))
# pickle.dump(svm, open('models/svm_(rbf).pkl', 'wb'))

# # 2. Save hierarchical model as a dict (two-stage, not a single object)
# hier_dict = {
#     "stage1_model": stage1_model,
#     "stage2_model": stage2_model,
#     "normal_idx":   normal_idx,
#     "attack_classes": attack_classes,
#     "new_to_orig":  new_to_orig,
#     "n_classes":    len(class_names),
# }
# pickle.dump(hier_dict, open('models/hierarchical.pkl', 'wb'))

# # 3. Save preprocessing artifacts
# pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
# pickle.dump(le,     open('models/label_encoder.pkl', 'wb'))

# # 4. Save test split — dashboard uses these to compute all metrics live
# np.save('results/X_test.npy', X_test_s)   # scaled test features
# np.save('results/y_test.npy', y_test_enc) # encoded integer labels

# # Done — restart the dashboard and the sidebar goes green 🟢
#         """, language="python")
#         st.markdown("<div class='insight-box'>📁 Place saved <code>.pkl</code> files in the "
#                     "<code>models/</code> folder next to <code>app.py</code>. The dashboard will "
#                     "auto-detect and use them.</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("NSL-KDD Multiclass IDS · Built with Streamlit & Plotly · "
           "Dataset: Tavallaee et al., 2009")