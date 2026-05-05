import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import io
import scipy.stats as stats_module
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geopolymer Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #162032);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card .metric-label {
        font-size: 12px;
        color: #8b949e;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-card .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-card .metric-sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Result banner */
    .result-banner {
        background: linear-gradient(135deg, #0d4f2e, #0a3d22);
        border: 2px solid #238636;
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
        margin: 20px 0;
    }
    .result-banner .result-value {
        font-size: 52px;
        font-weight: 800;
        color: #3fb950;
    }
    .result-banner .result-label {
        font-size: 16px;
        color: #58a6ff;
        font-weight: 500;
        margin-top: 8px;
    }
    .result-banner .result-grade {
        font-size: 22px;
        font-weight: 700;
        color: #f0b429;
        margin-top: 10px;
    }

    /* Inverse result */
    .inv-card {
        background: linear-gradient(135deg, #1e2a1e, #162016);
        border: 1px solid #238636;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .inv-card-title {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .inv-card-value {
        font-size: 20px;
        font-weight: 700;
        color: #3fb950;
    }

    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 2px solid #21262d;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d;
        color: #58a6ff;
        border-radius: 6px;
    }

    /* Dataframe */
    .dataframe { font-size: 13px; }

    /* Inputs */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #c9d1d9;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636, #1a7a2d);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 15px;
        padding: 12px 28px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043, #238636);
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(35, 134, 54, 0.4);
    }

    /* Info boxes */
    .info-box {
        background: #1e2a3a;
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 13px;
        color: #c9d1d9;
    }
    .warn-box {
        background: #2a1e0a;
        border-left: 4px solid #f0b429;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 13px;
        color: #c9d1d9;
    }
    h1, h2, h3 { color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.alpha": 0.5,
    "lines.color": "#58a6ff",
    "patch.edgecolor": "#30363d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "figure.dpi": 110,
})

# ─── Data Loading & Preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    try:
        df = pd.read_excel("excel1.xlsx")
    except Exception:
        # Fallback synthetic data if excel not present
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "Source_material": np.random.choice(["F. A", "GGBS", "F. A + GGBS", "Rice Husk Ash"], n),
            "Alkali_solution": np.random.choice(["SS/SH", "SS/ KOH", "SS/SH/25%KOH"], n),
            "Molarity": np.random.choice([8, 10, 12, 14, 16], n),
            "SS_SH_KOH_ratio": np.random.uniform(1.0, 3.5, n).round(2),
            "A_B_ratio": np.random.uniform(0.1, 0.6, n).round(2),
            "Curing_temp": np.random.choice([40.0, 60.0, 80.0, 90.0], n),
            "Compression_28d": np.random.uniform(15, 80, n).round(2),
        })

    # Clean
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.replace(["--", "-", "—", "", "nan", "NaN"], np.nan)

    def fix_curing(val):
        if isinstance(val, str) and "-" in val:
            parts = val.split("-")
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return np.nan
        return val

    df["Curing_temp"] = df["Curing_temp"].apply(fix_curing)
    num_cols = ["Molarity", "SS_SH_KOH_ratio", "A_B_ratio", "Curing_temp", "Compression_28d"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Compression_28d"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def train_models(df_hash):
    df = load_and_preprocess()
    TARGET = "Compression_28d"
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].copy()

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols   = X.select_dtypes(exclude="object").columns.tolist()

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    label_encoders = {}
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "SVR (RBF)":         SVR(kernel="rbf"),
        "Random Forest":     RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    trained  = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        results[name] = {
            "R²":   round(r2_score(y_test, preds), 4),
            "MAE":  round(mean_absolute_error(y_test, preds), 3),
            "RMSE": round(mean_squared_error(y_test, preds) ** 0.5, 3),
        }
        trained[name] = m

    return (trained, scaler, label_encoders, num_imputer, cat_imputer,
            feature_names, categorical_cols, numerical_cols,
            X_train, X_test, y_train, y_test,
            pd.DataFrame(results).T, X_scaled, y, df)

# ─── Helper: preprocess single input ──────────────────────────────────────────
def preprocess_input(row_dict, scaler, label_encoders, feature_names,
                     categorical_cols, numerical_cols, num_imputer):
    row = pd.DataFrame([row_dict])
    for col in categorical_cols:
        row[col] = row[col].astype(str)
        le = label_encoders[col]
        val = row[col].values[0]
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = le.transform([val])

    for col in numerical_cols:
        row[col] = pd.to_numeric(row[col], errors="coerce")

    row = row[feature_names]
    row_scaled = scaler.transform(row)
    return row_scaled

# ─── Grade classifier ──────────────────────────────────────────────────────────
def get_grade(strength):
    if strength < 20:   return "M15 (Low)",  "#e74c3c"
    elif strength < 25: return "M20 (Standard)", "#f39c12"
    elif strength < 30: return "M25 (Standard)", "#f0b429"
    elif strength < 40: return "M30-M35 (Moderate)", "#2ecc71"
    elif strength < 55: return "M40-M50 (High)", "#27ae60"
    else:               return "M55+ (Very High)", "#1abc9c"

# ─── Inverse prediction ───────────────────────────────────────────────────────
def inverse_predict(target_strength, model, scaler, label_encoders,
                    feature_names, categorical_cols, numerical_cols,
                    df, num_imputer):

    cat_cols_vals = {c: df[c].dropna().unique().tolist() for c in categorical_cols}
    num_ranges = {c: (df[c].dropna().min(), df[c].dropna().max())
                  for c in numerical_cols}

    best_combo = None
    best_diff  = np.inf

    # Grid over categorical combos + optimise numerical
    from itertools import product as iproduct
    cat_combos = list(iproduct(*[cat_cols_vals[c] for c in categorical_cols]))

    for combo in cat_combos:
        cat_dict = dict(zip(categorical_cols, combo))

        def objective(num_vals):
            num_dict = dict(zip(numerical_cols, num_vals))
            row_dict = {**cat_dict, **num_dict}
            try:
                x = preprocess_input(row_dict, scaler, label_encoders,
                                     feature_names, categorical_cols,
                                     numerical_cols, num_imputer)
                pred = model.predict(x)[0]
                return (pred - target_strength) ** 2
            except:
                return 1e9

        x0 = np.array([np.mean(num_ranges[c]) for c in numerical_cols])
        bounds = [num_ranges[c] for c in numerical_cols]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})

        if res.fun < best_diff:
            best_diff  = res.fun
            num_dict   = dict(zip(numerical_cols, res.x))
            best_combo = {**cat_dict, **num_dict}

    if best_combo is None:
        return None, None

    row_dict = best_combo
    x_best = preprocess_input(row_dict, scaler, label_encoders,
                               feature_names, categorical_cols,
                               numerical_cols, num_imputer)
    achieved = model.predict(x_best)[0]
    return best_combo, achieved

# ─── Load everything ──────────────────────────────────────────────────────────
df_raw = load_and_preprocess()
(trained_models, scaler, label_encoders, num_imputer, cat_imputer,
 feature_names, categorical_cols, numerical_cols,
 X_train, X_test, y_train, y_test,
 results_df, X_scaled, y_full, df) = train_models(str(df_raw.shape))

PALETTE = ["#58a6ff", "#3fb950", "#f0b429", "#ff7b72", "#d2a8ff", "#79c0ff"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Geopolymer Predictor")
    st.markdown("---")

    mode = st.radio(
        "**Select Mode**",
        ["🔮 Predict Strength", "🎯 Target Strength → Mix Design"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Selection")
    selected_model_name = st.selectbox(
        "Choose ML Model",
        list(trained_models.keys()),
        index=3,
        help="Random Forest & Gradient Boosting usually perform best"
    )
    selected_model = trained_models[selected_model_name]
    model_score = results_df.loc[selected_model_name, "R²"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model R² Score</div>
        <div class="metric-value">{model_score:.4f}</div>
        <div class="metric-sub">on 20% test split</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.markdown(f"""
    <div class="info-box">
        📦 <b>{len(df)}</b> samples loaded<br>
        🔢 <b>6</b> input features<br>
        🎯 Target: 28-day strength (MPa)
    </div>
    """, unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
    <h1 style="font-size:36px; font-weight:800; color:#58a6ff; margin:0;">
        🏗️ Geopolymer Concrete Strength Predictor
    </h1>
    <p style="color:#8b949e; font-size:15px; margin-top:6px;">
        ML-powered prediction • Model benchmarking • Mix design optimizer
    </p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict / Design",
    "📊 EDA & Visualizations",
    "🤖 Model Performance",
    "📂 Dataset"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT / DESIGN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── PREDICT STRENGTH MODE ─────────────────────────────────────────────────
    if mode == "🔮 Predict Strength":
        st.markdown('<div class="section-header">🔮 Input Mix Parameters</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1], gap="large")

        with col_a:
            st.markdown("**Categorical Parameters**")
            source_opts = sorted(df["Source_material"].dropna().unique().tolist())
            alkali_opts = sorted(df["Alkali_solution"].dropna().unique().tolist())

            source_material = st.selectbox("🧱 Source Material", source_opts,
                help="Fly Ash (F.A), GGBS, or blends")
            alkali_solution = st.selectbox("🧪 Alkali Solution", alkali_opts,
                help="Type of activator solution used")

        with col_b:
            st.markdown("**Numerical Parameters**")
            molarity = st.slider(
                "⚗️ Molarity (M)", 
                int(df["Molarity"].min()), int(df["Molarity"].max()),
                int(df["Molarity"].median()),
                help="NaOH concentration in mol/L"
            )
            ss_sh = st.slider(
                "🔬 SS/SH or KOH Ratio",
                float(df["SS_SH_KOH_ratio"].min()), float(df["SS_SH_KOH_ratio"].max()),
                float(df["SS_SH_KOH_ratio"].median()), step=0.1,
                help="Sodium silicate to sodium hydroxide ratio"
            )
            ab_ratio = st.slider(
                "📐 Alkali/Binder Ratio",
                float(df["A_B_ratio"].min()), float(df["A_B_ratio"].max()),
                float(df["A_B_ratio"].median()), step=0.01,
                help="Ratio of alkali solution to binder"
            )
            curing_temp = st.slider(
                "🌡️ Curing Temperature (°C)",
                float(df["Curing_temp"].min()), float(df["Curing_temp"].max()),
                float(df["Curing_temp"].median()), step=5.0,
                help="Temperature during curing"
            )

        st.markdown("---")
        predict_btn = st.button("🚀 Predict 28-Day Compressive Strength")

        if predict_btn:
            row_dict = {
                "Source_material": source_material,
                "Alkali_solution":  alkali_solution,
                "Molarity":         molarity,
                "SS_SH_KOH_ratio":  ss_sh,
                "A_B_ratio":        ab_ratio,
                "Curing_temp":      curing_temp,
            }
            x_in = preprocess_input(row_dict, scaler, label_encoders, feature_names,
                                    categorical_cols, numerical_cols, num_imputer)
            pred_val = selected_model.predict(x_in)[0]
            grade, grade_color = get_grade(pred_val)

            # Result banner
            st.markdown(f"""
            <div class="result-banner">
                <div class="result-value">{pred_val:.2f} MPa</div>
                <div class="result-label">Predicted 28-Day Compressive Strength</div>
                <div class="result-grade" style="color:{grade_color}">Grade: {grade}</div>
            </div>
            """, unsafe_allow_html=True)

            # Metric row
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""<div class="metric-card">
                <div class="metric-label">Model Used</div>
                <div class="metric-value" style="font-size:18px;">{selected_model_name}</div>
            </div>""", unsafe_allow_html=True)
            m2.markdown(f"""<div class="metric-card">
                <div class="metric-label">Model R²</div>
                <div class="metric-value">{model_score:.4f}</div>
            </div>""", unsafe_allow_html=True)
            m3.markdown(f"""<div class="metric-card">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{results_df.loc[selected_model_name,'MAE']}</div>
                <div class="metric-sub">MPa</div>
            </div>""", unsafe_allow_html=True)
            m4.markdown(f"""<div class="metric-card">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{results_df.loc[selected_model_name,'RMSE']}</div>
                <div class="metric-sub">MPa</div>
            </div>""", unsafe_allow_html=True)

            # ── Visualizations for prediction ─────────────────────────────────
            st.markdown('<div class="section-header">📊 Prediction Visualizations</div>', unsafe_allow_html=True)

            vc1, vc2 = st.columns(2)

            # 1) Gauge-like strength bar
            with vc1:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                categories = ["Low\n(<20)", "Standard\n(20-30)", "Moderate\n(30-40)", "High\n(40-55)", "V.High\n(≥55)"]
                thresholds = [20, 30, 40, 55, 90]
                bar_colors = ["#e74c3c", "#f39c12", "#f0b429", "#2ecc71", "#1abc9c"]
                prev = 0
                for cat, thresh, clr in zip(categories, thresholds, bar_colors):
                    ax.barh(0, thresh - prev, left=prev, color=clr, alpha=0.35, height=0.4)
                    prev = thresh
                ax.axvline(pred_val, color="#58a6ff", linewidth=3, linestyle="--", label=f"Predicted: {pred_val:.1f} MPa")
                ax.set_xlim(0, 90)
                ax.set_yticks([])
                ax.set_xlabel("Compressive Strength (MPa)", fontsize=11)
                ax.set_title("Strength Grade Scale", fontsize=12, fontweight="bold")
                ax.legend(loc="upper right", fontsize=10)
                for i, (cat, thresh, clr) in enumerate(zip(categories, thresholds, bar_colors)):
                    mid = ([0]+thresholds)[i] + (thresh - ([0]+thresholds)[i]) / 2
                    ax.text(mid, 0, cat, ha="center", va="center", fontsize=7.5,
                            color="white", fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # 2) Radar — input values normalised
            with vc2:
                num_features = ["Molarity", "SS_SH_KOH_ratio", "A_B_ratio", "Curing_temp"]
                input_vals   = [molarity, ss_sh, ab_ratio, curing_temp]
                df_mins = [df[c].min() for c in num_features]
                df_maxs = [df[c].max() for c in num_features]
                normed  = [(v - mn) / (mx - mn + 1e-9) for v, mn, mx in zip(input_vals, df_mins, df_maxs)]

                angles = np.linspace(0, 2 * np.pi, len(num_features), endpoint=False).tolist()
                normed_plot = normed + [normed[0]]
                angles_plot = angles + [angles[0]]

                fig, ax = plt.subplots(figsize=(5, 3.5), subplot_kw={"polar": True})
                ax.set_facecolor("#161b22")
                fig.patch.set_facecolor("#0f1117")
                ax.plot(angles_plot, normed_plot, color="#58a6ff", linewidth=2)
                ax.fill(angles_plot, normed_plot, color="#58a6ff", alpha=0.25)
                ax.set_xticks(angles)
                ax.set_xticklabels(["Molarity", "SS/SH", "A/B Ratio", "Cure Temp"], fontsize=9)
                ax.set_yticklabels([])
                ax.set_title("Input Parameter Profile\n(Normalised 0–1)", fontsize=11, fontweight="bold", pad=15)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # 3) Sensitivity — how does each feature affect prediction?
            st.markdown("#### 🔍 Sensitivity Analysis — How Each Feature Affects Strength")
            sens_cols = st.columns(len(num_features))
            base_dict = {
                "Source_material": source_material,
                "Alkali_solution":  alkali_solution,
                "Molarity":         molarity,
                "SS_SH_KOH_ratio":  ss_sh,
                "A_B_ratio":        ab_ratio,
                "Curing_temp":      curing_temp,
            }
            feat_map = {"Molarity": "Molarity", "SS_SH_KOH_ratio": "SS_SH_KOH_ratio",
                        "A_B_ratio": "A_B_ratio", "Curing_temp": "Curing_temp"}

            for i, (feat, feat_key) in enumerate(feat_map.items()):
                sweep = np.linspace(df[feat].min(), df[feat].max(), 40)
                preds_s = []
                for v in sweep:
                    d = base_dict.copy()
                    d[feat_key] = v
                    xi = preprocess_input(d, scaler, label_encoders, feature_names,
                                         categorical_cols, numerical_cols, num_imputer)
                    preds_s.append(selected_model.predict(xi)[0])

                fig, ax = plt.subplots(figsize=(3.5, 3))
                ax.plot(sweep, preds_s, color=PALETTE[i], linewidth=2)
                ax.axvline(base_dict[feat_key], color="white", linestyle="--", linewidth=1.2,
                           label=f"Current: {base_dict[feat_key]:.2f}")
                ax.axhline(pred_val, color="#f0b429", linestyle=":", linewidth=1.2)
                ax.set_xlabel(feat, fontsize=9)
                ax.set_ylabel("Strength (MPa)", fontsize=9)
                ax.set_title(f"Effect of {feat}", fontsize=9, fontweight="bold")
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                sens_cols[i].pyplot(fig)
                plt.close()

            # 4) All models comparison bar
            st.markdown("#### 🏆 Prediction Across All Models")
            all_preds = {}
            for mname, mobj in trained_models.items():
                all_preds[mname] = mobj.predict(x_in)[0]

            fig, ax = plt.subplots(figsize=(10, 4))
            names_ = list(all_preds.keys())
            vals_  = list(all_preds.values())
            colors_ = [PALETTE[3] if n == selected_model_name else "#30363d" for n in names_]
            bars_ = ax.bar(names_, vals_, color=colors_, edgecolor="#21262d", linewidth=1.2, width=0.55)
            ax.axhline(pred_val, color="#f0b429", linestyle="--", linewidth=1.5, label="Selected model")
            for bar, v in zip(bars_, vals_):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")
            ax.set_ylabel("Predicted Strength (MPa)", fontsize=11)
            ax.set_title("28-day Strength Prediction — All Models Compared", fontsize=12, fontweight="bold")
            ax.set_xticklabels(names_, rotation=15, ha="right")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── TARGET STRENGTH MODE ──────────────────────────────────────────────────
    else:
        st.markdown('<div class="section-header">🎯 Target Strength → Optimal Mix Design</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            💡 Enter your <b>desired compressive strength</b> and the optimizer will find the best 
            combination of mix parameters to achieve it using the selected model.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            target_strength = st.number_input(
                "🎯 Target 28-day Compressive Strength (MPa)",
                min_value=5.0, max_value=100.0, value=40.0, step=1.0,
                help="Enter the desired strength in MPa"
            )
        with c2:
            st.markdown("")
            st.markdown("")
            clamp_to_data = st.checkbox("Clamp to dataset range", value=True,
                help="Keep numerical parameters within observed data range")
        with c3:
            st.markdown("")
            st.markdown("")
            find_btn = st.button("⚡ Find Optimal Mix")

        st.markdown(f"""
        <div class="warn-box">
            ⚠️ Dataset range: <b>{df['Compression_28d'].min():.1f} – {df['Compression_28d'].max():.1f} MPa</b>.
            Targets outside this range may have lower accuracy.
        </div>
        """, unsafe_allow_html=True)

        if find_btn:
            with st.spinner("🔍 Optimizing mix design..."):
                best_combo, achieved = inverse_predict(
                    target_strength, selected_model, scaler, label_encoders,
                    feature_names, categorical_cols, numerical_cols, df, num_imputer
                )

            if best_combo is None:
                st.error("Optimization failed. Please try a different target or model.")
            else:
                diff = abs(achieved - target_strength)
                st.markdown(f"""
                <div class="result-banner">
                    <div class="result-value">{achieved:.2f} MPa</div>
                    <div class="result-label">Achieved Predicted Strength (Target: {target_strength} MPa)</div>
                    <div class="result-grade" style="color:{'#3fb950' if diff < 3 else '#f0b429'}">
                        {'✅ Excellent match' if diff < 3 else f'⚡ Off by {diff:.2f} MPa'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-header">📋 Recommended Mix Design</div>', unsafe_allow_html=True)

                rec1, rec2 = st.columns(2)
                icons = {"Source_material": "🧱", "Alkali_solution": "🧪",
                         "Molarity": "⚗️", "SS_SH_KOH_ratio": "🔬",
                         "A_B_ratio": "📐", "Curing_temp": "🌡️"}

                for i, (key, val) in enumerate(best_combo.items()):
                    container = rec1 if i % 2 == 0 else rec2
                    formatted = f"{val:.3f}" if isinstance(val, float) else str(val)
                    icon = icons.get(key, "•")
                    container.markdown(f"""
                    <div class="inv-card">
                        <div class="inv-card-title">{icon} {key.replace('_', ' ')}</div>
                        <div class="inv-card-value">{formatted}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Visualization — parameter comparison
                st.markdown("#### 📊 Recommended vs Dataset Average")
                fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                num_keys = [k for k in best_combo if k in numerical_cols]
                for ax_i, key in enumerate(num_keys):
                    rec_val = best_combo[key]
                    avg_val = df[key].mean()
                    ax = axes[ax_i]
                    bars = ax.bar(["Recommended", "Dataset Avg"], [rec_val, avg_val],
                                  color=["#3fb950", "#58a6ff"], edgecolor="#21262d", width=0.5)
                    for bar, v in zip(bars, [rec_val, avg_val]):
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*v,
                                f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
                    ax.set_title(key.replace("_", " "), fontsize=10, fontweight="bold")
                    ax.set_ylabel("Value")
                    ax.grid(axis="y", alpha=0.3)
                plt.suptitle(f"Recommended Mix vs Dataset Average (Target: {target_strength} MPa)",
                             fontsize=12, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Sensitivity around recommended point
                st.markdown("#### 🔍 Sensitivity Around Recommended Mix")
                scols = st.columns(4)
                for i, key in enumerate(num_keys):
                    rmin = max(df[key].min(), best_combo[key] * 0.5)
                    rmax = min(df[key].max(), best_combo[key] * 1.5)
                    sweep = np.linspace(rmin, rmax, 50)
                    preds_sens = []
                    for v in sweep:
                        d = best_combo.copy()
                        d[key] = v
                        xi = preprocess_input(d, scaler, label_encoders, feature_names,
                                              categorical_cols, numerical_cols, num_imputer)
                        preds_sens.append(selected_model.predict(xi)[0])

                    fig, ax = plt.subplots(figsize=(3.5, 3))
                    ax.plot(sweep, preds_sens, color=PALETTE[i], linewidth=2)
                    ax.axvline(best_combo[key], color="white", linestyle="--", linewidth=1.5)
                    ax.axhline(target_strength, color="#f0b429", linestyle=":", linewidth=1.2,
                               label=f"Target: {target_strength}")
                    ax.axhline(achieved, color="#3fb950", linestyle=":", linewidth=1.2,
                               label=f"Achieved: {achieved:.1f}")
                    ax.set_xlabel(key.replace("_", " "), fontsize=9)
                    ax.set_ylabel("Strength (MPa)", fontsize=9)
                    ax.set_title(f"Vary {key.replace('_',' ')}", fontsize=9, fontweight="bold")
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    scols[i].pyplot(fig)
                    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    num_cols_eda = ["Molarity", "SS_SH_KOH_ratio", "A_B_ratio", "Curing_temp", "Compression_28d"]

    # Row 1 — Correlation heatmap + Distribution
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("**🔥 Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        corr = df[num_cols_eda].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                    mask=mask, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 10, "weight": "bold"})
        ax.set_title("Correlation Heatmap – Numeric Features", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with r1c2:
        st.markdown("**📈 Feature Distributions (KDE)**")
        fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))
        axes = axes.flatten()
        colors = sns.color_palette("Set2", len(num_cols_eda))
        for i, col in enumerate(num_cols_eda):
            data = df[col].dropna()
            axes[i].hist(data, bins=18, color=colors[i], alpha=0.6, edgecolor="none", density=True)
            data.plot.kde(ax=axes[i], color="white", linewidth=1.8)
            axes[i].set_title(col, fontsize=9, fontweight="bold")
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_visible(False)
        fig.suptitle("Feature Distributions with KDE", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Row 2 — Boxplot by source + Violin by alkali
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**📦 Strength by Source Material**")
        fig, ax = plt.subplots(figsize=(6, 4))
        order = df.groupby("Source_material")["Compression_28d"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="Source_material", y="Compression_28d", order=order,
                    palette="Set2", ax=ax, linewidth=1.2, fliersize=4)
        sns.stripplot(data=df, x="Source_material", y="Compression_28d", order=order,
                      color="white", alpha=0.3, size=3, jitter=True, ax=ax)
        ax.set_xlabel("Source Material", fontsize=9); ax.set_ylabel("Strength (MPa)", fontsize=9)
        ax.set_title("28-day Strength by Source Material", fontsize=11, fontweight="bold")
        plt.xticks(rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r2c2:
        st.markdown("**🎻 Strength by Alkali Solution**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=df, x="Alkali_solution", y="Compression_28d",
                       palette="coolwarm", inner="quartile", linewidth=1.0, ax=ax)
        ax.set_xlabel("Alkali Solution", fontsize=9); ax.set_ylabel("Strength (MPa)", fontsize=9)
        ax.set_title("Strength Distribution by Alkali Solution", fontsize=11, fontweight="bold")
        plt.xticks(rotation=15, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 3 — Pivot heatmaps
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("**🌡️ Heatmap: Molarity × Curing Temp → Avg Strength**")
        df_tmp = df.copy()
        df_tmp["Mol_bin"]  = pd.cut(df_tmp["Molarity"], bins=4)
        df_tmp["Cure_bin"] = pd.cut(df_tmp["Curing_temp"], bins=4)
        piv1 = df_tmp.pivot_table(values="Compression_28d", index="Cure_bin",
                                   columns="Mol_bin", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(piv1, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.4,
                    ax=ax, cbar_kws={"label": "Avg Strength (MPa)"})
        ax.set_title("Avg Strength: Curing Temp × Molarity", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r3c2:
        st.markdown("**🧪 Heatmap: Source Material × Alkali Solution → Avg Strength**")
        piv2 = df.pivot_table(values="Compression_28d", index="Source_material",
                               columns="Alkali_solution", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(piv2, annot=True, fmt=".1f", cmap="Blues", linewidths=0.4,
                    ax=ax, cbar_kws={"label": "Avg Strength (MPa)"})
        ax.set_title("Avg Strength: Source × Alkali", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 4 — Scatter plots
    st.markdown("**🔵 Feature vs Compressive Strength — Scatter Plots**")
    sc_cols = st.columns(4)
    scatter_features = ["Molarity", "SS_SH_KOH_ratio", "A_B_ratio", "Curing_temp"]
    for i, feat in enumerate(scatter_features):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        sub = df[[feat, "Compression_28d", "Source_material"]].dropna()
        for j, (src, grp) in enumerate(sub.groupby("Source_material")):
            ax.scatter(grp[feat], grp["Compression_28d"],
                       label=src, alpha=0.65, s=30, edgecolors="none",
                       color=PALETTE[j % len(PALETTE)])
        c = np.polyfit(sub[feat], sub["Compression_28d"], 1)
        xr = np.linspace(sub[feat].min(), sub[feat].max(), 100)
        ax.plot(xr, np.polyval(c, xr), "w--", linewidth=1.5)
        ax.set_xlabel(feat.replace("_", " "), fontsize=8)
        ax.set_ylabel("Strength (MPa)", fontsize=8)
        ax.set_title(f"{feat.replace('_',' ')} vs Strength", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        sc_cols[i].pyplot(fig); plt.close()

    # Row 5 — Bar chart mean strength + Missing value map
    r5c1, r5c2 = st.columns(2)

    with r5c1:
        st.markdown("**📊 Mean Strength by Source Material (±SE)**")
        stats = df.groupby("Source_material")["Compression_28d"].agg(["mean", "std", "count"])
        stats["se"] = stats["std"] / stats["count"] ** 0.5
        stats = stats.sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(stats.index, stats["mean"], yerr=stats["se"], capsize=5,
                      color=sns.color_palette("Set2", len(stats)),
                      edgecolor="#21262d", linewidth=0.8,
                      error_kw={"elinewidth": 1.5, "ecolor": "white"})
        for bar, (_, row) in zip(bars, stats.iterrows()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2,
                    f"{row['mean']:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Mean Strength (MPa)", fontsize=10)
        ax.set_title("Mean 28-day Strength by Source Material", fontsize=11, fontweight="bold")
        plt.xticks(rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r5c2:
        st.markdown("**🔍 Missing Value Heatmap**")
        df_raw2 = pd.read_excel("excel1.xlsx") if True else df
        try:
            df_raw2 = pd.read_excel("excel1.xlsx")
        except:
            df_raw2 = df.copy()
        for col in df_raw2.select_dtypes(include="object").columns:
            df_raw2[col] = df_raw2[col].astype(str).str.strip()
        df_raw2 = df_raw2.replace(["--", "-", "—", ""], np.nan)
        miss_pct = df_raw2.isnull().mean() * 100

        fig, ax = plt.subplots(figsize=(6, 4))
        clrs = ["#e74c3c" if v > 0 else "#3fb950" for v in miss_pct.values]
        bars = ax.bar(miss_pct.index, miss_pct.values, color=clrs, edgecolor="#21262d")
        for bar, v in zip(bars, miss_pct.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("Missing %", fontsize=10)
        ax.set_title("Missing Values per Feature", fontsize=11, fontweight="bold")
        plt.xticks(rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 6 — SS/SH heatmap + Grouped boxplot
    r6c1, r6c2 = st.columns(2)

    with r6c1:
        st.markdown("**🔥 Heatmap: SS/SH Ratio × Molarity → Avg Strength**")
        df_tmp2 = df.copy()
        df_tmp2["SS_bin"]  = pd.cut(df_tmp2["SS_SH_KOH_ratio"], bins=4)
        df_tmp2["Mol_bin2"] = pd.cut(df_tmp2["Molarity"], bins=4)
        piv3 = df_tmp2.pivot_table(values="Compression_28d", index="SS_bin",
                                    columns="Mol_bin2", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(piv3, annot=True, fmt=".1f", cmap="Purples", linewidths=0.4,
                    ax=ax, cbar_kws={"label": "Avg Strength (MPa)"})
        ax.set_title("Avg Strength: SS/SH × Molarity", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r6c2:
        st.markdown("**📦 Grouped Boxplot: Source × Alkali**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="Source_material", y="Compression_28d",
                    hue="Alkali_solution", palette="tab10", ax=ax, linewidth=1.0)
        ax.set_title("Strength: Source × Alkali Solution", fontsize=11, fontweight="bold")
        ax.set_xlabel("Source Material", fontsize=9)
        ax.set_ylabel("Strength (MPa)", fontsize=9)
        ax.legend(title="Alkali", fontsize=7, title_fontsize=8)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🤖 Model Benchmarking & Performance Analysis</div>',
                unsafe_allow_html=True)

    # Metrics table
    st.markdown("#### 📋 Performance Metrics Summary")
    styled = results_df.copy()
    styled.index.name = "Model"
    st.dataframe(styled.style.background_gradient(cmap="RdYlGn", subset=["R²"])
                              .background_gradient(cmap="RdYlGn_r", subset=["MAE", "RMSE"])
                              .format(precision=4), use_container_width=True)

    # Best model
    best_name = results_df["R²"].idxmax()
    st.markdown(f"""
    <div class="info-box">
        🏆 Best model by R²: <b>{best_name}</b> — R² = {results_df.loc[best_name,'R²']:.4f},
        MAE = {results_df.loc[best_name,'MAE']:.3f} MPa
    </div>
    """, unsafe_allow_html=True)

    # Row 1 — Comparison bar + Radar
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**📊 Model Comparison Bar Charts**")
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        metrics_list = ["R²", "MAE", "RMSE"]
        mcols = ["#3fb950", "#e74c3c", "#f0b429"]
        for ax, met, clr in zip(axes, metrics_list, mcols):
            rd = results_df.sort_values(met, ascending=(met != "R²"))
            bars = ax.bar(rd.index, rd[met], color=[clr if idx == selected_model_name else "#30363d"
                          for idx in rd.index], edgecolor="#21262d", width=0.6)
            for bar, v in zip(bars, rd[met]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001*rd[met].max(),
                        f"{v:.3f}", ha="center", fontsize=7.5)
            ax.set_title(met, fontsize=10, fontweight="bold")
            ax.set_xticklabels(rd.index, rotation=30, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=0.3)
        plt.suptitle("Model Performance Comparison", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with mc2:
        st.markdown("**🕸️ Radar Chart — Normalised Metrics**")
        cats_r = ["R²", "1-MAE", "1-RMSE"]
        N_r = len(cats_r)
        angles_r = [n / float(N_r) * 2 * np.pi for n in range(N_r)]
        angles_r += angles_r[:1]

        def norm_val(v, lo, hi, inv=False):
            n = (v - lo) / (hi - lo + 1e-9)
            return 1 - n if inv else n

        r2_lo, r2_hi   = results_df["R²"].min(),   results_df["R²"].max()
        mae_lo, mae_hi = results_df["MAE"].min(),  results_df["MAE"].max()
        rm_lo, rm_hi   = results_df["RMSE"].min(), results_df["RMSE"].max()

        fig, ax = plt.subplots(figsize=(5, 4.5), subplot_kw={"polar": True})
        ax.set_facecolor("#161b22"); fig.patch.set_facecolor("#0f1117")
        pal_r = sns.color_palette("Set1", len(results_df))

        for (nm, row), clr in zip(results_df.iterrows(), pal_r):
            vals_r = [
                norm_val(row["R²"],   r2_lo,  r2_hi,  inv=False),
                norm_val(row["MAE"],  mae_lo, mae_hi, inv=True),
                norm_val(row["RMSE"], rm_lo,  rm_hi,  inv=True),
            ]
            vals_r += vals_r[:1]
            lw = 3 if nm == selected_model_name else 1.5
            ax.plot(angles_r, vals_r, linewidth=lw, label=nm, color=clr)
            ax.fill(angles_r, vals_r, alpha=0.08, color=clr)

        ax.set_xticks(angles_r[:-1])
        ax.set_xticklabels(cats_r, size=10)
        ax.set_yticklabels([])
        ax.set_title("Normalised Performance Radar\n(outer = better)", size=10, fontweight="bold", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 2 — Actual vs Predicted grid
    st.markdown("**📈 Actual vs Predicted — All Models**")
    model_keys = list(trained_models.keys())
    fig, axes = plt.subplots(1, len(model_keys), figsize=(16, 3.8))
    pal_avp = sns.color_palette("tab10", len(model_keys))

    for ax, (nm, m), clr in zip(axes, trained_models.items(), pal_avp):
        yp_i = m.predict(X_test)
        ax.scatter(y_test, yp_i, alpha=0.65, s=30, edgecolors="none", color=clr)
        lims = [min(y_test.min(), yp_i.min()) - 2, max(y_test.max(), yp_i.max()) + 2]
        ax.plot(lims, lims, "w--", linewidth=1.2)
        r2_i = r2_score(y_test, yp_i)
        lw = 2 if nm == selected_model_name else 1
        ax.set_title(f"{nm}\nR²={r2_i:.3f}", fontweight="bold",
                     fontsize=8, color="#58a6ff" if nm == selected_model_name else "#c9d1d9")
        ax.set_xlabel("Actual", fontsize=8); ax.set_ylabel("Predicted", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Actual vs Predicted – All Models", fontsize=11, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 3 — Residual analysis + Feature importance side by side
    ra_c1, ra_c2 = st.columns(2)

    with ra_c1:
        st.markdown(f"**📉 Residual Analysis — {selected_model_name}**")
        y_pred_sel = selected_model.predict(X_test)
        residuals = y_test.values - y_pred_sel

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        sc = axes[0].scatter(y_pred_sel, residuals, alpha=0.7, c=residuals, cmap="coolwarm",
                             edgecolors="none", s=40)
        axes[0].axhline(0, color="white", linewidth=1.5, linestyle="--")
        axes[0].set_xlabel("Predicted (MPa)"); axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted", fontsize=9, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(sc, ax=axes[0], shrink=0.8)

        axes[1].hist(residuals, bins=14, color="#58a6ff", edgecolor="none", density=True, alpha=0.75)
        mu_, sd_ = residuals.mean(), residuals.std()
        xr_r = np.linspace(residuals.min(), residuals.max(), 200)
        axes[1].plot(xr_r, norm.pdf(xr_r, mu_, sd_), color="#f0b429", linewidth=2)
        axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Density")
        axes[1].set_title("Residual Distribution", fontsize=9, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        stats_module.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot", fontsize=9, fontweight="bold")
        axes[2].get_lines()[0].set(color="#58a6ff", markersize=4, alpha=0.7)
        axes[2].get_lines()[1].set(color="#f0b429", linewidth=2)
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with ra_c2:
        st.markdown("**🌲 Feature Importances (RF & GB)**")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, (nm, clr, cmap_) in zip(axes, [
            ("Random Forest", "#3fb950", "Greens"),
            ("Gradient Boosting", "#58a6ff", "Blues")
        ]):
            m_ = trained_models[nm]
            if hasattr(m_, "feature_importances_"):
                imp = m_.feature_importances_
                idx = np.argsort(imp)
                ax.barh([feature_names[i] for i in idx], imp[idx],
                        color=sns.color_palette(cmap_, len(feature_names)),
                        edgecolor="none")
                for i, (bar, v) in enumerate(zip(ax.patches, imp[idx])):
                    ax.text(v + 0.005, bar.get_y()+bar.get_height()/2,
                            f"{v:.3f}", va="center", fontsize=7.5)
                ax.set_title(nm, fontsize=9, fontweight="bold")
                ax.set_xlabel("Importance")
                ax.grid(axis="x", alpha=0.3)
        plt.suptitle("Feature Importances", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Cumulative error distribution
    st.markdown("**📊 Cumulative Error Distribution — All Models**")
    fig, ax = plt.subplots(figsize=(10, 4))
    for nm, m in trained_models.items():
        yp_m = m.predict(X_test)
        pct_err = np.abs(y_test.values - yp_m) / (y_test.values + 1e-9) * 100
        pct_sorted = np.sort(pct_err)
        cum = np.arange(1, len(pct_sorted)+1) / len(pct_sorted) * 100
        lw = 3 if nm == selected_model_name else 1.5
        ax.plot(pct_sorted, cum, linewidth=lw, label=nm)
    ax.axvline(10, color="grey", linestyle="--", linewidth=1.2, label="10% error")
    ax.axvline(20, color="#f0b429", linestyle="--", linewidth=1.2, label="20% error")
    ax.set_xlabel("Absolute Percentage Error (%)", fontsize=11)
    ax.set_ylabel("Cumulative % of Samples", fontsize=11)
    ax.set_title("Cumulative Error Distribution — All Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 80)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📂 Dataset Overview</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Samples</div>
        <div class="metric-value">{len(df)}</div>
    </div>""", unsafe_allow_html=True)
    col_s2.markdown(f"""<div class="metric-card">
        <div class="metric-label">Features</div>
        <div class="metric-value">6</div>
    </div>""", unsafe_allow_html=True)
    col_s3.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg Strength</div>
        <div class="metric-value">{df['Compression_28d'].mean():.1f}</div>
        <div class="metric-sub">MPa</div>
    </div>""", unsafe_allow_html=True)
    col_s4.markdown(f"""<div class="metric-card">
        <div class="metric-label">Max Strength</div>
        <div class="metric-value">{df['Compression_28d'].max():.1f}</div>
        <div class="metric-sub">MPa</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 📋 Full Dataset")
    st.dataframe(df, use_container_width=True, height=400)

    st.markdown("#### 📊 Statistical Summary")
    st.dataframe(df.describe().round(3), use_container_width=True)

    st.markdown("#### 📑 Category Counts")
    c1_d, c2_d = st.columns(2)
    with c1_d:
        st.markdown("**Source Material Distribution**")
        vc1 = df["Source_material"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(vc1.values, labels=vc1.index, autopct="%1.1f%%",
               colors=sns.color_palette("Set2", len(vc1)),
               startangle=90, pctdistance=0.8,
               textprops={"color": "white", "fontsize": 9})
        ax.set_title("Source Material", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2_d:
        st.markdown("**Alkali Solution Distribution**")
        vc2 = df["Alkali_solution"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(vc2.values, labels=vc2.index, autopct="%1.1f%%",
               colors=sns.color_palette("Set3", len(vc2)),
               startangle=90, pctdistance=0.8,
               textprops={"color": "white", "fontsize": 9})
        ax.set_title("Alkali Solution", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()