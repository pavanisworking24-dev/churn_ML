import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ─────────────────────────────────────────────
#  CUSTOM CSS  — dark-telecom aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
 
:root {
    --bg:        #0b0e1a;
    --surface:   #131728;
    --border:    #1e2540;
    --accent:    #4f8ef7;
    --danger:    #f7554f;
    --warn:      #f7a74f;
    --ok:        #4fcc8e;
    --text:      #e8ecf7;
    --muted:     #6b738f;
}
 
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
h1,h2,h3,h4 { font-family: 'Space Mono', monospace !important; color: var(--text) !important; }
.stButton > button {
    background: var(--accent); color: #fff; border: none;
    border-radius: 8px; padding: 0.55rem 1.4rem;
    font-family: 'Space Mono', monospace; font-size: 0.85rem;
    letter-spacing: .04em; transition: opacity .2s;
}
.stButton > button:hover { opacity: .82; }
.stSelectbox > div, .stNumberInput > div, .stTextInput > div { color: var(--text) !important; }
div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.stDataFrame { background: var(--surface) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: 'Space Mono', monospace; font-size: .8rem; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent); }
hr { border-color: var(--border) !important; }
 
/* risk badge */
.badge-high   { background:#f7554f22; color:#f7554f; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.badge-medium { background:#f7a74f22; color:#f7a74f; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.badge-low    { background:#4fcc8e22; color:#4fcc8e; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
 
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: .7rem;
    letter-spacing: .14em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: .4rem;
}
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load("best_churn_model.pkl")
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception:
        scaler = joblib.load("scaler (2).pkl")

    try:
        sel_feats = joblib.load("selected_features.pkl")
    except Exception:
        sel_feats = joblib.load("selected_features (1).pkl")

    enc_cols  = joblib.load("encoded_columns.pkl")
    threshold = joblib.load("optimal_threshold.pkl")
    return model, scaler, sel_feats, enc_cols, threshold
 
try:
    model, scaler, selected_features, encoded_columns, THRESHOLD = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    st.error(f"❌ Could not load model artifacts: {e}\n\nMake sure all 5 .pkl files are in the same folder as app.py.")
    st.stop()
 
 
# ─────────────────────────────────────────────
#  PREPROCESSING  (mirrors the notebook exactly)
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # Basic cleaning
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
 
    # Target encode
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
 
    # Binary columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "MultipleLines", "OnlineSecurity",
                   "OnlineBackup", "DeviceProtection", "TechSupport",
                   "StreamingTV", "StreamingMovies"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0,
                                   "No phone service": 0, "No internet service": 0}).fillna(0).astype(int)
 
    # Feature engineering
    df["charges_per_tenure"]   = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["total_charges_ratio"]  = df["TotalCharges"] / (df["MonthlyCharges"] * (df["tenure"] + 1) + 1)
    df["is_new_customer"]      = (df["tenure"] <= 3).astype(int)
    df["is_loyal_customer"]    = (df["tenure"] >= 48).astype(int)
    df["avg_monthly_spend"]    = df["TotalCharges"] / (df["tenure"] + 1)
 
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    available = [c for c in service_cols if c in df.columns]
    df["num_services"] = df[available].sum(axis=1)
    df["has_multiple_services"] = (df["num_services"] >= 3).astype(int)
 
    df["tenure_monthly_interaction"] = df["tenure"] * df["MonthlyCharges"]
    df["high_monthly_charges"]       = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
 
    # Tenure bins
    df["tenure_bin"] = pd.cut(df["tenure"], bins=[0,12,24,48,72],
                              labels=["0-12","12-24","24-48","48-72"], include_lowest=True)
    df["tenure_bin"] = df["tenure_bin"].astype(str)
 
    # One-hot encode
    cat_cols = ["InternetService", "Contract", "PaymentMethod", "tenure_bin"]
    existing_cat = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cat, drop_first=False)
 
    # Align to training columns
    for col in encoded_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[[c for c in encoded_columns if c in df.columns]]
 
    # Scale
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
 
    # Select features
    available_sel = [f for f in selected_features if f in df_scaled.columns]
    return df_scaled[available_sel]
 
 
def predict(df_processed: pd.DataFrame):
    probs = model.predict_proba(df_processed)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)
    return probs, preds
 
 
def risk_label(prob):
    if prob >= 0.65:
        return "High", "badge-high"
    elif prob >= 0.35:
        return "Medium", "badge-medium"
    else:
        return "Low", "badge-low"


REQUIRED_INPUT_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


def validate_input_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    return missing
 
 
# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.markdown("## 📡 Churn Risk")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "👤 Single Prediction", "📂 Batch Prediction"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:.78rem; color:#6b738f;'>
<b style='color:#e8ecf7;'>Model</b><br>Logistic Regression<br><br>
<b style='color:#e8ecf7;'>ROC-AUC</b><br>0.8392<br><br>
<b style='color:#e8ecf7;'>Threshold</b><br>{THRESHOLD}<br><br>
<b style='color:#e8ecf7;'>Recall</b><br>0.92
</div>
""", unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════
#  PAGE 1 — DASHBOARD  (requires uploaded CSV)
# ═══════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# 🏠 Churn Risk Dashboard")
    st.markdown("Upload a customer CSV to explore the dashboard.")
 
    uploaded = st.file_uploader("Upload customer dataset (CSV)", type="csv", key="dash_upload")
 
    if uploaded:
        raw = pd.read_csv(uploaded)
        missing_cols = validate_input_columns(raw)
        if missing_cols:
            st.error(
                "Uploaded CSV is missing required columns: "
                + ", ".join(missing_cols)
            )
            st.markdown("#### Required columns")
            st.code(", ".join(REQUIRED_INPUT_COLUMNS), language="text")
            st.stop()
        try:
            processed = preprocess(raw)
            probs, preds = predict(processed)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()
 
        raw["churn_prob"]  = probs
        raw["churn_pred"]  = preds
        raw["risk_level"]  = raw["churn_prob"].apply(lambda p: risk_label(p)[0])
 
        total      = len(raw)
        high_risk  = (raw["risk_level"] == "High").sum()
        med_risk   = (raw["risk_level"] == "Medium").sum()
        avg_prob   = probs.mean()
 
        # ── KPI Cards
        st.markdown("### Key Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", f"{total:,}")
        c2.metric("High Risk 🔴", f"{high_risk:,}", f"{high_risk/total*100:.1f}%")
        c3.metric("Medium Risk 🟠", f"{med_risk:,}", f"{med_risk/total*100:.1f}%")
        c4.metric("Avg Churn Prob", f"{avg_prob:.1%}")
 
        st.markdown("---")
 
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔬 Feature Insights", "📋 Customer Table"])
 
        # ── Tab 1: Distribution charts
        with tab1:
            col_a, col_b = st.columns(2)
 
            with col_a:
                fig_hist = px.histogram(
                    raw, x="churn_prob", nbins=40,
                    color="risk_level",
                    color_discrete_map={"High": "#f7554f", "Medium": "#f7a74f", "Low": "#4fcc8e"},
                    title="Churn Probability Distribution",
                    labels={"churn_prob": "Churn Probability", "count": "Customers"},
                    template="plotly_dark",
                )
                fig_hist.update_layout(
                    paper_bgcolor="#131728", plot_bgcolor="#0b0e1a",
                    legend_title="Risk Level", bargap=0.05,
                )
                fig_hist.add_vline(x=THRESHOLD, line_dash="dash", line_color="#4f8ef7",
                                   annotation_text=f"Threshold={THRESHOLD}", annotation_font_color="#4f8ef7")
                st.plotly_chart(fig_hist, use_container_width=True)
 
            with col_b:
                risk_counts = raw["risk_level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                fig_pie = px.pie(
                    risk_counts, names="Risk Level", values="Count",
                    color="Risk Level",
                    color_discrete_map={"High": "#f7554f", "Medium": "#f7a74f", "Low": "#4fcc8e"},
                    title="Risk Segment Breakdown",
                    hole=0.52,
                    template="plotly_dark",
                )
                fig_pie.update_layout(paper_bgcolor="#131728")
                st.plotly_chart(fig_pie, use_container_width=True)
 
            # Contract & tenure breakdown if columns exist
            if "Contract" in raw.columns:
                col_c, col_d = st.columns(2)
                with col_c:
                    contract_churn = raw.groupby("Contract")["churn_prob"].mean().reset_index()
                    fig_bar = px.bar(
                        contract_churn, x="Contract", y="churn_prob",
                        title="Avg Churn Prob by Contract Type",
                        color="churn_prob", color_continuous_scale=["#4fcc8e","#f7a74f","#f7554f"],
                        template="plotly_dark",
                    )
                    fig_bar.update_layout(paper_bgcolor="#131728", plot_bgcolor="#0b0e1a")
                    st.plotly_chart(fig_bar, use_container_width=True)
 
            if "tenure" in raw.columns:
                col_e = st.columns(1)[0]
                fig_scatter = px.scatter(
                    raw.sample(min(1000, len(raw))), x="tenure", y="churn_prob",
                    color="risk_level",
                    color_discrete_map={"High":"#f7554f","Medium":"#f7a74f","Low":"#4fcc8e"},
                    opacity=0.6, title="Tenure vs Churn Probability (sample ≤ 1000)",
                    template="plotly_dark",
                )
                fig_scatter.update_layout(paper_bgcolor="#131728", plot_bgcolor="#0b0e1a")
                fig_scatter.add_hline(y=THRESHOLD, line_dash="dash", line_color="#4f8ef7")
                st.plotly_chart(fig_scatter, use_container_width=True)
 
        # ── Tab 2: Feature importance via SHAP
        with tab2:
            st.markdown("#### SHAP Feature Importance (sample ≤ 200 rows)")
            sample_size = min(200, len(processed))
            X_sample = processed.sample(sample_size, random_state=42)
 
            try:
                explainer  = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
                shap_vals  = explainer.shap_values(X_sample)
                importance = pd.DataFrame({
                    "Feature":    X_sample.columns,
                    "Mean |SHAP|": np.abs(shap_vals).mean(axis=0),
                }).sort_values("Mean |SHAP|", ascending=False).head(15)
 
                fig_shap = px.bar(
                    importance, x="Mean |SHAP|", y="Feature", orientation="h",
                    title="Top 15 Features by Mean |SHAP| Value",
                    color="Mean |SHAP|", color_continuous_scale=["#4f8ef7","#f7554f"],
                    template="plotly_dark",
                )
                fig_shap.update_layout(
                    paper_bgcolor="#131728", plot_bgcolor="#0b0e1a",
                    yaxis={"autorange":"reversed"},
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP could not run: {e}")
 
        # ── Tab 3: Customer table with filters
        with tab3:
            st.markdown("#### Filter Customers")
            f1, f2 = st.columns(2)
            risk_filter = f1.multiselect("Risk Level", ["High","Medium","Low"], default=["High","Medium","Low"])
            prob_range  = f2.slider("Churn Probability Range", 0.0, 1.0, (0.0, 1.0), 0.01)
 
            filtered = raw[
                (raw["risk_level"].isin(risk_filter)) &
                (raw["churn_prob"] >= prob_range[0]) &
                (raw["churn_prob"] <= prob_range[1])
            ].copy()
 
            filtered["churn_prob"] = filtered["churn_prob"].map("{:.1%}".format)
            show_cols = ["customerID"] if "customerID" in filtered.columns else []
            show_cols += ["tenure","MonthlyCharges","Contract","churn_prob","risk_level"] if all(
                c in filtered.columns for c in ["tenure","MonthlyCharges","Contract"]
            ) else ["churn_prob","risk_level"]
 
            st.markdown(f"**{len(filtered):,} customers** match the filter.")
            st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True, height=420)
 
            csv_out = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Filtered CSV", csv_out, "filtered_customers.csv", "text/csv")
 
    else:
        st.info("👆 Upload a CSV file with customer data to activate the dashboard.")
 
 
# ═══════════════════════════════════════════════
#  PAGE 2 — SINGLE PREDICTION
# ═══════════════════════════════════════════════
elif page == "👤 Single Prediction":
    st.markdown("# 👤 Single Customer Prediction")
    st.markdown("Fill in customer details to get an instant churn risk score.")
    st.markdown("---")
 
    with st.form("single_pred_form"):
        c1, c2, c3 = st.columns(3)
 
        with c1:
            st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
            gender           = st.selectbox("Gender", ["Male","Female"])
            senior_citizen   = st.selectbox("Senior Citizen", ["No","Yes"])
            partner          = st.selectbox("Partner", ["No","Yes"])
            dependents       = st.selectbox("Dependents", ["No","Yes"])
 
        with c2:
            st.markdown('<div class="section-header">Account</div>', unsafe_allow_html=True)
            tenure           = st.number_input("Tenure (months)", 0, 72, 12)
            monthly_charges  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
            total_charges    = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure, 1.0)
            contract         = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            payment_method   = st.selectbox("Payment Method", [
                "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            paperless        = st.selectbox("Paperless Billing", ["No","Yes"])
 
        with c3:
            st.markdown('<div class="section-header">Services</div>', unsafe_allow_html=True)
            phone_service    = st.selectbox("Phone Service", ["Yes","No"])
            multiple_lines   = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            online_security  = st.selectbox("Online Security", ["No","Yes","No internet service"])
            online_backup    = st.selectbox("Online Backup", ["No","Yes","No internet service"])
            device_prot      = st.selectbox("Device Protection", ["No","Yes","No internet service"])
            tech_support     = st.selectbox("Tech Support", ["No","Yes","No internet service"])
            streaming_tv     = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
 
        submitted = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True)
 
    if submitted:
        input_dict = {
            "gender": gender, "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_prot,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }
        input_df = pd.DataFrame([input_dict])
 
        try:
            proc = preprocess(input_df)
            prob, pred = predict(proc)
            prob_val = float(prob[0])
            label, badge_cls = risk_label(prob_val)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()
 
        st.markdown("---")
        r1, r2, r3 = st.columns([1,1,2])
 
        r1.metric("Churn Probability", f"{prob_val:.1%}")
        r2.metric("Prediction", "Will Churn ⚠️" if pred[0] == 1 else "Will Stay ✅")
        r3.markdown(f"""
        <br>
        <span class="{badge_cls}" style="font-size:1rem;padding:6px 18px;">
            {label} Risk
        </span>
        """, unsafe_allow_html=True)
 
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_val * 100,
            title={"text": "Churn Risk Score", "font": {"color": "#e8ecf7", "family": "Space Mono"}},
            number={"suffix": "%", "font": {"color": "#e8ecf7"}},
            gauge={
                "axis": {"range": [0,100], "tickcolor": "#6b738f"},
                "bar":  {"color": "#f7554f" if label=="High" else ("#f7a74f" if label=="Medium" else "#4fcc8e")},
                "bgcolor": "#131728",
                "bordercolor": "#1e2540",
                "steps": [
                    {"range": [0,35],  "color": "#0d1f2d"},
                    {"range": [35,65], "color": "#1a1a10"},
                    {"range": [65,100],"color": "#1f0d0d"},
                ],
                "threshold": {
                    "line": {"color": "#4f8ef7","width": 3},
                    "thickness": 0.75,
                    "value": THRESHOLD * 100,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#131728", font_color="#e8ecf7",
            height=300, margin=dict(t=40,b=10,l=30,r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
 
        # SHAP waterfall for this customer
        st.markdown("#### Why this prediction? (SHAP)")
        try:
            explainer = shap.LinearExplainer(model, proc, feature_perturbation="interventional")
            sv = explainer.shap_values(proc)[0]
            shap_df = pd.DataFrame({"Feature": proc.columns, "SHAP": sv})
            shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index).head(12)
            shap_df["Color"] = shap_df["SHAP"].apply(lambda x: "#f7554f" if x > 0 else "#4fcc8e")
 
            fig_wf = go.Figure(go.Bar(
                x=shap_df["SHAP"], y=shap_df["Feature"],
                orientation="h",
                marker_color=shap_df["Color"].tolist(),
            ))
            fig_wf.update_layout(
                title="SHAP Contribution (red = increases churn risk)",
                paper_bgcolor="#131728", plot_bgcolor="#0b0e1a",
                font_color="#e8ecf7", yaxis={"autorange":"reversed"},
                height=380, xaxis_title="SHAP Value",
            )
            st.plotly_chart(fig_wf, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")
 
 
# ═══════════════════════════════════════════════
#  PAGE 3 — BATCH PREDICTION
# ═══════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.markdown("# 📂 Batch Prediction")
    st.markdown("Upload a CSV of customers to score them all at once.")
    st.markdown("---")
 
    uploaded_batch = st.file_uploader("Upload customer CSV (no Churn column required)", type="csv", key="batch")
 
    if uploaded_batch:
        raw_batch = pd.read_csv(uploaded_batch)
        missing_cols = validate_input_columns(raw_batch)
        if missing_cols:
            st.error(
                "Uploaded CSV is missing required columns: "
                + ", ".join(missing_cols)
            )
            st.markdown("#### Required columns")
            st.code(", ".join(REQUIRED_INPUT_COLUMNS), language="text")
            st.stop()
        st.markdown(f"**{len(raw_batch):,} rows** loaded.")
 
        if st.button("🚀 Run Batch Prediction"):
            with st.spinner("Processing..."):
                try:
                    proc_batch = preprocess(raw_batch)
                    probs_b, preds_b = predict(proc_batch)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()
 
            result = raw_batch.copy()
            result["churn_probability"] = probs_b
            result["churn_prediction"]  = preds_b
            result["risk_level"]        = result["churn_probability"].apply(lambda p: risk_label(p)[0])
 
            # Summary KPIs
            total_b = len(result)
            high_b  = (result["risk_level"]=="High").sum()
            med_b   = (result["risk_level"]=="Medium").sum()
            low_b   = (result["risk_level"]=="Low").sum()
 
            st.success("✅ Predictions complete!")
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total Scored",  f"{total_b:,}")
            k2.metric("High Risk 🔴",  f"{high_b:,}")
            k3.metric("Medium Risk 🟠",f"{med_b:,}")
            k4.metric("Low Risk 🟢",   f"{low_b:,}")
 
            st.markdown("---")
 
            # Distribution
            fig_b = px.histogram(
                result, x="churn_probability", nbins=40, color="risk_level",
                color_discrete_map={"High":"#f7554f","Medium":"#f7a74f","Low":"#4fcc8e"},
                title="Batch Churn Probability Distribution", template="plotly_dark",
            )
            fig_b.add_vline(x=THRESHOLD, line_dash="dash", line_color="#4f8ef7")
            fig_b.update_layout(paper_bgcolor="#131728", plot_bgcolor="#0b0e1a")
            st.plotly_chart(fig_b, use_container_width=True)
 
            # Table preview
            st.markdown("#### Results Preview (top 100)")
            preview_cols = (["customerID"] if "customerID" in result.columns else []) + \
                           ["churn_probability","churn_prediction","risk_level"]
            st.dataframe(
                result[preview_cols].head(100).style.format({"churn_probability": "{:.1%}"}),
                use_container_width=True, height=380,
            )
 
            # Download
            csv_dl = result.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Full Results CSV", csv_dl,
                               "batch_churn_predictions.csv", "text/csv")
    else:
        st.info("👆 Upload a CSV to begin batch scoring.")
 
        st.markdown("#### Expected CSV columns")
        expected = ["customerID"] + REQUIRED_INPUT_COLUMNS
        st.code(", ".join(expected), language="text")
 



