import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ChurnAI | Predictive Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background-color: #0b0c10;
        background-image:
            radial-gradient(rgba(255, 107, 0, 0.10) 1px, transparent 0),
            linear-gradient(to right, rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 24px 24px, 40px 40px, 40px 40px;
        color: #e0e6ed;
    }

    section[data-testid="stSidebar"] {
        background: rgba(14, 16, 22, 0.9);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(18, 20, 29, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5) !important;
        padding: 10px !important;
        transition: all 0.25s ease-in-out;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(255, 107, 0, 0.4) !important;
        box-shadow: 0 14px 40px -6px rgba(255, 107, 0, 0.18) !important;
        transform: translateY(-2px);
    }

    .bento-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        color: #ff6b00;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .metric-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-box:hover { border-color: rgba(255,107,0,0.35); }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 800; color: #fff; }
    .metric-label { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #12141d !important;
        border-color: rgba(255,255,255,0.12) !important;
        color: #fff !important;
        border-radius: 8px !important;
    }
    .stSlider [data-baseweb="slider"] { padding-top: 6px; }

    div.stButton > button:first-child {
        background: linear-gradient(135deg, #ff6b00, #ff8f2e) !important;
        color: #0b0c10 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.5rem !important;
        box-shadow: 0 0 20px rgba(255, 107, 0, 0.4) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        margin-top: 6px !important;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 0 32px rgba(255, 107, 0, 0.7) !important;
        transform: translateY(-2px) scale(1.01);
    }

    .risk-chip {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 12px; border-radius: 20px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700;
        background: rgba(255,255,255,0.04); margin: 4px 6px 4px 0;
    }

    @keyframes fadein { from {opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);} }
    .fadein { animation: fadein 0.45s ease-out; }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,107,0,0.35); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ASSET LOADING (with friendly error handling)
# ============================================================
@st.cache_resource
def load_assets():
    model = load_model('model.keras')
    with open('lebel_encoder_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
    with open('onehot_encoder_geo.pkl', 'rb') as f:
        ohe_geo = pickle.load(f)
    with open('scalar.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler

try:
    model, le_gender, ohe_geo, scaler = load_assets()
    assets_ok = True
except Exception as e:
    assets_ok = False
    st.error(f"⚠️ Couldn't load model assets: `{e}`. Make sure model.keras, "
             f"lebel_encoder_gender.pkl, onehot_encoder_geo.pkl and scalar.pkl "
             f"are in the same folder as this script.")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚡ CHURN.AI")
    st.caption("Deep learning risk-scoring engine")
    st.markdown("---")
    st.markdown(
        """
        **How to use**
        1. Fill in the customer profile
        2. Click **Run Inference**
        3. Read the risk score & drivers

        Built with an ANN classifier trained on
        bank customer data.
        """
    )
    st.markdown("---")
    if "history" not in st.session_state:
        st.session_state.history = []
    st.markdown(f"**Predictions this session:** {len(st.session_state.history)}")
    if st.session_state.history and st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ============================================================
# HEADER / STATUS BAR
# ============================================================
col_logo, col_stat1, col_stat2, col_stat3 = st.columns([3, 1, 1, 1])
with col_logo:
    st.markdown("<h2 style='margin:0; font-weight:800; letter-spacing:-1px; color:#fff;'>CHURN<span style='color:#ff6b00;'>.AI</span></h2>", unsafe_allow_html=True)
    st.caption("Deep Learning Risk Scoring Engine • v2.5")
with col_stat1:
    st.markdown("<div class='metric-box'><div class='metric-value'>ANN</div><div class='metric-label'>Model Type</div></div>", unsafe_allow_html=True)
with col_stat2:
    st.markdown("<div class='metric-box'><div class='metric-value'>v2.5</div><div class='metric-label'>Build</div></div>", unsafe_allow_html=True)
with col_stat3:
    status = "ONLINE" if assets_ok else "OFFLINE"
    color = "#00e676" if assets_ok else "#ff3344"
    st.markdown(f"<div class='metric-box'><div class='metric-value' style='color:{color};'>{status}</div><div class='metric-label'>Model Status</div></div>", unsafe_allow_html=True)

st.write("")

if not assets_ok:
    st.stop()

# ============================================================
# INPUT FORM
# ============================================================
left_col, mid_col, right_col = st.columns([1.1, 1.2, 1.3])

with left_col:
    with st.container(border=True):
        st.markdown("<span class='bento-header'>👤 01 · Demographics</span>", unsafe_allow_html=True)
        geography = st.selectbox('Geography', ohe_geo.categories_[0])
        gender = st.selectbox('Gender', le_gender.classes_)
        age = st.slider('Age', 18, 92, 35)

    with st.container(border=True):
        st.markdown("<span class='bento-header'>📊 02 · Activity & Engagement</span>", unsafe_allow_html=True)
        tenure = st.slider('Tenure (Years)', 0, 10, 3)
        num_of_products = st.slider('Active Products', 1, 4, 2)
        is_active_member = st.select_slider(
            'Is Active Member',
            options=[0, 1],
            value=1,
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

with mid_col:
    with st.container(border=True):
        st.markdown("<span class='bento-header'>💳 03 · Financial Profile</span>", unsafe_allow_html=True)
        credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650, step=1)
        balance = st.number_input('Account Balance ($)', min_value=0.0, value=50000.0, step=1000.0)
        estimated_salary = st.number_input('Estimated Annual Salary ($)', min_value=0.0, value=75000.0, step=1000.0)
        has_cr_card = st.select_slider(
            'Has Credit Card',
            options=[0, 1],
            value=1,
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        st.write("")
        predict_clicked = st.button("RUN INFERENCE ⚡", use_container_width=True)

# ============================================================
# GAUGE CHART HELPER
# ============================================================
def make_gauge(probability, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 40, 'color': '#ffffff', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8b949e', 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0,230,118,0.12)'},
                {'range': [40, 70], 'color': 'rgba(255,193,7,0.12)'},
                {'range': [70, 100], 'color': 'rgba(255,51,68,0.12)'},
            ],
            'threshold': {'line': {'color': '#fff', 'width': 2}, 'thickness': 0.8, 'value': 50}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=10, b=10),
        height=230,
        font={'color': '#e0e6ed'}
    )
    return fig

# ============================================================
# RISK DRIVERS (simple, transparent heuristics — not SHAP)
# ============================================================
def get_risk_factors():
    factors = []
    if num_of_products == 1:
        factors.append(("Only 1 product", "risk"))
    if is_active_member == 0:
        factors.append(("Inactive member", "risk"))
    if age >= 50:
        factors.append(("Older age band", "risk"))
    if balance == 0:
        factors.append(("Zero balance", "risk"))
    if tenure <= 1:
        factors.append(("New customer", "risk"))
    if num_of_products >= 2 and is_active_member == 1:
        factors.append(("Multi-product & active", "safe"))
    if tenure >= 5:
        factors.append(("Long tenure", "safe"))
    if has_cr_card == 1:
        factors.append(("Has credit card", "safe"))
    return factors

# ============================================================
# OUTPUT
# ============================================================
with right_col:
    with st.container(border=True):
        st.markdown("<span class='bento-header'>🎯 04 · Inference Output</span>", unsafe_allow_html=True)

        if predict_clicked:
            with st.spinner("Scoring customer profile..."):
                input_data = pd.DataFrame({
                    'CreditScore': [credit_score],
                    'Geography': [geography],
                    'Gender': [le_gender.transform([gender])[0]],
                    'Age': [age],
                    'Tenure': [tenure],
                    'Balance': [balance],
                    'NumOfProducts': [num_of_products],
                    'HasCrCard': [has_cr_card],
                    'IsActiveMember': [is_active_member],
                    'EstimatedSalary': [estimated_salary]
                })

                geo_encoded = pd.DataFrame(
                    ohe_geo.transform(input_data[['Geography']]).toarray(),
                    columns=ohe_geo.get_feature_names_out(['Geography'])
                )

                data = pd.concat([input_data, geo_encoded], axis=1).drop('Geography', axis=1)
                data_scaled = scaler.transform(data)

                prediction = model.predict(data_scaled, verbose=0)
                probability = float(prediction[0][0])

            is_churn = probability >= 0.5
            status_color = "#ff3344" if is_churn else "#00e676"
            status_label = "HIGH CHURN RISK" if is_churn else "LIKELY RETAINED"

            st.session_state.history.append({
                "Geography": geography, "Gender": gender, "Age": age,
                "Probability": round(probability, 3), "Result": status_label
            })

            st.plotly_chart(make_gauge(probability, status_color), use_container_width=True)

            st.markdown(f"""
                <div class="fadein" style="text-align:center; margin-top:-10px;">
                    <span class="risk-chip" style="border:1px solid {status_color}; color:{status_color};">
                        ● {status_label}
                    </span>
                </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.markdown("<span class='bento-header' style='margin-bottom:6px;'>🔍 Key Drivers</span>", unsafe_allow_html=True)
            factors = get_risk_factors()
            if factors:
                chips = ""
                for label, kind in factors:
                    c = "#ff3344" if kind == "risk" else "#00e676"
                    chips += f"<span class='risk-chip' style='border:1px solid {c}55; color:{c};'>{'▲' if kind=='risk' else '▼'} {label}</span>"
                st.markdown(f"<div class='fadein'>{chips}</div>", unsafe_allow_html=True)
            else:
                st.caption("No strong risk or retention signals detected.")

            st.markdown(f"""
                <div style="margin-top:16px; font-size:0.8rem; color:#8b949e; line-height:1.6; border-top:1px solid rgba(255,255,255,0.08); padding-top:14px;">
                    <b style="color:#fff;">Summary:</b> This customer profile carries a
                    <b style="color:{status_color};">{probability:.1%}</b> churn probability.
                    {"Consider proactive retention outreach — check pricing tier, product bundle, and recent support tickets." if is_churn else "Profile looks healthy — standard engagement cadence should suffice."}
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
                <div style="text-align:center; padding:70px 15px; color:#4b5363;">
                    <div style="font-size:2.2rem; margin-bottom:8px;">⚡</div>
                    <div style="font-family:'JetBrains Mono'; font-size:0.85rem; font-weight:700; color:#8b949e;">STANDBY MODE</div>
                    <div style="font-size:0.75rem; margin-top:6px;">Adjust parameters and click <b>Run Inference</b> to generate real-time churn telemetry.</div>
                </div>
            """, unsafe_allow_html=True)

# ============================================================
# SESSION HISTORY (optional, collapsible)
# ============================================================
if st.session_state.history:
    with st.expander(f"📜 Session history ({len(st.session_state.history)} predictions)"):
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)