# app.py – FULL, FINAL, PRODUCTION-READY VERSION
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from explainability import SimpleExplainer
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier
import torch
import torch.nn as nn

# --------------------------------------------------------------
# Helper: matplotlib → base64
# --------------------------------------------------------------
def fig_to_base64(fig):
    if fig is None:
        return None
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    facecolor='#0E1117', dpi=200)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    except Exception:
        if fig:
            plt.close(fig)
        return None

# --------------------------------------------------------------
# Helper: top5 vs stress image
# --------------------------------------------------------------
def display_top5_vs_stress(explainer, X, idx=0):
    impacts, fig = explainer.plot_top5_vs_stress(X, instance_idx=idx)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight',
                facecolor='#0E1117', dpi=200)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    st.image(f"data:image/png;base64,{img_base64}", use_column_width=True)
    return impacts

# --------------------------------------------------------------
# Streamlit config + CSS
# --------------------------------------------------------------
st.set_page_config(page_title="MindTrack Pro", page_icon="Brain",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
.stSidebar { background-color: #16181D; }
h1, h2, h3 { color: #00D4FF; font-family: 'Arial', sans-serif; }
.stButton>button { background-color: #00D4FF; color: #0E1117;
                   border-radius: 8px; font-weight: bold; }
.stButton>button:hover { background-color: #00B0D4; }
.stMetric { background-color: #1A1F2A; border: 1px solid #00D4FF;
            border-radius: 8px; padding: 10px; }
.risk-high { color: #FF3333; font-weight: bold; }
.risk-moderate { color: #FFB84D; font-weight: bold; }
.risk-low { color: #33FF33; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# MLP definition
# --------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

# --------------------------------------------------------------
# Load everything (cached)
# --------------------------------------------------------------
@st.cache_resource
def load_resources():
    pipeline = joblib.load('artifacts/mindtrack_pipeline.pkl')
    feature_names = json.load(open('artifacts/feature_names.json'))

    models = {}
    model_paths = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Random Forest': 'models/random_forest.pkl',
        'XGBoost': 'models/xgboost.pkl',
        'MLP (PyTorch)': 'models/mlp.pth'
    }

    for name, path in model_paths.items():
        try:
            if name == 'MLP (PyTorch)':
                mlp = MLP(len(feature_names))
                mlp.load_state_dict(torch.load(path, map_location='cpu'))
                mlp.eval()
                models[name] = mlp
            else:
                models[name] = joblib.load(path)
            st.success(f"{name} loaded")
        except Exception as e:
            st.warning(f"{name} load failed: {e}")
            models[name] = None

    # Train data for ensemble
    try:
        X_train = pd.read_csv('artifacts/X_train.csv')
        y_train = pd.read_csv('artifacts/y_train.csv').values.ravel()
        st.info("Train data loaded for ensemble")
    except Exception as e:
        st.warning(f"Train data missing: {e}")
        X_train = y_train = None

    # Ensemble (fit on train to avoid "not fitted")
    if (all(models.get(k) is not None for k in ['Logistic Regression',
                                                'Random Forest',
                                                'XGBoost'])
        and X_train is not None and y_train is not None):
        ensemble = VotingClassifier(
            estimators=[
                ('lr', models['Logistic Regression']),
                ('rf', models['Random Forest']),
                ('xgb', models['XGBoost'])
            ],
            voting='soft'
        )
        ensemble.fit(X_train, y_train)
        models['Ensemble (LR+RF+XGB)'] = ensemble
        st.success("Ensemble fitted on TRAIN data")
    else:
        st.warning("Ensemble skipped")

    explainer = SimpleExplainer()

    try:
        X_test = pd.read_csv('artifacts/X_test.csv')
        y_test = pd.read_csv('artifacts/y_test.csv').values.ravel()
    except Exception as e:
        st.warning(f"Test data: {e}")
        X_test = y_test = None

    return pipeline, models, feature_names, explainer, X_test, y_test, X_train, y_train

pipeline, models, feature_names, explainer, X_test_global, y_test_global, X_train_global, y_train_global = load_resources()

# --------------------------------------------------------------
# Background data (for explanations)
# --------------------------------------------------------------
@st.cache_data
def load_background():
    try:
        return pd.read_csv("artifacts/X_test.csv")
    except:
        return pd.DataFrame(columns=feature_names)

X_bg = load_background()

# --------------------------------------------------------------
# Preprocess user input – exactly matches training pipeline
# --------------------------------------------------------------
def preprocess_user_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # ---- defaults (never missing) ----
    defaults = {
        'tech_exposure': 5, 'exercise_frequency': 3, 'meditation_frequency': 1,
        'relationship_status': 'Single', 'treatment': 0, 'can_share_problems': 1,
        'perceived_stigma': 0, 'sought_help': 0, 'self_esteem': 7,
        'major_life_event': 0, 'trauma_history': 0, 'parenting_status': 'None',
        'insurance': 1, 'sleep_medication': 0, 'screen_addiction': 0,
        'drug_dependence': 0, 'regular_checkup': 1, 'mental_health_awareness': 1,
        'housing_status': 'Rented', 'substance_use': 'None', 'sleep_duration': 7.0,
        'social_interaction_score': 50, 'work_interference': 'Never',
        'support_network': 'Moderate', 'workplace_support': 'Moderate',
        'occupation': 'IT', 'diet_quality': 'Average', 'education_level': 'High School',
        'bmi': 25.0, 'sleep_quality': 7, 'financial_stress': 5, 'physical_activity': 4.0
    }
    for k, v in defaults.items():
        df[k] = df.get(k, v)

    # ---- boolean mapping ----
    bool_cols = [
        'family_history', 'mental_health_history', 'treatment', 'can_share_problems',
        'perceived_stigma', 'sought_help', 'anxiety_attacks', 'depression_episodes',
        'suicidal_thoughts', 'major_life_event', 'trauma_history', 'insurance',
        'sleep_medication', 'screen_addiction', 'drug_dependence', 'regular_checkup',
        'mental_health_awareness', 'pet_ownership', 'marital_conflict',
        'caretaking_duty', 'remote_work'
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # ---- gender ordinal ----
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2}).fillna(-1)

    # ---- BMI category ----
    def bmi_cat(bmi):
        try:
            b = float(bmi)
            if b < 18.5: return 'Underweight'
            elif b < 25: return 'Normal'
            elif b < 30: return 'Overweight'
            else: return 'Obese'
        except:
            return 'Normal'
    df['bmi_category'] = df['bmi'].apply(bmi_cat)

    # ---- one-hot with True/False suffix ----
    def one_hot_with_suffix(df, col, prefix):
        dummies = pd.get_dummies(df[col], prefix=prefix).astype(bool)
        expanded = pd.DataFrame()
        for c in dummies.columns:
            expanded[f"{c}_True"] = dummies[c].astype(int)
            expanded[f"{c}_False"] = (~dummies[c]).astype(int)
        return expanded

    for col, prefix in [
        ('bmi_category', 'bmi'), ('diet_quality', 'diet'),
        ('education_level', 'education'), ('housing_status', 'housing'),
        ('relationship_status', 'relationship')
    ]:
        df = pd.concat([df, one_hot_with_suffix(df, col, prefix)], axis=1)

    # ---- ordinal maps (from pipeline) ----
    df['support_network_ord'] = df['support_network'].map(pipeline['support_map']).fillna(-1)
    df['workplace_support_ord'] = df['workplace_support'].map(pipeline['wp_map']).fillna(-1)
    df['work_interference_ord'] = df['work_interference'].map(pipeline['wi_map']).fillna(-1)
    df['occupation_ord'] = df['occupation'].map(pipeline['occupation_ord']).fillna(-1)
    df['substance_use_ord'] = df['substance_use'].map(pipeline['substance_ord']).fillna(-1)
    df['parenting_status_ord'] = df['parenting_status'].map(pipeline['parenting_ord']).fillna(-1)

    # ---- engineered features ----
    df['sleep_efficiency'] = df['sleep_duration'] * (df['sleep_quality'] / 10)
    df['working_hours'] = df['working_hours'].replace(0, np.nan)
    df['work_life_balance_index'] = (1 / (df['stress_level'] + 1)) * (40 / df['working_hours'])
    df['work_life_balance_index'] = df['work_life_balance_index'].replace([np.inf, -np.inf], np.nan)
    df['work_life_balance_index'] = df['work_life_balance_index'].fillna(pipeline['wlb_median'])

    product = df['screen_time'] * df['tech_exposure']
    df['tech_exposure_index'] = (product - pipeline['tech_min']) / (pipeline['tech_max'] - pipeline['tech_min'] + 1e-5)

    df['interaction_feature'] = df['stress_level'] * df['work_interference_ord']

    # ---- drop originals ----
    drop = [
        'bmi', 'bmi_category', 'diet_quality', 'education_level', 'housing_status',
        'relationship_status', 'support_network', 'workplace_support',
        'work_interference', 'occupation', 'substance_use', 'parenting_status'
    ]
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True, errors='ignore')

    # ---- ensure exact feature order ----
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # ---- scaling (robust + standard) ----
    robust_cols = [c for c in [
        'monthly_income','screen_time','working_hours','stress_level',
        'physical_activity','exercise_frequency','meditation_frequency',
        'job_satisfaction','life_satisfaction','quality_of_friends','self_esteem'
    ] if c in df.columns]

    standard_cols = [c for c in df.columns
                     if c not in robust_cols and df[c].dtype in [np.float64, np.int64]]

    if robust_cols:
        df[robust_cols] = pipeline['scaler_robust'].transform(df[robust_cols])
    if standard_cols:
        df[standard_cols] = pipeline['scaler_standard'].transform(df[standard_cols])

    return df

# --------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------
def get_recommendations(risk_level, top_contributors):
    recs = {
        'low': ["Maintain habits!", "Mindfulness", "Check-ins"],
        'moderate': ["Monitor stress", "Exercise", "Talk"],
        'high': ["Seek help", "Hotline", "Support"]
    }
    base = recs.get(risk_level.lower(), [])
    pers = [f"Lower {f}" if v > 0 else f"Boost {f}" for f, v in top_contributors]
    return list(set(base + pers))

# --------------------------------------------------------------
# Global plots (cached)
# --------------------------------------------------------------
@st.cache_resource
def get_global_importance_fig():
    if X_bg.empty or models.get('XGBoost') is None:
        return None
    _, fig = explainer.global_importance(X_bg)
    return fig

@st.cache_resource
def get_global_signed_fig():
    if X_bg.empty or models.get('XGBoost') is None:
        return None
    _, fig = explainer.global_signed_importance(X_bg, n_samples=20, n_repeats=5)
    return fig

@st.cache_resource
def get_cm_fig():
    if X_test_global is None or models.get('XGBoost') is None:
        return None
    _, fig = explainer.get_metrics_and_cm(X_test_global, y_test_global)
    return fig

# --------------------------------------------------------------
# Multi-model metrics + ROC data
# --------------------------------------------------------------
@st.cache_resource
def get_multi_model_metrics():
    if X_test_global is None or y_test_global is None:
        return {}
    results = {}
    classes = ['Low', 'Moderate', 'High']
    for name, model in models.items():
        if model is None:
            continue
        try:
            if name == 'MLP (PyTorch)':
                X_tensor = torch.FloatTensor(X_test_global.values)
                with torch.no_grad():
                    out = model(X_tensor)
                    y_proba = out.numpy()
                    y_pred = np.argmax(y_proba, axis=1)
            elif name.startswith('Ensemble'):
                y_pred = model.predict(X_test_global)
                y_proba = model.predict_proba(X_test_global)
            else:
                y_pred = model.predict(X_test_global)
                y_proba = model.predict_proba(X_test_global) if hasattr(model, 'predict_proba') else None

            precision = precision_score(y_test_global, y_pred,
                                        average=None, labels=[0,1,2], zero_division=0)
            recall = recall_score(y_test_global, y_pred,
                                  average=None, labels=[0,1,2], zero_division=0)
            f1 = f1_score(y_test_global, y_pred,
                          average=None, labels=[0,1,2], zero_division=0)

            per_class = pd.DataFrame({
                'Class': classes,
                'Precision': [f"{p:.3f}" for p in precision],
                'Recall': [f"{r:.3f}" for r in recall],
                'F1-Score': [f"{f:.3f}" for f in f1]
            })

            macro_prec = precision_score(y_test_global, y_pred, average='macro', zero_division=0)
            macro_rec = recall_score(y_test_global, y_pred, average='macro', zero_division=0)
            macro_f1 = f1_score(y_test_global, y_pred, average='macro', zero_division=0)

            results[name] = {
                'accuracy': f"{accuracy_score(y_test_global, y_pred):.3f}",
                'macro_precision': f"{macro_prec:.3f}",
                'macro_recall': f"{macro_rec:.3f}",
                'macro_f1': f"{macro_f1:.3f}",
                'per_class': per_class,
                'y_true': y_test_global,
                'y_proba': y_proba,
                'y_pred': y_pred
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    return results

# --------------------------------------------------------------
# ROC curve plot (macro-averaged)
# --------------------------------------------------------------
@st.cache_resource
def plot_roc_curves(metrics_data):
    if not metrics_data:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             line=dict(color='gray', dash='dash'),
                             name='Chance'))

    colors = ['#00D4FF', '#FF3333', '#33FF33', '#FFB84D', '#9C27B0']
    color_idx = 0
    n_classes = 3
    y_true_bin = label_binarize(y_test_global, classes=[0, 1, 2])

    for name, data in metrics_data.items():
        if 'y_proba' not in data or data['y_proba'] is None:
            continue

        y_proba = data['y_proba']
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)

        fig.add_trace(go.Scatter(
            x=all_fpr, y=mean_tpr,
            mode='lines',
            line=dict(color=colors[color_idx % len(colors)], width=3),
            name=f"{name} (AUC = {roc_auc_macro:.3f})"
        ))
        color_idx += 1

    fig.update_layout(
        title="ROC Curves Comparison (Macro Avg AUC)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#1A1F2A",
        font_color="#FAFAFA",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )
    return fig

# --------------------------------------------------------------
# PERSISTENT HISTORY FUNCTIONS
# --------------------------------------------------------------
HISTORY_DIR = "history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "user_history.json")

def ensure_history_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)

def load_history():
    ensure_history_dir()
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                return data
        except Exception as e:
            st.warning(f"Could not read history: {e}")
    return []

def save_history(history):
    ensure_history_dir()
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

# Load persistent history at startup
if 'user_history' not in st.session_state:
    st.session_state.user_history = load_history()

# --------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------
st.sidebar.title("MindTrack Pro")
page = st.sidebar.radio("Navigate",
                        ["Home", "Risk Assessment", "Tracking",
                         "Global Insights", "About"])

# --------------------------------------------------------------
# PAGE: Home
# --------------------------------------------------------------
if page == "Home":
    st.markdown("<h1 style='text-align:center;color:#00D4FF;'>MindTrack Pro</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background:#1A1F2A;padding:25px;border-radius:12px;"
        "border:1px solid #00D4FF;text-align:center;'>"
        "<h3 style='color:#00D4FF;'>AI Mental Health Guardian</h3>"
        "<p>5 Models | Ensemble Boost | Persistent Tracking</p></div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    ### Why Mental Health Matters Today
    In today's fast-paced world, mental health issues have become increasingly prevalent...
    """)

# --------------------------------------------------------------
# PAGE: Risk Assessment
# --------------------------------------------------------------
elif page == "Risk Assessment":
    st.header("Personal Risk Assessment")
    st.markdown("Answer the questions below to get a **personalized mental health risk prediction**. All data stays **local** and is **saved for tracking**.")

    with st.form("assessment_form"):
        tabs = st.tabs(["Personal", "Work", "Lifestyle", "MH History", "Social"])

        # ---- Personal ----
        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 18, 100, 35)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                occupation = st.selectbox("Occupation",
                    ["IT", "Medical", "Education", "Engineering",
                     "Arts", "Business", "Unemployed", "Student"])
            with c2:
                monthly_income = st.number_input("Monthly Income ($)", 0, 300000, 60000, step=1000)
                education_level = st.selectbox("Education Level",
                    ["High School", "Diploma", "Graduate", "Postgraduate", "PhD"])

        # ---- Work ----
        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                working_hours = st.slider("Working Hours/Week", 0, 80, 40)
                remote_work = st.selectbox("Remote Work", ["No", "Yes"])
                work_interference = st.selectbox("Work Interferes with MH",
                    ["Never", "Rarely", "Sometimes", "Often", "Always"])
            with c2:
                job_satisfaction = st.slider("Job Satisfaction", 1, 10, 7)
                workplace_support = st.selectbox("Workplace MH Support",
                    ["Weak", "Moderate", "Strong"])

        # ---- Lifestyle ----
        with tabs[2]:
            c1, c2 = st.columns(2)
            with c1:
                stress_level = st.slider("Stress Level", 1, 10, 5)
                financial_stress = st.slider("Financial Stress", 1, 10, 5)
                sleep_duration = st.slider("Sleep Duration (hrs)", 3.0, 12.0, 7.0, step=0.5)
                sleep_quality = st.slider("Sleep Quality", 1, 10, 7)
            with c2:
                screen_time = st.slider("Daily Screen Time (hrs)", 0.0, 15.0, 5.0, step=0.5)
                tech_exposure = st.slider("Tech Exposure (apps, devices)", 1, 10, 5)
                physical_activity = st.slider("Exercise (hrs/week)", 0.0, 20.0, 4.0, step=0.5)
                bmi = st.number_input("BMI", 15.0, 50.0, 25.0, step=0.1)
                diet_quality = st.selectbox("Diet Quality",
                    ["Poor", "Average", "Good", "Excellent"])

        # ---- MH History ----
        with tabs[3]:
            c1, c2 = st.columns(2)
            with c1:
                family_history = st.selectbox("Family MH History", ["No", "Yes"])
                mental_health_history = st.selectbox("Personal MH History", ["No", "Yes"])
                treatment = st.selectbox("Ever in Treatment", ["No", "Yes"])
            with c2:
                anxiety_attacks = st.selectbox("Anxiety Attacks", ["No", "Yes"])
                depression_episodes = st.selectbox("Depression Episodes", ["No", "Yes"])
                suicidal_thoughts = st.selectbox("Suicidal Thoughts", ["No", "Yes"])

        # ---- Social ----
        with tabs[4]:
            c1, c2 = st.columns(2)
            with c1:
                support_network = st.selectbox("Support Network", ["Low", "Moderate", "High"])
                quality_of_friends = st.slider("Quality of Friends", 1, 10, 7)
                life_satisfaction = st.slider("Life Satisfaction", 1, 10, 7)
            with c2:
                pet_ownership = st.selectbox("Pet Ownership", ["No", "Yes"])
                marital_conflict = st.selectbox("Marital Conflict", ["No", "Yes"])
                caretaking_duty = st.selectbox("Caretaking Duty", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Risk", type="primary", use_container_width=True)

    if submitted:
        try:
            user_data = {
                'age': age,
                'gender': gender,
                'occupation': occupation,
                'monthly_income': monthly_income,
                'education_level': education_level,
                'working_hours': working_hours,
                'remote_work': 1 if remote_work == "Yes" else 0,
                'work_interference': work_interference,
                'job_satisfaction': job_satisfaction,
                'workplace_support': workplace_support,
                'stress_level': stress_level,
                'financial_stress': financial_stress,
                'sleep_duration': sleep_duration,
                'sleep_quality': sleep_quality,
                'screen_time': screen_time,
                'tech_exposure': tech_exposure,
                'physical_activity': physical_activity,
                'bmi': bmi,
                'diet_quality': diet_quality,
                'family_history': 1 if family_history == "Yes" else 0,
                'mental_health_history': 1 if mental_health_history == "Yes" else 0,
                'treatment': 1 if treatment == "Yes" else 0,
                'anxiety_attacks': 1 if anxiety_attacks == "Yes" else 0,
                'depression_episodes': 1 if depression_episodes == "Yes" else 0,
                'suicidal_thoughts': 1 if suicidal_thoughts == "Yes" else 0,
                'support_network': support_network,
                'quality_of_friends': quality_of_friends,
                'life_satisfaction': life_satisfaction,
                'pet_ownership': 1 if pet_ownership == "Yes" else 0,
                'marital_conflict': 1 if marital_conflict == "Yes" else 0,
                'caretaking_duty': 1 if caretaking_duty == "Yes" else 0
            }

            X_user = preprocess_user_input(user_data)

            # ---- prediction (prefer ensemble) ----
            ensemble = models.get('Ensemble (LR+RF+XGB)')
            if ensemble and X_train_global is not None:
                proba = ensemble.predict_proba(X_user)[0]
                model_used = "Ensemble (LR+RF+XGB)"
            else:
                proba = models['XGBoost'].predict_proba(X_user)[0]
                model_used = "XGBoost"

            pred = int(np.argmax(proba))
            risk = ["Low", "Moderate", "High"][pred]
            confidence = proba[pred]

            # ---- layout ----
            col1, col2, col3 = st.columns([1, 1, 2])

            # gauge
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence * 100,
                    number={'suffix': "%"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#33FF33" if pred == 0 else "#FFB84D" if pred == 1 else "#FF3333"},
                        'steps': [
                            {'range': [0, 33], 'color': '#33FF33'},
                            {'range': [33, 66], 'color': '#FFB84D'},
                            {'range': [66, 100], 'color': '#FF3333'}
                        ]
                    },
                    title={'text': "Risk Probability"}
                ))
                fig.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA", height=220)
                st.plotly_chart(fig, use_container_width=True)

            # risk label
            with col2:
                st.markdown(f"<h2 class='risk-{risk.lower()}'>{risk} Risk</h2>", unsafe_allow_html=True)
                st.metric("Confidence", f"{confidence:.1%}")
                st.caption(f"Model: **{model_used}**")

            # ---- explanations (combine background + user) ----
            X_full = pd.concat([X_bg, X_user], ignore_index=True) if not X_bg.empty else X_user
            instance_idx = len(X_full) - 1

            # local explanation
            top_contributors, contrib_fig = explainer.local_explanation(
                X_full, instance_idx=instance_idx, top_n=5
            )

            with col3:
                st.subheader("Top 5 Drivers")
                for i, (f, v) in enumerate(top_contributors, 1):
                    color = "#FF3333" if v > 0 else "#33FF33"
                    st.markdown(
                        f"{i}. **{f}** <span style='color:{color};'>[{v:+.4f}]</span>",
                        unsafe_allow_html=True
                    )

                img_data = fig_to_base64(contrib_fig)
                if img_data:
                    st.image(f"data:image/png;base64,{img_data}",
                             caption="Local Feature Impact")
                else:
                    st.warning("Explanation plot failed.")

                # top-5 vs stress
                st.subheader("How Your Top 5 Affect Risk (Your Stress Fixed)")
                _, top5_fig = explainer.plot_top5_vs_stress(
                    X_full, instance_idx=instance_idx, top_n=5
                )
                buf = BytesIO()
                top5_fig.savefig(buf, format="png", bbox_inches='tight',
                                 facecolor='#0E1117', dpi=200)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode()
                st.image(f"data:image/png;base64,{img_base64}", use_column_width=True)

            # ---- recommendations ----
            st.subheader("Personalized Recommendations")
            recs = get_recommendations(risk, top_contributors)
            cols = st.columns(3)
            for i, rec in enumerate(recs):
                with cols[i % 3]:
                    st.markdown(
                        f"<div style='background:#1A1F2A;padding:12px;"
                        "border-radius:8px;border:1px solid #00D4FF;"
                        "height:100px;overflow:auto;'>{rec}</div>",
                        unsafe_allow_html=True
                    )

            # ---- SAVE TO PERSISTENT HISTORY ----
            new_entry = {
                'date': datetime.now().isoformat(),
                'risk': risk,
                'proba': float(confidence),
                'top': top_contributors,
                'model': model_used
            }
            st.session_state.user_history.append(new_entry)
            save_history(st.session_state.user_history)

            st.success("Assessment saved to **persistent** history!")
            st.balloons()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Check that all models and artifacts are present.")

# --------------------------------------------------------------
# PAGE: Tracking (now uses persistent history)
# --------------------------------------------------------------
# --------------------------------------------------------------
# PAGE: Tracking – CURVED FORECAST (Moderate → Low → High possible)
# --------------------------------------------------------------
elif page == "Tracking":
    st.header("Risk Trend (Persistent)")

    if not st.session_state.user_history:
        st.info("No assessments yet. Complete one to start tracking.")
    else:
        # ------------------------------------------------------------------
        # 1. Prepare history
        # ------------------------------------------------------------------
        df_hist = pd.DataFrame(st.session_state.user_history)
        df_hist["date"] = pd.to_datetime(df_hist["date"])
        df_hist = df_hist.sort_values("date").reset_index(drop=True)
        df_hist["risk_num"] = df_hist["risk"].map({"Low": 0, "Moderate": 1, "High": 2})

        # ------------------------------------------------------------------
        # 2. Plot base line + moving average
        # ------------------------------------------------------------------
        fig_line = go.Figure()

        fig_line.add_trace(
            go.Scatter(
                x=df_hist["date"],
                y=df_hist["risk_num"],
                mode="lines+markers",
                name="Risk Level",
                line=dict(color="#00D4FF", width=3),
                marker=dict(size=10, color="#00D4FF"),
            )
        )

        # 3-day moving average
        if len(df_hist) >= 3:
            df_hist["ma_3"] = df_hist["risk_num"].rolling(window=3, min_periods=1).mean()
            fig_line.add_trace(
                go.Scatter(
                    x=df_hist["date"],
                    y=df_hist["ma_3"],
                    mode="lines",
                    name="3-Day MA",
                    line=dict(dash="dash", color="#FFB84D", width=2),
                )
            )

        # ------------------------------------------------------------------
        # 3. CURVED FORECAST – requires ≥3 points
        # ------------------------------------------------------------------
        if len(df_hist) >= 3:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline

            X = np.arange(len(df_hist)).reshape(-1, 1)
            y = df_hist["risk_num"].values

            # recent points get higher weight
            weights = np.exp(np.linspace(0, 1.5, len(df_hist)))

            # quadratic (degree=2) → can bend up/down
            poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
            poly_model.fit(X, y, linearregression__sample_weight=weights)

            # predict next 3 days
            future_idx = np.array([[len(df_hist)], [len(df_hist) + 1], [len(df_hist) + 2]])
            future_raw = poly_model.predict(future_idx)
            future_pred = np.clip(np.round(future_raw, 2), 0, 2)   # clamp to [0,2]

            # future dates
            last_date = df_hist["date"].iloc[-1]
            future_dates = [
                last_date + pd.Timedelta(days=1),
                last_date + pd.Timedelta(days=2),
                last_date + pd.Timedelta(days=3),
            ]

            # risk level labels for hover & annotation
            risk_labels = ["Low", "Moderate", "High"]
            pred_labels = [
                risk_labels[int(np.clip(round(p), 0, 2))] for p in future_pred
            ]

            # add forecast trace
            fig_line.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=future_pred,
                    mode="lines+markers",
                    name="Forecast (Day +1 to +3)",
                    line=dict(color="#FF3333", width=4, dash="dot"),
                    marker=dict(
                        symbol="diamond",
                        size=12,
                        color="#FF3333",
                        line=dict(width=2, color="white"),
                    ),
                    text=[f"Day +{i+1}: {pred_labels[i]}" for i in range(3)],
                    hovertemplate="<b>%{text}</b><br>Risk: %{y:.2f}<extra></extra>",
                )
            )

            # BIG RED ALERT if Day +3 is High
            if future_pred[-1] >= 1.7:
                fig_line.add_annotation(
                    x=future_dates[-1],
                    y=future_pred[-1],
                    text="HIGH RISK AHEAD!",
                    showarrow=True,
                    arrowhead=3,
                    ax=50,
                    ay=-50,
                    bgcolor="#FF3333",
                    font=dict(color="white", size=13, family="Arial Black"),
                    bordercolor="white",
                    borderwidth=2,
                    borderpad=6,
                )
        else:
            st.info(
                "Need **3+ assessments** to show a curved forecast "
                "(e.g., Moderate → Low → High)."
            )

        # ------------------------------------------------------------------
        # 4. Layout
        # ------------------------------------------------------------------
        fig_line.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1A1F2A",
            font_color="#FAFAFA",
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=["Low", "Moderate", "High"],
                range=[-0.3, 2.3],
            ),
            xaxis=dict(showgrid=True, gridcolor="#2D323B"),
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            title="Risk Trend with 3-Day Forecast",
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # ------------------------------------------------------------------
        # 5. Trend Summary
        # ------------------------------------------------------------------
        st.subheader("Trend Summary")
        if len(df_hist) >= 2:
            delta = df_hist["risk_num"].iloc[-1] - df_hist["risk_num"].iloc[0]
            if delta < 0:
                st.success(f"Risk decreased by {abs(delta)} level(s) since first assessment!")
            elif delta > 0:
                st.warning(f"Risk increased by {delta} level(s) since first assessment.")
            else:
                st.info("Risk level has been stable.")
            st.metric("Average Confidence", f"{df_hist['proba'].mean():.1%}")

        # ------------------------------------------------------------------
        # 6. History Table
        # ------------------------------------------------------------------
        display_df = df_hist[["date", "risk", "proba", "model"]].copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["proba"] = display_df["proba"].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True)

        # ------------------------------------------------------------------
        # 7. Download CSV
        # ------------------------------------------------------------------
        csv = display_df.to_csv(index=False).encode()
        st.download_button(
            "Download History CSV",
            csv,
            f"mindtrack_history_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
        )

        # ------------------------------------------------------------------
        # 8. Clear History
        # ------------------------------------------------------------------
        if st.button("Clear All History"):
            st.session_state.user_history = []
            save_history([])
            st.success("History cleared!")
            st.rerun()

# --------------------------------------------------------------
# PAGE: Global Insights
# --------------------------------------------------------------
elif page == "Global Insights":
    st.header("Model Insights")
    if X_test_global is None:
        st.warning("No test data.")
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Importance", "Signed Impact", "Metrics",
            "Confusion Matrix", "ROC Curves"
        ])

        with tab1:
            fig = get_global_importance_fig()
            img = fig_to_base64(fig)
            if img:
                st.image(f"data:image/png;base64,{img}",
                         caption="XGBoost Global Importance")
            else:
                st.warning("Plot failed.")

        with tab2:
            fig = get_global_signed_fig()
            img = fig_to_base64(fig)
            if img:
                st.image(f"data:image/png;base64,{img}",
                         caption="XGBoost Signed Impacts")
            else:
                st.warning("Plot failed.")

        with tab3:
            st.subheader("5-Model Comparison (+ Ensemble)")
            metrics_data = get_multi_model_metrics()
            if not metrics_data:
                st.warning("No models.")
            else:
                df_metrics = pd.DataFrame.from_dict(metrics_data, orient='index')[
                    ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
                ]
                df_metrics = df_metrics.astype(float)
                df_metrics.index.name = 'Model'

                fig_bar = go.Figure()
                colors_bar = ['#00D4FF', '#FF3333', '#33FF33', '#FFB84D']
                for i, col in enumerate(df_metrics.columns):
                    fig_bar.add_trace(go.Bar(
                        x=df_metrics.index,
                        y=df_metrics[col],
                        name=col.replace('_', ' ').capitalize(),
                        marker_color=colors_bar[i]
                    ))

                fig_bar.update_layout(
                    barmode='group',
                    title='Model Performance Comparison (Macro Metrics)',
                    xaxis_title='Models',
                    yaxis_title='Score',
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#1A1F2A",
                    font_color="#FAFAFA",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                model_tabs = st.tabs(metrics_data.keys())
                for idx, name in enumerate(metrics_data.keys()):
                    with model_tabs[idx]:
                        d = metrics_data[name]
                        if 'error' in d:
                            st.error(f"Error: {d['error']}")
                            continue
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("Accuracy", d['accuracy'])
                        with col2:
                            st.markdown("**Macro**")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("P", d['macro_precision'])
                            c2.metric("R", d['macro_recall'])
                            c3.metric("F1", d['macro_f1'])
                        st.markdown("**Per-Class**")
                        st.dataframe(d['per_class'].style.background_gradient(cmap='Blues'),
                                     use_container_width=True, hide_index=True)

        with tab4:
            fig = get_cm_fig()
            img = fig_to_base64(fig)
            if img:
                st.image(f"data:image/png;base64,{img}",
                         caption="XGBoost Confusion Matrix")
            else:
                st.warning("CM failed.")

        with tab5:
            st.subheader("ROC Curves (One-vs-Rest, Macro Average)")
            roc_fig = plot_roc_curves(metrics_data)
            if roc_fig:
                st.plotly_chart(roc_fig, use_container_width=True)
                st.caption("Each line shows macro-averaged ROC across 3 classes. Higher AUC = better discrimination.")
            else:
                st.warning("ROC curves could not be generated.")

# --------------------------------------------------------------
# PAGE: About
# --------------------------------------------------------------
elif page == "About":
    st.header("About MindTrack Pro")
    st.markdown("""
    ### Project Abstract: MindTrack Pro - A Comprehensive AI-Driven Mental Health Risk Assessment System
    
    **Overview and Creation Process**  
    MindTrack Pro is an end-to-end machine learning application designed to assess mental health risks based on a multifaceted dataset encompassing lifestyle, socioeconomic, and psychological factors. Built using Python and Streamlit for the user interface, the project integrates data preprocessing, model training, ensemble prediction, explainable AI (XAI), and visualization components. The system was created in phases: (1) Data preprocessing to handle raw inputs from 'mindtrack_dataset.csv'; (2) Training multiple ML models with resampling for class imbalance; (3) Implementing perturbation-based explainability for local and global insights; (4) Developing a Streamlit app for interactive assessment, tracking, and insights; and (5) Ensuring robustness with fixes like explicit base_score handling in XGBoost to prevent serialization issues.
    
    **Dataset and Preprocessing Details (preprocess.py)**  
    The dataset ('mindtrack_dataset.csv') contains ~1000+ records with 50+ features like age, gender, stress_level, sleep_duration, bmi, substance_use, and target 'mental_health_risk' (low/moderate/high). Preprocessing starts by loading the CSV via pandas and dropping rows with missing targets. Boolean columns (e.g., family_history) are mapped to 0/1. Numeric imputation uses SimpleImputer with median strategy for columns like monthly_income and bmi. Categorical imputation fills with 'missing'. Gender is ordinal-encoded (Male:0, Female:1, Other:2). BMI is categorized into 'Underweight', 'Normal', 'Overweight', 'Obese' using thresholds: bmi < 18.5 → Underweight; 18.5 ≤ bmi < 25 → Normal; 25 ≤ bmi < 30 → Overweight; bmi ≥ 30 → Obese.  
    Feature engineering includes:  
    - sleep_efficiency = sleep_duration * (sleep_quality / 10)  
    - work_life_balance_index = (1 / (stress_level + 1)) * (40 / working_hours), with inf/NaN replaced by median  
    - tech_exposure_index = (screen_time * tech_exposure - min_product) / (max_product - min_product + 1e-5) for normalization [0-1]  
    - interaction_feature = stress_level * work_interference_ord  
    Ordinal mappings: work_interference ('Never':0 to 'Always':4), support_network ('Low':0 to 'High':2), etc. One-hot encoding with True/False suffixes for categories like bmi_category (e.g., bmi_Normal_True, bmi_Normal_False). Original categoricals are dropped. Scaling: RobustScaler for skewed columns (e.g., monthly_income), StandardScaler for others. Data is split 80/20 with stratification, saved as CSVs in 'artifacts/', and pipeline artifacts (imputers, scalers, maps) dumped via joblib/JSON.
    
    **Model Training and Calculations (train_model.py)**  
    Models are trained on resampled data using SMOTE to handle class imbalance (oversampling minority classes). Four base models:  
    - **Logistic Regression**: GridSearchCV with C=[0.1,1,10], max_iter=[500,1000], class_weight='balanced'. Fits linear coefficients for multi-class (one-vs-rest internally).  
    - **Random Forest**: GridSearchCV with n_estimators=[100,200], max_depth=[10,20], class_weight='balanced'. Uses Gini impurity for splits: Gini = 1 - Σ(p_i^2), averaged over trees.  
    - **XGBoost**: GridSearchCV with n_estimators=[200,300], learning_rate=[0.01,0.1], objective='multi:softprob', base_score=0.5 (explicitly set to avoid array issues). Boosting gradients: loss = -Σ y_i log(p_i), with regularization. Post-fit, booster attributes set for compatibility.  
    - **MLP (PyTorch)**: 3-layer neural net (input→128→128→3) with ReLU and softmax. Trained 50 epochs, batch=32, Adam optimizer (lr=0.001), CrossEntropyLoss = -Σ y_i log(p_i).  
    Evaluation metrics: Accuracy = TP+TN / total; Precision = TP / (TP+FP); Recall = TP / (TP+FN); F1 = 2*(P*R)/(P+R) (macro-averaged); AUC-OVR = area under ROC for each class vs rest; MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)). Models saved via joblib/torch.save in 'models/'.
    
    **Ensemble and Prediction**  
    A soft-voting ensemble (VotingClassifier) combines LR, RF, XGBoost, fitted on full train data (X_train, y_train) to avoid 'not fitted' errors. Prediction: average class probabilities across models, argmax for class (0=Low,1=Moderate,2=High). User inputs are preprocessed similarly (with defaults for missing values) and concatenated to background data for contextual explanations.
    
    **Explainability (explainability.py)**  
    Uses a custom SimpleExplainer class on XGBoost.  
    - **Global Importance**: Feature importances from model.feature_importances_ (gain-based: total reduction in loss per feature). Sorted and plotted top-10.  
    - **Global Signed Impacts**: Averages local perturbations over n_samples=20 instances, each with n_repeats=5. For each feature, mean delta = baseline_proba - perturbed_proba.  
    - **Local Explanation**: For a instance, baseline_proba = model.predict_proba(instance)[pred_class]. Perturb each feature n_repeats=5 times (random sample from column or Gaussian noise if small data), compute delta = baseline - new_proba. Sort by abs(delta) for top impacts.  
    Plots use matplotlib with dark theme, saved as PNGs, closed via decorator to prevent leaks. Metrics/CM also computed/plotted.
    
    **Streamlit App (app.py)**  
    Pages: Assessment (form-based prediction with gauge chart, top drivers, recommendations); Tracking (line chart with MA-3, linear forecast via LinearRegression: y = β0 + β1*x, extended 2 points); Global Insights (tabs for importances, impacts, multi-model metrics, CM); About (this abstract). Uses Plotly for interactives, base64 for matplotlib images. Session state tracks history. Resources cached for efficiency.
    
    **Tools and Libraries Used**  
    Pandas/Numpy for data; Scikit-learn for models/imbalance/metrics; XGBoost/PyTorch for advanced ML; Matplotlib/Plotly for viz; Streamlit for UI; Joblib/JSON/Torch for serialization. No external APIs; all local.
    
    **Necessity and Impact**  
    MindTrack Pro addresses mental health gaps by providing explainable, multi-model predictions, fostering trust and action. It calculates risks with high accuracy (e.g., ~0.8+ F1 in tests) while detailing 'why' via perturbations, enabling users to focus on modifiable factors like sleep or stress.
    """, unsafe_allow_html=True)