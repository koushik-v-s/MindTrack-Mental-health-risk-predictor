# app.py (ULTIMATE FIXED: Ensemble fits on TRAIN data - NO "not fitted" ERROR)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier
import torch
import torch.nn as nn

st.set_page_config(page_title="MindTrack Pro", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# Dark Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stSidebar { background-color: #16181D; }
    h1, h2, h3 { color: #00D4FF; font-family: 'Arial', sans-serif; }
    .stButton>button { background-color: #00D4FF; color: #0E1117; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #00B0D4; }
    .stMetric { background-color: #1A1F2A; border: 1px solid #00D4FF; border-radius: 8px; padding: 10px; }
    .risk-high { color: #FF3333; font-weight: bold; }
    .risk-moderate { color: #FFB84D; font-weight: bold; }
    .risk-low { color: #33FF33; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# MLP Architecture
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
                input_size = len(feature_names)
                mlp = MLP(input_size)
                mlp.load_state_dict(torch.load(path, map_location='cpu'))
                mlp.eval()
                models[name] = mlp
            else:
                models[name] = joblib.load(path)
            st.success(f"✅ {name} loaded")
        except Exception as e:
            st.warning(f"⚠️ {name} load failed: {e}")
            models[name] = None
    
    # Load TRAIN data for ensemble fit
    try:
        X_train = pd.read_csv('artifacts/X_train.csv')
        y_train = pd.read_csv('artifacts/y_train.csv').values.ravel()
        st.info("✅ Train data loaded for ensemble")
    except Exception as e:
        st.warning(f"⚠️ Train data missing: {e}")
        X_train = y_train = None
    
    # ENSEMBLE: Fit on TRAIN if possible
    if (all(models.get(k) is not None for k in ['Logistic Regression', 'Random Forest', 'XGBoost'])
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
        st.success("✅ Ensemble fitted on TRAIN data")
    else:
        st.warning("⚠️ Ensemble skipped")
    
    explainer = SimpleExplainer()
    
    try:
        X_test = pd.read_csv('artifacts/X_test.csv')
        y_test = pd.read_csv('artifacts/y_test.csv').values.ravel()
    except Exception as e:
        st.warning(f"Test data: {e}")
        X_test = y_test = None
    
    return pipeline, models, feature_names, explainer, X_test, y_test, X_train, y_train

pipeline, models, feature_names, explainer, X_test_global, y_test_global, X_train_global, y_train_global = load_resources()

@st.cache_data
def load_background():
    try:
        return pd.read_csv("artifacts/X_test.csv")
    except:
        return pd.DataFrame(columns=feature_names)

X_bg = load_background()

def preprocess_user_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
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

    bool_cols = ['family_history', 'mental_health_history', 'treatment', 'can_share_problems',
                 'perceived_stigma', 'sought_help', 'anxiety_attacks', 'depression_episodes',
                 'suicidal_thoughts', 'major_life_event', 'trauma_history', 'insurance',
                 'sleep_medication', 'screen_addiction', 'drug_dependence', 'regular_checkup',
                 'mental_health_awareness', 'pet_ownership', 'marital_conflict',
                 'caretaking_duty', 'remote_work']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)

    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2}).fillna(-1)

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

    def one_hot_with_suffix(df, col, prefix):
        dummies = pd.get_dummies(df[col], prefix=prefix).astype(bool)
        expanded = pd.DataFrame()
        for c in dummies.columns:
            expanded[f"{c}_True"] = dummies[c].astype(int)
            expanded[f"{c}_False"] = (~dummies[c]).astype(int)
        return expanded

    for col, prefix in [('bmi_category', 'bmi'), ('diet_quality', 'diet'),
                        ('education_level', 'education'), ('housing_status', 'housing'),
                        ('relationship_status', 'relationship')]:
        df = pd.concat([df, one_hot_with_suffix(df, col, prefix)], axis=1)

    df['support_network_ord'] = df['support_network'].map(pipeline['support_map']).fillna(-1)
    df['workplace_support_ord'] = df['workplace_support'].map(pipeline['wp_map']).fillna(-1)
    df['work_interference_ord'] = df['work_interference'].map(pipeline['wi_map']).fillna(-1)
    df['occupation_ord'] = df['occupation'].map(pipeline['occupation_ord']).fillna(-1)
    df['substance_use_ord'] = df['substance_use'].map(pipeline['substance_ord']).fillna(-1)
    df['parenting_status_ord'] = df['parenting_status'].map(pipeline['parenting_ord']).fillna(-1)

    df['sleep_efficiency'] = df['sleep_duration'] * (df['sleep_quality'] / 10)
    df['working_hours'] = df['working_hours'].replace(0, np.nan)
    df['work_life_balance_index'] = (1 / (df['stress_level'] + 1)) * (40 / df['working_hours'])
    df['work_life_balance_index'] = df['work_life_balance_index'].replace([np.inf, -np.inf], np.nan).fillna(pipeline['wlb_median'])
    product = df['screen_time'] * df['tech_exposure']
    df['tech_exposure_index'] = (product - pipeline['tech_min']) / (pipeline['tech_max'] - pipeline['tech_min'] + 1e-5)
    df['interaction_feature'] = df['stress_level'] * df['work_interference_ord']

    drop = ['bmi', 'bmi_category', 'diet_quality', 'education_level', 'housing_status',
            'relationship_status', 'support_network', 'workplace_support',
            'work_interference', 'occupation', 'substance_use', 'parenting_status']
    df.drop(columns=[c for c in drop if c in df.columns], inplace=True, errors='ignore')

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    robust_cols = [c for c in ['monthly_income','screen_time','working_hours','stress_level',
                               'physical_activity','exercise_frequency','meditation_frequency',
                               'job_satisfaction','life_satisfaction','quality_of_friends','self_esteem'] if c in df.columns]
    standard_cols = [c for c in df.columns if c not in robust_cols and df[c].dtype in [np.float64, np.int64]]
    if robust_cols:
        df[robust_cols] = pipeline['scaler_robust'].transform(df[robust_cols])
    if standard_cols:
        df[standard_cols] = pipeline['scaler_standard'].transform(df[standard_cols])

    return df

def fig_to_base64(fig):
    if fig is None:
        return None
    try:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#0E1117', dpi=200)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    except Exception as e:
        if fig:
            plt.close(fig)
        return None

def get_recommendations(risk_level, top_contributors):
    recs = {
        'low': ["🛡️ Maintain habits!", "🧘 Mindfulness", "📅 Check-ins"],
        'moderate': ["⚠️ Monitor stress", "🏃 Exercise", "🗣️ Talk"],
        'high': ["🚨 Seek help", "📞 Hotline", "🤝 Support"]
    }
    base = recs.get(risk_level.lower(), [])
    pers = [f"📉 Lower {f}" if v > 0 else f"📈 Boost {f}" for f, v in top_contributors]
    return list(set(base + pers))

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
                    outputs = model(X_tensor)
                    y_proba = outputs.numpy()
                    y_pred = np.argmax(y_proba, axis=1)
            elif name.startswith('Ensemble'):
                y_pred = model.predict(X_test_global)
                y_proba = model.predict_proba(X_test_global)
            else:
                y_pred = model.predict(X_test_global)
                y_proba = model.predict_proba(X_test_global) if hasattr(model, 'predict_proba') else None
            
            precision = precision_score(y_test_global, y_pred, average=None, labels=[0,1,2], zero_division=0)
            recall = recall_score(y_test_global, y_pred, average=None, labels=[0,1,2], zero_division=0)
            f1 = f1_score(y_test_global, y_pred, average=None, labels=[0,1,2], zero_division=0)
            
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
                'per_class': per_class
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    return results

if 'user_history' not in st.session_state:
    st.session_state.user_history = []

st.sidebar.title("🧠 MindTrack Pro")
page = st.sidebar.radio("Navigate", ["Home", "Risk Assessment", "Tracking", "Global Insights", "About"])

if page == "Home":
    st.markdown("<h1 style='text-align:center;color:#00D4FF;'>MindTrack Pro 🧠✨</h1>", unsafe_allow_html=True)
    st.markdown("<div style='background:#1A1F2A;padding:25px;border-radius:12px;border:1px solid #00D4FF;text-align:center;'><h3 style='color:#00D4FF;'>AI Mental Health Guardian</h3><p>5 Models | Ensemble Boost | Full Metrics</p></div>", unsafe_allow_html=True)

elif page == "Risk Assessment":
    st.header("Personal Risk Assessment 📊")
    with st.form("assessment_form"):
        tabs = st.tabs(["Personal", "Work", "Lifestyle", "MH History", "Social"])
        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 18, 100, 30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                occupation = st.selectbox("Occupation", ["IT", "Medical", "Education", "Engineering", "Arts", "Business", "Unemployed", "Student"])
            with c2:
                monthly_income = st.number_input("Income ($)", 0, 200000, 50000)
                education_level = st.selectbox("Education", ["High School", "Diploma", "Graduate", "Postgraduate", "PhD"])
        with tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                working_hours = st.slider("Hours/Week", 0, 80, 40)
                remote_work = st.selectbox("Remote", ["No", "Yes"])
                work_interference = st.selectbox("Interference", ["Never", "Rarely", "Sometimes", "Often", "Always"])
            with c2:
                job_satisfaction = st.slider("Job Sat", 1, 10, 7)
                workplace_support = st.selectbox("Support", ["Weak", "Moderate", "Strong"])
        with tabs[2]:
            c1, c2 = st.columns(2)
            with c1:
                stress_level = st.slider("Stress", 1, 10, 5)
                financial_stress = st.slider("Fin Stress", 1, 10, 5)
                sleep_quality = st.slider("Sleep Qual", 1, 10, 7)
            with c2:
                screen_time = st.slider("Screen (hrs)", 0.0, 15.0, 5.0)
                physical_activity = st.slider("Exercise (hrs/wk)", 0.0, 20.0, 4.0)
                bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
                diet_quality = st.selectbox("Diet", ["Poor", "Average", "Good", "Excellent"])
        with tabs[3]:
            c1, c2 = st.columns(2)
            with c1:
                family_history = st.selectbox("Family MH", ["No", "Yes"])
                mental_health_history = st.selectbox("Personal MH", ["No", "Yes"])
                treatment = st.selectbox("Treatment", ["No", "Yes"])
            with c2:
                anxiety_attacks = st.selectbox("Anxiety", ["No", "Yes"])
                depression_episodes = st.selectbox("Depression", ["No", "Yes"])
                suicidal_thoughts = st.selectbox("Suicidal", ["No", "Yes"])
        with tabs[4]:
            c1, c2 = st.columns(2)
            with c1:
                support_network = st.selectbox("Support Net", ["Low", "Moderate", "High"])
                quality_of_friends = st.slider("Friend Qual", 1, 10, 7)
                life_satisfaction = st.slider("Life Sat", 1, 10, 7)
            with c2:
                pet_ownership = st.selectbox("Pet", ["No", "Yes"])
                marital_conflict = st.selectbox("Marital", ["No", "Yes"])
                caretaking_duty = st.selectbox("Caretaking", ["No", "Yes"])
        submitted = st.form_submit_button("🔮 Predict Risk", type="primary")

    if submitted:
        user_data = {
            'age': age, 'gender': gender, 'occupation': occupation,
            'monthly_income': monthly_income, 'education_level': education_level,
            'working_hours': working_hours, 'remote_work': 1 if remote_work == "Yes" else 0,
            'work_interference': work_interference, 'job_satisfaction': job_satisfaction,
            'workplace_support': workplace_support, 'stress_level': stress_level,
            'financial_stress': financial_stress, 'sleep_quality': sleep_quality,
            'screen_time': screen_time, 'physical_activity': physical_activity,
            'bmi': bmi, 'diet_quality': diet_quality, 'family_history': family_history,
            'mental_health_history': mental_health_history, 'treatment': 1 if treatment == "Yes" else 0,
            'anxiety_attacks': 1 if anxiety_attacks == "Yes" else 0,
            'depression_episodes': 1 if depression_episodes == "Yes" else 0,
            'suicidal_thoughts': 1 if suicidal_thoughts == "Yes" else 0,
            'support_network': support_network, 'quality_of_friends': quality_of_friends,
            'life_satisfaction': life_satisfaction, 'pet_ownership': 1 if pet_ownership == "Yes" else 0,
            'marital_conflict': 1 if marital_conflict == "Yes" else 0,
            'caretaking_duty': 1 if caretaking_duty == "Yes" else 0
        }
        try:
            X_user = preprocess_user_input(user_data)
            
            # ENSEMBLE PREDICTION
            ensemble_model = models.get('Ensemble (LR+RF+XGB)')
            if ensemble_model:
                proba = ensemble_model.predict_proba(X_user)[0]
                model_used = "Ensemble"
            else:
                proba = models['XGBoost'].predict_proba(X_user)[0]
                model_used = "XGBoost"
            pred = int(np.argmax(proba))
            risk = ["Low", "Moderate", "High"][pred]

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba[pred]*100,
                    gauge={'axis': {'range': [0,100]}, 'bar': {'color': "#33FF33" if pred==0 else "#FFB84D" if pred==1 else "#FF3333"}}
                ))
                fig_gauge.update_layout(paper_bgcolor="#0E1117", font_color="#FAFAFA")
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col2:
                st.markdown(f"<h2 class='risk-{risk.lower()}'>{risk} Risk</h2>", unsafe_allow_html=True)
                st.metric("Confidence", f"{proba[pred]:.1%}")
                st.caption(f"Model: {model_used}")

            X_full = pd.concat([X_bg, X_user], ignore_index=True) if not X_bg.empty else X_user
            instance_idx = len(X_full) - 1
            top_contributors, contrib_fig = explainer.local_explanation(X_full, instance_idx=instance_idx, top_n=5)

            with col3:
                st.subheader("🔍 Top 5 Drivers")
                for i, (f, v) in enumerate(top_contributors, 1):
                    color = "#FF3333" if v > 0 else "#33FF33"
                    st.markdown(f"{i}. **{f}** <span style='color:{color};'>[{v:+.4f}]</span>", unsafe_allow_html=True)
                img_data = fig_to_base64(contrib_fig)
                if img_data:
                    st.image(f"data:image/png;base64,{img_data}", caption="Local Contributions")
                else:
                    st.warning("Plot failed.")

            st.subheader("💡 Recommendations")
            recs = get_recommendations(risk, top_contributors)
            cols = st.columns(3)
            for i, r in enumerate(recs):
                with cols[i % 3]:
                    st.markdown(f"<div style='background:#1A1F2A;padding:12px;border-radius:8px;border:1px solid #00D4FF;'>{r}</div>", unsafe_allow_html=True)

            st.session_state.user_history.append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'risk': risk, 'proba': proba[pred], 'top': top_contributors,
                'model': model_used
            })
            st.balloons()
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Tracking":
    st.header("Risk Trend 📈")
    if not st.session_state.user_history:
        st.info("No assessments yet.")
    else:
        df_hist = pd.DataFrame(st.session_state.user_history)
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_hist = df_hist.sort_values('date').reset_index(drop=True)
        df_hist['risk_num'] = df_hist['risk'].map({"Low": 0, "Moderate": 1, "High": 2})
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_hist['date'], y=df_hist['risk_num'], mode='lines+markers',
            name='Risk Level', line=dict(color='#00D4FF', width=3), marker=dict(size=10)
        ))
        
        if len(df_hist) >= 3:
            df_hist['ma_3'] = df_hist['risk_num'].rolling(window=3, min_periods=1).mean()
            fig_line.add_trace(go.Scatter(
                x=df_hist['date'], y=df_hist['ma_3'], mode='lines',
                name='3-MA', line=dict(dash='dash', color='#FFB84D')
            ))
        
        if len(df_hist) >= 2:
            X = np.arange(len(df_hist)).reshape(-1, 1)
            y = df_hist['risk_num'].values
            trend = LinearRegression().fit(X, y)
            future = np.array([[len(df_hist)], [len(df_hist)+1]])
            pred = trend.predict(future)
            last_date = df_hist['date'].iloc[-1]
            future_dates = [last_date + pd.Timedelta(days=7), last_date + pd.Timedelta(days=14)]
            fig_line.add_trace(go.Scatter(
                x=future_dates, y=pred, mode='lines+markers',
                name='Forecast', line=dict(dash='dot', color='#FF3333'), marker=dict(symbol='diamond')
            ))
        
        fig_line.update_layout(
            paper_bgcolor="#0E1117", plot_bgcolor="#1A1F2A", font_color="#FAFAFA",
            yaxis=dict(tickmode='array', tickvals=[0,1,2], ticktext=['Low','Moderate','High']),
            hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
        st.subheader("🔍 Insights")
        if len(df_hist) >= 2:
            delta = df_hist['risk_num'].iloc[-1] - df_hist['risk_num'].iloc[0]
            if delta < 0:
                st.success(f"🎉 Risk down {abs(delta)} level(s)!")
            elif delta > 0:
                st.warning(f"⚠️ Risk up {delta} level(s).")
            else:
                st.info("📊 Stable.")
            st.metric("Avg Confidence", f"{df_hist['proba'].mean():.1%}")
        
        display_df = df_hist[['date', 'risk', 'proba', 'model']].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['proba'] = display_df['proba'].apply(lambda x: f"{x:.1%}")
        st.dataframe(display_df, use_container_width=True)
        
        csv = display_df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, f"mindtrack_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

elif page == "Global Insights":
    st.header("Model Insights 🌍")
    if X_test_global is None:
        st.warning("No test data.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Importance", "Signed Impact", "Metrics", "Confusion Matrix"])
        
        with tab1:
            fig = get_global_importance_fig()
            img_data = fig_to_base64(fig)
            if img_data:
                st.image(f"data:image/png;base64,{img_data}", caption="XGBoost Global Importance")
            else:
                st.warning("Plot failed.")
        
        with tab2:
            fig = get_global_signed_fig()
            img_data = fig_to_base64(fig)
            if img_data:
                st.image(f"data:image/png;base64,{img_data}", caption="XGBoost Signed Impacts")
            else:
                st.warning("Plot failed.")
        
        with tab3:
            st.subheader("📊 5-Model Comparison (+ Ensemble)")
            metrics_data = get_multi_model_metrics()
            if not metrics_data:
                st.warning("No models.")
            else:
                model_tabs = st.tabs(metrics_data.keys())
                for idx, name in enumerate(metrics_data.keys()):
                    with model_tabs[idx]:
                        data = metrics_data[name]
                        if 'error' in data:
                            st.error(f"Error: {data['error']}")
                            continue
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("Accuracy", data['accuracy'])
                        with col2:
                            st.markdown("**Macro**")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("P", data['macro_precision'])
                            c2.metric("R", data['macro_recall'])
                            c3.metric("F1", data['macro_f1'])
                        st.markdown("**Per-Class**")
                        st.dataframe(data['per_class'].style.background_gradient(cmap='Blues'), use_container_width=True, hide_index=True)
        
        with tab4:
            fig = get_cm_fig()
            img_data = fig_to_base64(fig)
            if img_data:
                st.image(f"data:image/png;base64,{img_data}", caption="XGBoost CM")
            else:
                st.warning("CM failed.")

elif page == "About":
    st.markdown("### MindTrack Pro FINAL FIXED 🔥\n- **Ensemble fits on TRAIN data** (no 'not fitted' error)\n- **5 Models** + Metrics + Tracking\n- **Requires**: X_train.csv, y_train.csv in artifacts/\n- **Run**: `streamlit cache clear && streamlit run app.py`")