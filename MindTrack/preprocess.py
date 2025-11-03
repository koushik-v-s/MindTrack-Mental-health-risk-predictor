# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
import json
import os

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['mental_health_risk'])

    # Boolean columns
    bool_cols = [
        'family_history', 'mental_health_history', 'treatment', 'can_share_problems',
        'perceived_stigma', 'sought_help', 'anxiety_attacks', 'depression_episodes',
        'suicidal_thoughts', 'major_life_event', 'trauma_history', 'insurance',
        'sleep_medication', 'screen_addiction', 'drug_dependence', 'regular_checkup',
        'mental_health_awareness', 'pet_ownership', 'marital_conflict', 'caretaking_duty', 'remote_work'
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # Numeric imputation
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['monthly_income', 'bmi']]
    num_imputer = SimpleImputer(strategy='median')
    if num_cols:
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
    for col in ['monthly_income', 'bmi']:
        if col in df.columns:
            df[col] = num_imputer.fit_transform(df[[col]])

    # Categorical imputation
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'mental_health_risk' in cat_cols:
        cat_cols.remove('mental_health_risk')
    df[cat_cols] = df[cat_cols].fillna('missing')

    # Gender
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2, 'missing': -1}).fillna(-1)

    # BMI category
    def bmi_cat(bmi):
        try:
            bmi = float(bmi)
            if bmi < 18.5: return 'Underweight'
            elif bmi < 25: return 'Normal'
            elif bmi < 30: return 'Overweight'
            else: return 'Obese'
        except:
            return 'missing'
    df['bmi_category'] = df['bmi'].apply(bmi_cat)

    # Feature engineering
    df['sleep_efficiency'] = df['sleep_duration'] * (df['sleep_quality'] / 10)
    df['working_hours'] = df['working_hours'].replace(0, np.nan)
    df['work_life_balance_index'] = (1 / (df['stress_level'] + 1)) * (40 / df['working_hours'])
    df['work_life_balance_index'] = df['work_life_balance_index'].replace([np.inf, -np.inf], np.nan)
    wlb_median = df['work_life_balance_index'].median()
    df['work_life_balance_index'] = df['work_life_balance_index'].fillna(wlb_median)

    product = df['screen_time'] * df['tech_exposure']
    tech_min, tech_max = product.min(), product.max()
    df['tech_exposure_index'] = (product - tech_min) / (tech_max - tech_min + 1e-5)

    # Ordinal maps
    wi_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4, 'missing': -1}
    support_map = {'Low': 0, 'Moderate': 1, 'High': 2, 'missing': -1}
    wp_map = {'Weak': 0, 'Moderate': 1, 'Strong': 2, 'missing': -1}
    df['work_interference_ord'] = df['work_interference'].map(wi_map).fillna(-1)
    df['support_network_ord'] = df['support_network'].map(support_map).fillna(-1)
    df['workplace_support_ord'] = df['workplace_support'].map(wp_map).fillna(-1)
    df['interaction_feature'] = df['stress_level'] * df['work_interference_ord']

    # One-hot encoding
    def one_hot_with_suffix(df, col, prefix):
        dummies = pd.get_dummies(df[col], prefix=prefix).astype(bool)
        expanded = pd.DataFrame(index=df.index)
        for c in dummies.columns:
            expanded[f"{c}_True"] = dummies[c].astype(int)
            expanded[f"{c}_False"] = (~dummies[c]).astype(int)
        return expanded

    for col, prefix in [
        ('bmi_category', 'bmi'),
        ('diet_quality', 'diet'),
        ('relationship_status', 'relationship'),
        ('education_level', 'education'),
        ('housing_status', 'housing')
    ]:
        df = pd.concat([df, one_hot_with_suffix(df, col, prefix)], axis=1)

    # Ordinal for others
    occupation_ord = {k: i for i, k in enumerate(df['occupation'].dropna().unique())}
    substance_ord = {k: i for i, k in enumerate(df['substance_use'].dropna().unique())}
    parenting_ord = {k: i for i, k in enumerate(df['parenting_status'].dropna().unique())}
    df['occupation_ord'] = df['occupation'].map(occupation_ord).fillna(-1)
    df['substance_use_ord'] = df['substance_use'].map(substance_ord).fillna(-1)
    df['parenting_status_ord'] = df['parenting_status'].map(parenting_ord).fillna(-1)

    # Drop originals
    drop_cols = ['bmi', 'bmi_category', 'diet_quality', 'education_level', 'housing_status',
                 'relationship_status', 'support_network', 'workplace_support', 'work_interference',
                 'occupation', 'substance_use', 'parenting_status']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # Target
    y = df['mental_health_risk'].map({'low': 0, 'moderate': 1, 'high': 2})
    X = df.drop(columns=['mental_health_risk'])
    X = X.fillna(X.median(numeric_only=True))

    # Scaling
    robust_cols = [c for c in [
        'monthly_income', 'screen_time', 'working_hours', 'stress_level', 'physical_activity',
        'exercise_frequency', 'meditation_frequency', 'job_satisfaction', 'life_satisfaction',
        'quality_of_friends', 'self_esteem'
    ] if c in X.columns]
    standard_cols = [c for c in X.columns if c not in robust_cols and X[c].dtype in [np.float64, np.int64]]

    scaler_robust = RobustScaler()
    scaler_standard = StandardScaler()
    if robust_cols:
        X[robust_cols] = scaler_robust.fit_transform(X[robust_cols])
    if standard_cols:
        X[standard_cols] = scaler_standard.fit_transform(X[standard_cols])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save pipeline
    pipeline = {
        'num_imputer': num_imputer,
        'scaler_robust': scaler_robust,
        'scaler_standard': scaler_standard,
        'wi_map': wi_map,
        'support_map': support_map,
        'wp_map': wp_map,
        'occupation_ord': occupation_ord,
        'substance_ord': substance_ord,
        'parenting_ord': parenting_ord,
        'tech_min': tech_min,
        'tech_max': tech_max,
        'wlb_median': wlb_median
    }
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(pipeline, 'artifacts/mindtrack_pipeline.pkl')
    json.dump(list(X.columns), open('artifacts/feature_names.json', 'w'))

    X_train.to_csv('artifacts/X_train.csv', index=False)
    X_test.to_csv('artifacts/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['mental_health_risk']).to_csv('artifacts/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['mental_health_risk']).to_csv('artifacts/y_test.csv', index=False)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_data(r'data/mindtrack_dataset.csv')
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')