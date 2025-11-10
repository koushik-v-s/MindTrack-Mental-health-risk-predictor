# check_model.py
import joblib
import xgboost as xgb

model = joblib.load('models/xgboost.pkl')
booster = model.get_booster()

# Correct way: Use attribute() to get parameters
params = booster.attributes()
base_score = params.get('base_score') if params else None

if base_score is None:
    # Fallback to model attribute
    base_score = getattr(model, 'base_score', 'Not found')

print(f"base_score: {base_score}")  # Expected: '0.0' (as string) or 0.0