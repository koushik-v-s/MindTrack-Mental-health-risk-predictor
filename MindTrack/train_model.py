# train_model.py (UPDATED: Save XGBoost with explicit base_score fix to prevent future issues)
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}

    def load_data(self):
        X_train = pd.read_csv('artifacts/X_train.csv')
        X_test = pd.read_csv('artifacts/X_test.csv')
        y_train = pd.read_csv('artifacts/y_train.csv').values.ravel()
        y_test = pd.read_csv('artifacts/y_test.csv').values.ravel()
        return X_train, X_test, y_train, y_test

    def resample(self, X, y):
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)

    def train_logistic(self, X_res, y_res):
        param_grid = {'C': [0.1, 1, 10], 'max_iter': [500, 1000]}
        grid = GridSearchCV(LogisticRegression(class_weight='balanced', random_state=42), param_grid, cv=3)
        grid.fit(X_res, y_res)
        self.models['logistic_regression'] = grid.best_estimator_

    def train_rf(self, X_res, y_res):
        param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
        grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, cv=3)
        grid.fit(X_res, y_res)
        self.models['random_forest'] = grid.best_estimator_

    def train_xgb(self, X_res, y_res):
        # Explicitly set base_score to None or a float to avoid array string issue
        param_grid = {'n_estimators': [200, 300], 'learning_rate': [0.01, 0.1]}
        xgb_clf = xgb.XGBClassifier(
            eval_metric='mlogloss', random_state=42, objective='multi:softprob',
            num_class=3, base_score=0.5  # Force single float
        )
        grid = GridSearchCV(xgb_clf, param_grid, cv=3)
        grid.fit(X_res, y_res)
        best_model = grid.best_estimator_
        
        # EXTRA FIX: Manually set base_score after fitting
        booster = best_model.get_booster()
        booster.set_attr(base_score='0.5')  # String '0.5' for single-class compat, but works for multi
        self.models['xgboost'] = best_model

    def train_dl_model(self, X_res, y_res):
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
        
        X_tensor = torch.FloatTensor(X_res.values)
        y_tensor = torch.LongTensor(y_res)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = MLP(X_res.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(50):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        self.models['mlp'] = model

    def evaluate(self, model, X_test, y_test, name):
        if name == 'mlp':
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test.values)
                outputs = model(X_tensor)
                y_pred = torch.argmax(outputs, dim=1).numpy()
                y_proba = outputs.numpy()
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        self.results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'f1': f1_score(y_test, y_pred, average='macro'),
            'auc': roc_auc_score(y_test, y_proba, multi_class='ovr'),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }

    def save_model(self, name):
        model = self.models[name]
        path = f'models/{name}.pkl'
        os.makedirs('models', exist_ok=True)
        if name == 'mlp':
            torch.save(model.state_dict(), path.replace('.pkl', '.pth'))
        else:
            joblib.dump(model, path)

    def train_all(self):
        X_train, X_test, y_train, y_test = self.load_data()
        X_res, y_res = self.resample(X_train, y_train)

        self.train_logistic(X_res, y_res)
        self.train_rf(X_res, y_res)
        self.train_xgb(X_res, y_res)
        self.train_dl_model(X_res, y_res)

        for name, model in self.models.items():
            self.evaluate(model, X_test, y_test, name)
            self.save_model(name)

        print("\nTRAINING COMPLETE!")
        for name, res in self.results.items():
            print(f"\n{name.upper()}:")
            for metric, value in res.items():
                print(f"  {metric}: {value:.4f}")

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_all()