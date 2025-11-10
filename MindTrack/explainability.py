# explainability.py
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import functools
import os

plt.style.use('dark_background')

def close_fig(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        plt.close('all')
        return result
    return wrapper

class SimpleExplainer:
    def __init__(self, model_path='models/xgboost.pkl', feature_names_path='artifacts/feature_names.json'):
        self.model = joblib.load(model_path)
        with open(feature_names_path) as f:
            self.feature_names = json.load(f)
        
        self.global_sorted = sorted(
            [(self.feature_names[i] if i < len(self.feature_names) else f"f{i}", float(v)) 
             for i, v in enumerate(self.model.feature_importances_)],
            key=lambda x: x[1], reverse=True
        )

    @close_fig
    def local_explanation(self, X: pd.DataFrame, instance_idx: int = 0, top_n: int = 5, n_repeats: int = 5):
        instance = X.iloc[[instance_idx]]
        pred_class = np.argmax(self.model.predict_proba(instance)[0])
        baseline_proba = self.model.predict_proba(instance)[0][pred_class]
        
        impacts = []
        col_values = {col: X[col].values for col in X.columns}
        instance_vals = instance.iloc[0].to_dict()
        
        for col in X.columns:
            deltas = []
            values = col_values[col]
            inst_val = instance_vals[col]
            for _ in range(n_repeats):
                perturbed = instance.copy()
                if len(values) <= 1:
                    noise = np.random.normal(0, 0.01 * abs(inst_val) if abs(inst_val) > 0 else 0.01)
                    perturbed[col] = inst_val + noise
                else:
                    perturbed[col] = np.random.choice(values, size=1)
                new_proba = self.model.predict_proba(perturbed)[0][pred_class]
                deltas.append(baseline_proba - new_proba)
            impacts.append((col, float(np.mean(deltas))))
        
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_names, top_vals = zip(*impacts[:10])
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0E1117')
        colors = ['#FF3333' if v > 0 else '#33FF33' for v in top_vals]
        y_pos = np.arange(len(top_names))
        bars = ax.barh(y_pos, top_vals, color=colors, edgecolor='#FAFAFA', linewidth=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, color='#FAFAFA', fontsize=12)
        ax.set_xlabel('Impact on Risk Probability', color='#FAFAFA', fontsize=14)
        ax.set_title('Top 10 Local Feature Contributions', color='#00D4FF', fontsize=16)
        ax.axvline(0, color='#FAFAFA', linewidth=1.5)
        ax.invert_yaxis()
        ax.grid(axis='x', color='#FAFAFA', alpha=0.3, linestyle='--')
        for spine in ax.spines.values():
            spine.set_color('#FAFAFA')
        ax.tick_params(axis='x', colors='#FAFAFA')
        
        for bar in bars:
            width = bar.get_width()
            label = f'{width:+.4f}'
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                    label, ha='left' if width >= 0 else 'right', va='center', color='#FAFAFA', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return impacts[:top_n], fig

    @close_fig
    def global_importance(self, X: pd.DataFrame, top_n: int = 10):
        top = self.global_sorted[:top_n]
        top_names, top_vals = zip(*top)
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0E1117')
        colors = ['#00D4FF'] * len(top_vals)
        y_pos = np.arange(len(top_names))
        bars = ax.barh(y_pos, top_vals, color=colors, edgecolor='#FAFAFA', linewidth=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, color='#FAFAFA', fontsize=12)
        ax.set_xlabel('Feature Importance (Gain)', color='#FAFAFA', fontsize=14)
        ax.set_title('Global Feature Importance (Top 10)', color='#00D4FF', fontsize=16)
        ax.invert_yaxis()
        ax.grid(axis='x', color='#FAFAFA', alpha=0.3, linestyle='--')
        for spine in ax.spines.values():
            spine.set_color('#FAFAFA')
        ax.tick_params(axis='x', colors='#FAFAFA')
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', color='#FAFAFA', fontsize=10, fontweight='bold')
        plt.tight_layout()
        return top, fig

    @close_fig
    def global_signed_importance(self, X: pd.DataFrame, n_samples: int = 20, n_repeats: int = 5):
        X_sample = X.sample(n=min(n_samples, len(X)), random_state=42)
        all_impacts = {col: [] for col in X.columns}
        
        def cached_local(idx):
            return self.local_explanation(X_sample, instance_idx=idx, n_repeats=n_repeats)[0]
        
        for idx in tqdm(range(len(X_sample)), desc="Global Signed", leave=False):
            impacts = cached_local(idx)
            for col, val in impacts:
                all_impacts[col].append(val)
        
        avg_impacts = [(col, np.mean(vals)) for col, vals in all_impacts.items() if vals]
        avg_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_n = 10
        top_names, top_vals = zip(*avg_impacts[:top_n])
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#0E1117')
        colors = ['#FF3333' if v > 0 else '#33FF33' for v in top_vals]
        y_pos = np.arange(len(top_names))
        bars = ax.barh(y_pos, top_vals, color=colors, edgecolor='#FAFAFA', linewidth=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, color='#FAFAFA', fontsize=12)
        ax.set_xlabel('Average Impact on Risk', color='#FAFAFA', fontsize=14)
        ax.set_title('Global Signed Feature Impacts (Top 10)', color='#00D4FF', fontsize=16)
        ax.axvline(0, color='#FAFAFA', linewidth=1.5)
        ax.invert_yaxis()
        ax.grid(axis='x', color='#FAFAFA', alpha=0.3, linestyle='--')
        for spine in ax.spines.values():
            spine.set_color('#FAFAFA')
        ax.tick_params(axis='x', colors='#FAFAFA')
        
        for bar in bars:
            width = bar.get_width()
            label = f'{width:+.4f}'
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                    label, ha='left' if width >= 0 else 'right', va='center', color='#FAFAFA', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return avg_impacts[:top_n], fig

    @close_fig
    def get_metrics_and_cm(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro'),
            '的三Recall': recall_score(y_test, y_pred, average='macro'),
            'F1-Score': f1_score(y_test, y_pred, average='macro')
        }
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(9, 7), facecolor='#0E1117')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Moderate', 'High'])
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        ax.set_title('Confusion Matrix', color='#00D4FF', fontsize=16)
        ax.set_xlabel('Predicted', color='#FAFAFA')
        ax.set_ylabel('Actual', color='#FAFAFA')
        plt.tight_layout()
        return metrics, fig

    # ==================================================================
    # NEW METHOD: How top-5 features affect risk (with stress fixed)
    # ==================================================================
    @close_fig
    def plot_top5_vs_stress(self,
                            X: pd.DataFrame,
                            instance_idx: int = 0,
                            n_points: int = 30,
                            top_n: int = 5) -> tuple[list[tuple[str, float]], plt.Figure]:
        """
        For a single instance, show how the top-N local features affect
        the predicted risk probability as a function of the feature value,
        while fixing `stress_level` to the value observed in the instance.

        Returns
        -------
        impacts : list[tuple[str, float]]
            Top-N (feature, impact) pairs.
        fig : matplotlib Figure
            Line plot with one line per top feature.
        """
        instance = X.iloc[[instance_idx]]
        impacts, _ = self.local_explanation(X, instance_idx, top_n=top_n, n_repeats=1)
        top_features = [f for f, _ in impacts]

        pred_class = np.argmax(self.model.predict_proba(instance)[0])
        stress_val = instance["stress_level"].values[0]

        def risk_at(row: pd.DataFrame) -> float:
            return self.model.predict_proba(row)[0][pred_class]

        baseline_risk = risk_at(instance)
        lines = []

        for feat in top_features:
            col = X[feat]
            low = col.quantile(0.05)
            high = col.quantile(0.95)
            grid = np.linspace(low, high, n_points)

            probs = []
            for val in grid:
                perturbed = instance.copy()
                perturbed[feat] = val
                perturbed["stress_level"] = stress_val
                probs.append(risk_at(perturbed))

            lines.append((feat, grid, np.array(probs) - baseline_risk))

        fig, ax = plt.subplots(figsize=(13, 8), facecolor='#0E1117')
        colors = plt.cm.tab10(np.linspace(0, 1, len(lines)))

        for (feat, xgrid, delta), col in zip(lines, colors):
            ax.plot(xgrid, delta, label=feat, color=col, linewidth=3)

        ax.axhline(0, color='#FAFAFA', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Feature Value (Scaled)', color='#FAFAFA', fontsize=14)
        ax.set_ylabel('Δ Risk Probability (vs. baseline)', color='#FAFAFA', fontsize=14)
        ax.set_title(f'How Top {top_n} Features Affect Risk\n(stress_level = {stress_val:.2f} fixed)', 
                     color='#00D4FF', fontsize=16)
        ax.legend(facecolor='#1A1F2A', edgecolor='#FAFAFA', labelcolor='#FAFAFA', fontsize=11)
        ax.grid(True, color='#FAFAFA', alpha=0.2, linestyle='--')
        for spine in ax.spines.values():
            spine.set_color('#FAFAFA')
        ax.tick_params(colors='#FAFAFA')

        plt.tight_layout()
        return impacts, fig


# ======================================================================
# MAIN: Generate ALL plots including the new one
# ======================================================================
if __name__ == "__main__":
    plt.close('all')
    cwd = os.getcwd()
    print(f"Saving plots to: {cwd}")
    
    X_test = pd.read_csv("artifacts/X_test.csv")
    expl = SimpleExplainer()
    
    print("\n=== LOCAL EXPLANATION (Instance 0) ===")
    local, lfig = expl.local_explanation(X_test, 0)
    for f, v in local:
        print(f"{f}: {v:+.4f}")
    lfig.savefig("local_final.png", bbox_inches='tight', facecolor='#0E1117', dpi=200)
    print("Saved: local_final.png")
    
    print("\n=== GLOBAL IMPORTANCE ===")
    glob, gfig = expl.global_importance(X_test)
    for f, v in glob:
        print(f"{f}: {v:.4f}")
    gfig.savefig("global_importance.png", bbox_inches='tight', facecolor='#0E1117', dpi=200)
    print("Saved: global_importance.png")
    
    print("\n=== GLOBAL SIGNED IMPACTS ===")
    signed, sfig = expl.global_signed_importance(X_test)
    for f, v in signed:
        print(f"{f}: {v:+.4f}")
    sfig.savefig("global_signed.png", bbox_inches='tight', facecolor='#0E1117', dpi=200)
    print("Saved: global_signed.png")

    # NEW: TOP 5 vs STRESS
    print("\n=== TOP 5 FEATURES vs STRESS (Instance 0) ===")
    top5_impacts, stress_fig = expl.plot_top5_vs_stress(X_test, instance_idx=0)
    for f, v in top5_impacts:
        print(f"{f}: {v:+.4f}")
    stress_fig.savefig("top5_vs_stress.png", bbox_inches='tight', facecolor='#0E1117', dpi=200)
    print("Saved: top5_vs_stress.png")