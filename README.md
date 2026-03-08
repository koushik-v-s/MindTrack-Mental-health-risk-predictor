<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=40&pause=1000&color=00D4FF&center=true&vCenter=true&width=800&lines=MindTrack+Pro;AI+Mental+Health+Guardian;Predict.+Track.+Improve." alt="Typing SVG" />
</div>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
  <a href="#"><img src="https://img.shields.io/badge/Machine%20Learning-black?style=for-the-badge&logo=openai&logoColor=white" alt="Machine Learning"></a>
</p>

---

> [!NOTE]  
> **What is MindTrack Pro?**  
> MindTrack Pro is an advanced AI-powered web application that provides a **personalized mental health risk prediction**. It uses an ensemble of robust machine learning models (XGBoost, Random Forest, Logistic Regression, and PyTorch MLPs) to evaluate risk levels, provide local feature explainability, and track mental health trends over time.

> [!IMPORTANT]  
> All data stays **local** and is saved securely for your personal tracking. Your privacy is paramount.

## ✨ Dazzling Features

*   🎯 **High-Accuracy Assessment:** Input personalized data across 5 categories (Personal, Work, Lifestyle, MH History, Social) to predict mental health risk (Low, Moderate, High).
*   🧠 **5-Model Ensemble Boost:** Aggregates predictions from Logistic Regression, Random Forest, XGBoost, and a PyTorch MLP for peak reliability.
*   🔍 **AI Explainability:** Understand *why* the AI made its prediction. See your **Top 5 Drivers** with intuitive feature impact plots.
*   📈 **Persistent Tracking:** A built-in history dashboard that maps your risk trend over time with advanced moving averages and curved forecasts.
*   💡 **Personalized Recommendations:** Actionable, customized feedback based on the exact factors driving your risk score.

---

## 📸 Visuals & Insights

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Top 5 Features vs Stress</b></td>
      <td align="center"><b>Local Feature Impact</b></td>
    </tr>
    <tr>
      <td><img src="MindTrack/top5_vs_stress.png" width="400" alt="Top 5 vs Stress"></td>
      <td><img src="MindTrack/local_final.png" width="400" alt="Local Feature Impact"></td>
    </tr>
    <tr>
      <td align="center"><b>Global Feature Importance</b></td>
      <td align="center"><b>Global Signed Importance</b></td>
    </tr>
    <tr>
      <td><img src="MindTrack/global_importance.png" width="400" alt="Global Importance"></td>
      <td><img src="MindTrack/global_signed.png" width="400" alt="Global Signed"></td>
    </tr>
  </table>
</div>

---

## 🛠️ Supercharged Tech Stack

| Domain | Tools Used |
| :--- | :--- |
| **Frontend UI** | Streamlit, HTML/CSS injects |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, XGBoost, PyTorch |
| **Explainability** | Custom Explainer module |
| **Visualizations** | Plotly Graph Objects, Matplotlib |

---

> [!TIP]
> **Pro Tip:** For the best performance, ensure your Python environment matches the required dependency versions!

## ⚙️ How to Run Locally

Get the guardian up and running in seconds:

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/MindTrack.git
   cd MindTrack
   ```

2. **Install the dependencies**  
   Create a virtual environment (recommended) and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have streamlit, pandas, scikit-learn, xgboost, and torch installed depending on your setup)*

3. **Fire up the Streamlit App**
   ```bash
   streamlit run app.py
   ```

4. **Explore & Track**
   Open your browser at `http://localhost:8501`. Navigate through the sidebar to take your first Risk Assessment!

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00D4FF&height=120&section=footer" />
</div>
