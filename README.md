# 🍽️ NYC Restaurant Health Inspection Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nycresturanthealthinspectionpredictor.streamlit.app)
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-E97627?logo=tableau&logoColor=white)](https://public.tableau.com/views/NYCResturantHealthInspection/ExecutiveOverview)

> **ML system predicting restaurant inspection failures across 30,000+ NYC establishments with 90% accuracy.**

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 90% |
| **AUC-ROC** | 0.956 |
| **Records Analyzed** | 292,725 |
| **Restaurants** | 30,499 |
| **Features Engineered** | 36 |

---

## 🚀 Live Demo

### Streamlit Dashboard
**[→ Try the Interactive App](https://nycresturanthealthinspectionpredictor.streamlit.app)**
...

### Tableau Dashboard
**[→ Try the Live App](https://public.tableau.com/views/NYCResturantHealthInspection/ExecutiveOverview?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

---

## 🔑 Key Findings

- **#1 Predictor**: Critical violations (42.5% importance)
- **Highest Risk Cuisine**: Turkish (91% failure rate)
- **Highest Risk Borough**: Queens (77% failure rate)
- **Model**: XGBoost outperformed Random Forest & Logistic Regression

---

## 🛠️ Tech Stack

**ML**: Python, XGBoost, Scikit-learn, Pandas  
**Viz**: Streamlit, Tableau, Plotly  
**Data**: NYC Open Data API (DOHMH)

---

## ⚡ Quick Start

```bash
git clone https://github.com/ZeroZulu/NYCResturantHealthInspectionPredictor.git
cd NYCResturantHealthInspectionPredictor
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

---

## 📁 Structure

```
├── src/                  # ML pipeline (data_loader, feature_engineering, model_training)
├── streamlit_app/        # Web application
├── notebooks/            # EDA notebook
├── models/               # Trained models (.joblib)
└── data/                 # Raw & processed data
```

---

## 📈 Model Comparison

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| **XGBoost** | 89.9% | **0.956** |
| Random Forest | 90.0% | 0.942 |
| Logistic Regression | 85.6% | 0.891 |

---

## 👤 Author

**Shril Patel** — [GitHub](https://github.com/ZeroZulu)
