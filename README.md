## 🎯 The Problem

New York City's Department of Health conducts approximately **24,000 restaurant inspections annually** across 27,000+ establishments. Currently, inspections follow a relatively uniform schedule, but not all restaurants pose equal risk.

**Key Questions:**
- Can we predict which restaurants are likely to fail their next inspection?
- Can we identify high-risk establishments *before* violations occur?
- How can we help the city allocate inspection resources more efficiently?

---

## 💡 Solution & Impact

This project builds a **predictive classification system** that:

1. **Predicts inspection outcomes** (Pass/Fail/Critical Violations) based on historical patterns
2. **Identifies high-risk restaurants** that warrant priority inspection
3. **Provides interpretable insights** into what factors drive violations

### Potential Impact
- **Earlier intervention**: Identify high-risk establishments 2-4 weeks before scheduled inspection
- **Resource optimization**: Focus inspector time on establishments most likely to have violations
- **Public health improvement**: Reduce foodborne illness through proactive monitoring

---

## 🚀 Live Demo

**[→ Try the Live App](https://your-app-url.streamlit.app)**

Features in the demo:
- 🔍 Look up any NYC restaurant's risk score
- 🗺️ Interactive map of high-risk areas
- 📊 Explore violation patterns by cuisine, borough, and time
- 🧠 See model explanations (SHAP values) for predictions

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Risk Scoring** | Real-time risk assessment for any NYC restaurant |
| **Geospatial Analysis** | Interactive Folium maps showing violation hotspots |
| **Model Comparison** | XGBoost vs. Random Forest vs. Logistic Regression |
| **Explainable AI** | SHAP values reveal why each prediction was made |
| **Time Series Patterns** | Seasonal and temporal violation trends |
| **API Ready** | Modular code structure for easy API deployment |

### Tech Stack

| Component | Technology |
|-----------|------------|
| **Data Source** | NYC Open Data API (DOHMH Restaurant Inspections) |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLite (local) / PostgreSQL (production) |
| **ML Models** | Scikit-learn, XGBoost |
| **Explainability** | SHAP |
| **Visualization** | Plotly, Folium, Matplotlib |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Cloud |

---
### Top Predictive Features

1. **Previous violation history** — Past behavior strongly predicts future outcomes
2. **Time since last inspection** — Longer gaps correlate with higher risk
3. **Cuisine type** — Certain cuisines have inherently higher violation rates
4. **Borough/Neighborhood** — Geographic patterns exist
5. **Inspection type** — Re-inspections vs. initial inspections differ significantly

*(Update these with your actual results after training)*

---
---

## 🔧 Feature Engineering

### Base Features (from raw data)
- Restaurant metadata (cuisine, borough, zip code)
- Inspection history (dates, scores, grades)
- Violation details (codes, descriptions, critical flags)

### Engineered Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `violation_rate_historical` | % of past inspections with violations | Past behavior predicts future |
| `days_since_last_inspection` | Time gap since previous visit | Longer gaps = more risk |
| `critical_violation_count` | # of critical violations historically | Severity matters |
| `cuisine_risk_score` | Average violation rate for cuisine type | Some cuisines are higher risk |
| `borough_risk_score` | Average violation rate for borough | Geographic patterns |
| `seasonal_risk` | Month/season of inspection | Summer = higher risk |
| `inspection_cycle` | Which inspection in the cycle (1st, 2nd, re-inspect) | Re-inspections differ |
| `complaint_density` | Nearby complaints in past 30 days | Community signals |

---

## 📚 What I Learned

### Technical Skills
- **End-to-end ML pipeline**: From raw data to deployed application
- **Feature engineering at scale**: Creating meaningful features from 200K+ records
- **Model interpretability**: Using SHAP to explain black-box predictions
- **Geospatial analysis**: Working with location data and creating interactive maps
- **Production deployment**: Streamlit Cloud, environment management

### Domain Insights
- Restaurant inspection outcomes are **highly predictable** from historical patterns
- **Cuisine type** and **geographic location** matter more than expected
- **Seasonality** plays a role — summer months show higher violation rates
- The **inspection cycle** (initial vs. re-inspection) requires different modeling approaches

### Challenges Overcome
- **Class imbalance**: Most restaurants pass — required SMOTE and threshold tuning
- **Data quality**: Missing values, inconsistent formatting needed careful handling
- **Temporal leakage**: Had to be careful not to leak future information into features

---

## 🔮 Future Improvements

- [ ] **Add weather data** — Temperature/humidity affects food safety
- [ ] **Incorporate Yelp reviews** — NLP on review text for early warning signals
- [ ] **Real-time API** — Deploy as FastAPI endpoint for integration
- [ ] **Automated retraining** — MLflow pipeline for model updates
- [ ] **A/B testing framework** — Measure real-world impact of predictions

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
