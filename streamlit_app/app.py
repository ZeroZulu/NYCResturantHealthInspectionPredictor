"""
NYC Restaurant Health Inspection Predictor
==========================================
A Streamlit application for predicting restaurant inspection outcomes.

Features:
- Restaurant risk lookup
- Interactive map of high-risk areas
- Model explanation with SHAP
- Data exploration dashboard

Author: [Shril Patel]
Date: [Dec 13, 2025]
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Try to import folium for maps
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="NYC Restaurant Inspector",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Risk score badge */
    .risk-high {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .risk-medium {
        background-color: #ffaa00;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .risk-low {
        background-color: #00C851;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_data
def load_data():
    """Load the processed inspection data."""
    data_path = DATA_DIR / "inspections_featured.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=['inspection_date'])
        return df
    else:
        # Return sample data for demo if no data exists
        return create_demo_data()


@st.cache_resource
def load_model():
    """Load the trained model."""
    # Find the most recent model
    model_files = list(MODELS_DIR.glob("xgboost_*.joblib"))
    
    if model_files:
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        return joblib.load(latest_model)
    else:
        return None


def create_demo_data():
    """Create demo data if no real data is available."""
    np.random.seed(42)
    n = 1000
    
    boroughs = ['MANHATTAN', 'BROOKLYN', 'QUEENS', 'BRONX', 'STATEN ISLAND']
    cuisines = ['AMERICAN', 'CHINESE', 'ITALIAN', 'MEXICAN', 'JAPANESE', 'INDIAN', 'PIZZA']
    
    demo_df = pd.DataFrame({
        'camis': np.random.randint(10000000, 99999999, n),
        'dba': [f"Restaurant {i}" for i in range(n)],
        'boro': np.random.choice(boroughs, n, p=[0.35, 0.25, 0.2, 0.12, 0.08]),
        'cuisine_description': np.random.choice(cuisines, n),
        'inspection_date': pd.date_range(start='2020-01-01', periods=n, freq='H'),
        'score': np.random.randint(0, 50, n),
        'grade': np.random.choice(['A', 'B', 'C'], n, p=[0.6, 0.25, 0.15]),
        'latitude': np.random.uniform(40.5, 40.9, n),
        'longitude': np.random.uniform(-74.2, -73.7, n),
        'inspection_failed': np.random.choice([0, 1], n, p=[0.65, 0.35]),
        'historical_fail_rate': np.random.uniform(0, 0.5, n),
        'prev_score': np.random.randint(0, 50, n),
        'cuisine_risk_score': np.random.uniform(0.2, 0.5, n),
        'borough_risk_score': np.random.uniform(0.25, 0.4, n)
    })
    
    return demo_df


def get_risk_level(probability):
    """Categorize risk based on probability."""
    if probability >= 0.7:
        return "HIGH", "risk-high", "🔴"
    elif probability >= 0.4:
        return "MEDIUM", "risk-medium", "🟡"
    else:
        return "LOW", "risk-low", "🟢"


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("# 🍽️ NYC Inspector")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Risk Lookup", "🗺️ Risk Map", "📊 Analytics", "🧠 Model Insights"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app predicts restaurant health inspection outcomes
using machine learning trained on NYC DOHMH data.

**Data Source**: [NYC Open Data](https://data.cityofnewyork.us)

**GitHub**: [View Code](https://github.com/yourusername/nyc-restaurant-inspector)
""")

# ============================================================================
# MAIN PAGES
# ============================================================================

# Load data
df = load_data()
model = load_model()

if page == "🏠 Home":
    # ========== HOME PAGE ==========
    
    st.markdown('<h1 class="main-header">NYC Restaurant Health Inspection Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predicting restaurant inspection outcomes to enable smarter resource allocation</p>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Inspections",
            value=f"{len(df):,}",
            delta="Updated daily"
        )
    
    with col2:
        unique_restaurants = df['camis'].nunique() if 'camis' in df.columns else 0
        st.metric(
            label="Unique Restaurants",
            value=f"{unique_restaurants:,}"
        )
    
    with col3:
        fail_rate = df['inspection_failed'].mean() * 100 if 'inspection_failed' in df.columns else 35
        st.metric(
            label="Avg. Failure Rate",
            value=f"{fail_rate:.1f}%"
        )
    
    with col4:
        if 'inspection_date' in df.columns and df['inspection_date'].notna().any():
            latest = df['inspection_date'].max().strftime('%Y-%m-%d')
        else:
            latest = "N/A"
        st.metric(
            label="Latest Data",
            value=latest
        )
    
    st.markdown("---")
    
    # Quick overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Inspection Trends")
        
        if 'inspection_date' in df.columns:
            monthly = df.groupby(df['inspection_date'].dt.to_period('M')).agg({
                'camis': 'count',
                'inspection_failed': 'mean'
            }).reset_index()
            monthly.columns = ['Month', 'Count', 'Fail Rate']
            monthly['Month'] = monthly['Month'].astype(str)
            
            # Only use recent data for the chart
            monthly = monthly.tail(24)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly['Month'],
                y=monthly['Count'],
                name='Inspections',
                marker_color='#667eea'
            ))
            fig.add_trace(go.Scatter(
                x=monthly['Month'],
                y=monthly['Fail Rate'] * monthly['Count'].max(),
                name='Fail Rate (scaled)',
                yaxis='y2',
                line=dict(color='#ff4444', width=2)
            ))
            fig.update_layout(
                yaxis2=dict(overlaying='y', side='right', showticklabels=False),
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏙️ By Borough")
        
        if 'boro' in df.columns:
            borough_stats = df.groupby('boro').agg({
                'camis': 'count',
                'inspection_failed': 'mean'
            }).reset_index()
            borough_stats.columns = ['Borough', 'Inspections', 'Fail Rate']
            borough_stats = borough_stats.sort_values('Inspections', ascending=True)
            
            fig = px.bar(
                borough_stats,
                y='Borough',
                x='Inspections',
                orientation='h',
                color='Fail Rate',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                coloraxis_colorbar=dict(title="Fail %")
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # How it works
    st.markdown("---")
    st.markdown("### 🔧 How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Data Collection**
        
        We pull inspection records from NYC Open Data,
        including violation codes, scores, and grades.
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Feature Engineering**
        
        We create 30+ predictive features including
        historical patterns, geography, and seasonality.
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ ML Prediction**
        
        XGBoost model predicts inspection outcomes
        with ~XX% accuracy and ~XX% AUC.
        """)


elif page == "🔍 Risk Lookup":
    # ========== RISK LOOKUP PAGE ==========
    
    st.markdown("## 🔍 Restaurant Risk Lookup")
    st.markdown("Search for any NYC restaurant to see its predicted inspection risk.")
    
    # Search options
    search_method = st.radio("Search by:", ["Restaurant Name", "CAMIS ID", "Address"], horizontal=True)
    
    if search_method == "Restaurant Name":
        search_term = st.text_input("Enter restaurant name:", placeholder="e.g., Joe's Pizza")
        
        if search_term:
            # Find matching restaurants
            matches = df[df['dba'].str.contains(search_term.upper(), na=False)].drop_duplicates('camis')
            
            if len(matches) > 0:
                st.markdown(f"Found **{len(matches)}** matches:")
                
                # Show selection
                options = matches[['dba', 'boro', 'cuisine_description']].head(10)
                selection = st.selectbox(
                    "Select restaurant:",
                    options.index,
                    format_func=lambda x: f"{matches.loc[x, 'dba']} - {matches.loc[x, 'boro']} ({matches.loc[x, 'cuisine_description']})"
                )
                
                if selection is not None:
                    restaurant = matches.loc[selection]
                    
                    # Display restaurant info and risk
                    st.markdown("---")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"### {restaurant['dba']}")
                        st.markdown(f"**Borough:** {restaurant['boro']}")
                        st.markdown(f"**Cuisine:** {restaurant['cuisine_description']}")
                        
                        if 'historical_fail_rate' in restaurant:
                            st.markdown(f"**Historical Fail Rate:** {restaurant['historical_fail_rate']*100:.1f}%")
                        
                        if 'prev_score' in restaurant:
                            st.markdown(f"**Previous Score:** {restaurant['prev_score']}")
                    
                    with col2:
                        # Calculate risk score (demo: use historical fail rate)
                        risk_prob = restaurant.get('historical_fail_rate', 0.35)
                        risk_level, risk_class, risk_emoji = get_risk_level(risk_prob)
                        
                        st.markdown("### Risk Assessment")
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1rem;'>
                            <span style='font-size: 3rem;'>{risk_emoji}</span>
                            <h2 class='{risk_class}'>{risk_level} RISK</h2>
                            <p style='font-size: 1.5rem;'>{risk_prob*100:.0f}% failure probability</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No restaurants found matching your search.")
    
    elif search_method == "CAMIS ID":
        camis_id = st.text_input("Enter CAMIS ID:", placeholder="e.g., 41234567")
        
        if camis_id:
            try:
                camis_int = int(camis_id)
                restaurant = df[df['camis'] == camis_int].iloc[-1] if len(df[df['camis'] == camis_int]) > 0 else None
                
                if restaurant is not None:
                    st.success(f"Found: {restaurant['dba']}")
                    # ... display similar info as above
                else:
                    st.warning("No restaurant found with that CAMIS ID.")
            except ValueError:
                st.error("Please enter a valid numeric CAMIS ID.")


elif page == "🗺️ Risk Map":
    # ========== RISK MAP PAGE ==========
    
    st.markdown("## 🗺️ High-Risk Area Map")
    st.markdown("Interactive map showing restaurant inspection risk by location.")
    
    if FOLIUM_AVAILABLE and 'latitude' in df.columns and 'longitude' in df.columns:
        # Filter for valid coordinates
        map_df = df[
            (df['latitude'].notna()) & 
            (df['longitude'].notna()) &
            (df['latitude'] != 0) &
            (df['longitude'] != 0)
        ].copy()
        
        if len(map_df) > 0:
            # Sample for performance
            map_sample = map_df.sample(min(1000, len(map_df)), random_state=42)
            
            # Create map centered on NYC
            m = folium.Map(
                location=[40.7128, -74.0060],
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Add markers
            for _, row in map_sample.iterrows():
                risk = row.get('historical_fail_rate', 0.35)
                color = 'red' if risk >= 0.5 else 'orange' if risk >= 0.3 else 'green'
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    popup=f"{row['dba']}<br>Risk: {risk*100:.0f}%"
                ).add_to(m)
            
            # Display map
            st_folium(m, width=None, height=600)
            
            st.markdown("""
            **Legend:** 🔴 High Risk (>50%) | 🟠 Medium Risk (30-50%) | 🟢 Low Risk (<30%)
            """)
        else:
            st.warning("No valid location data available for mapping.")
    else:
        st.warning("Folium not installed or no location data. Install with: `pip install folium streamlit-folium`")
        
        # Show alternative visualization
        st.markdown("### Alternative: Risk by Borough")
        if 'boro' in df.columns and 'inspection_failed' in df.columns:
            borough_risk = df.groupby('boro')['inspection_failed'].mean().sort_values(ascending=False)
            
            fig = px.bar(
                x=borough_risk.index,
                y=borough_risk.values * 100,
                labels={'x': 'Borough', 'y': 'Failure Rate (%)'},
                color=borough_risk.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


elif page == "📊 Analytics":
    # ========== ANALYTICS PAGE ==========
    
    st.markdown("## 📊 Data Analytics Dashboard")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_borough = st.selectbox(
            "Borough",
            ["All"] + list(df['boro'].dropna().unique()) if 'boro' in df.columns else ["All"]
        )
    
    with col2:
        if 'cuisine_description' in df.columns:
            top_cuisines = df['cuisine_description'].value_counts().head(20).index.tolist()
            selected_cuisine = st.selectbox("Cuisine", ["All"] + top_cuisines)
        else:
            selected_cuisine = "All"
    
    with col3:
        if 'inspection_date' in df.columns:
            year_options = sorted(df['inspection_date'].dt.year.dropna().unique().astype(int), reverse=True)
            selected_year = st.selectbox("Year", ["All"] + list(year_options))
        else:
            selected_year = "All"
    
    # Apply filters
    filtered_df = df.copy()
    if selected_borough != "All":
        filtered_df = filtered_df[filtered_df['boro'] == selected_borough]
    if selected_cuisine != "All":
        filtered_df = filtered_df[filtered_df['cuisine_description'] == selected_cuisine]
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['inspection_date'].dt.year == int(selected_year)]
    
    st.markdown(f"**Showing {len(filtered_df):,} inspections**")
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Grade Distribution")
        if 'grade' in filtered_df.columns:
            grade_counts = filtered_df['grade'].value_counts()
            fig = px.pie(
                values=grade_counts.values,
                names=grade_counts.index,
                color=grade_counts.index,
                color_discrete_map={'A': '#00C851', 'B': '#ffaa00', 'C': '#ff4444'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Score Distribution")
        if 'score' in filtered_df.columns:
            fig = px.histogram(
                filtered_df[filtered_df['score'].notna()],
                x='score',
                nbins=30,
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    # Cuisine analysis
    st.markdown("### Failure Rate by Cuisine Type")
    if 'cuisine_description' in filtered_df.columns and 'inspection_failed' in filtered_df.columns:
        cuisine_stats = filtered_df.groupby('cuisine_description').agg({
            'inspection_failed': 'mean',
            'camis': 'count'
        }).reset_index()
        cuisine_stats.columns = ['Cuisine', 'Fail Rate', 'Count']
        cuisine_stats = cuisine_stats[cuisine_stats['Count'] >= 50].sort_values('Fail Rate', ascending=False).head(15)
        
        fig = px.bar(
            cuisine_stats,
            x='Fail Rate',
            y='Cuisine',
            orientation='h',
            color='Fail Rate',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


elif page == "🧠 Model Insights":
    # ========== MODEL INSIGHTS PAGE ==========
    
    st.markdown("## 🧠 Model Insights & Explainability")
    st.markdown("Understanding what drives inspection failure predictions.")
    
    # Feature importance
    st.markdown("### Top Predictive Features")
    
    # Demo feature importance (replace with real if model loaded)
    feature_importance = pd.DataFrame({
        'Feature': [
            'historical_fail_rate', 'prev_score', 'days_since_last_inspection',
            'cuisine_risk_score', 'borough_risk_score', 'historical_critical_count',
            'consecutive_fails', 'is_summer', 'is_reinspection', 'restaurant_age_days'
        ],
        'Importance': [0.25, 0.18, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04]
    })
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature explanations
    st.markdown("### Feature Descriptions")
    
    feature_descriptions = {
        "historical_fail_rate": "Restaurant's historical failure rate across all past inspections",
        "prev_score": "Score from the most recent inspection",
        "days_since_last_inspection": "Days elapsed since the previous inspection",
        "cuisine_risk_score": "Average failure rate for this cuisine type",
        "borough_risk_score": "Average failure rate for this borough",
        "historical_critical_count": "Total critical violations in restaurant's history",
        "consecutive_fails": "Number of consecutive failed inspections",
        "is_summer": "Whether inspection occurs in summer months (June-August)",
        "is_reinspection": "Whether this is a re-inspection (vs. initial)",
        "restaurant_age_days": "Days since the restaurant's first inspection"
    }
    
    for feature, description in feature_descriptions.items():
        st.markdown(f"**{feature}**: {description}")
    
    # Model performance
    st.markdown("---")
    st.markdown("### Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", "XX%", help="Overall prediction accuracy")
    
    with col2:
        st.metric("AUC-ROC", "0.XX", help="Area under ROC curve")
    
    with col3:
        st.metric("F1 Score", "0.XX", help="Harmonic mean of precision and recall")
    
    st.markdown("""
    > **Note**: Replace XX with your actual model metrics after training.
    > Run `python src/model_training.py` to train the model and generate metrics.
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Built with ❤️ using Streamlit | 
    <a href='https://github.com/ZeroZulu/NYCResturantHealthInspectionPredictor'>GitHub</a> |
    Data from <a href='https://data.cityofnewyork.us'>NYC Open Data</a>
</div>
""", unsafe_allow_html=True)
