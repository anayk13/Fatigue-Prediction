"""
Streamlit Frontend Application
Basketball Player Fatigue and Injury Risk Monitor Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import FatiguePredictor
from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineering

# Page configuration
st.set_page_config(
    page_title="Basketball Fatigue Monitor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .risk-low {
        color: #00aa00;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the fatigue predictor model"""
    try:
        return FatiguePredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    """Main application"""
    st.markdown('<h1 class="main-header">🏀 Basketball Player Fatigue Monitor</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Fatigue Prediction", "Player Comparison", "Data Overview", "Model Info"]
    )
    
    # Load model
    predictor = load_predictor()
    
    if page == "Fatigue Prediction":
        show_prediction_page(predictor)
    elif page == "Player Comparison":
        show_comparison_page(predictor)
    elif page == "Data Overview":
        show_data_overview()
    elif page == "Model Info":
        show_model_info()


def show_prediction_page(predictor):
    """Fatigue prediction page"""
    st.header("Player Fatigue Risk Prediction")
    st.markdown("Enter player statistics to predict fatigue and injury risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Performance")
        avg_minutes = st.slider("Average Minutes (Last 5 Games)", 0, 48, 30)
        games_last_7 = st.slider("Games Played (Last 7 Days)", 0, 7, 4)
        back_to_back = st.checkbox("Back-to-Back Games", value=False)
        
        st.subheader("Performance Metrics")
        points = st.number_input("Points Scored (per game)", value=20, step=1, min_value=0)
        rebounds = st.number_input("Rebounds (per game)", value=7, step=1, min_value=0)
        assists = st.number_input("Assists (per game)", value=5, step=1, min_value=0)
        
        st.subheader("Additional Stats")
        turnovers = st.number_input("Turnovers (per game)", value=3, step=1, min_value=0)
        steals = st.number_input("Steals (per game)", value=1, step=1, min_value=0)
        blocks = st.number_input("Blocks (per game)", value=1, step=1, min_value=0)
        personal_fouls = st.number_input("Personal Fouls (per game)", value=2, step=1, min_value=0)
        fga = st.number_input("Field Goal Attempts (FGA)", value=15, step=1, min_value=0)
        fgm = st.number_input("Field Goals Made (FGM)", value=7, step=1, min_value=0)
        fg3m = st.number_input("3-Pointers Made", value=2, step=1, min_value=0)
        fta = st.number_input("Free Throw Attempts (FTA)", value=5, step=1, min_value=0)
    
    with col2:
        st.subheader("Player Demographics")
        age = st.slider("Age", 18, 45, 28)
        height = st.number_input("Height (cm)", 150, 230, 200)
        weight = st.number_input("Weight (kg)", 60, 150, 95)
        
        st.subheader("Advanced Metrics")
        st.caption("Usage Rate = (FGA + FTA) / Minutes * 100")
        usage_rate = st.number_input("Usage Rate (%)", value=25.0, step=0.1, min_value=0.0, max_value=100.0, help="How much player is used: (Field Goal Attempts + Free Throw Attempts) / Minutes * 100")
        fg_pct = st.number_input("Field Goal Percentage (0-1)", value=0.45, step=0.01, min_value=0.0, max_value=1.0, help="Field Goals Made / Field Goal Attempts")
        efficiency = st.number_input("Efficiency", value=10.0, step=0.1, help="Points - Turnovers - Personal Fouls")
    
    # Predict button
    if st.button("Predict Fatigue Risk", type="primary"):
        if predictor is None:
            st.error("Model not available. Please train the model first.")
            return
        
        # Calculate advanced metrics from user inputs (no auto-calculation)
        # These are calculated from what user entered, not auto-generated
        true_shooting_pct = points / (2 * (fga + 0.44 * fta + 1)) if (fga + fta) > 0 else 0.5
        effective_fg_pct = (fgm + 0.5 * fg3m) / (fga + 1) if fga > 0 else fg_pct
        turnover_rate = (turnovers / (avg_minutes + 1)) * 100 if avg_minutes > 0 else 0
        rebound_rate = (rebounds / (avg_minutes + 1)) * 100 if avg_minutes > 0 else 0
        assist_rate = (assists / (avg_minutes + 1)) * 100 if avg_minutes > 0 else 0
        defensive_activity = ((steals + blocks) / (avg_minutes + 1)) * 100 if avg_minutes > 0 else 0
        foul_rate = (personal_fouls / (avg_minutes + 1)) * 100 if avg_minutes > 0 else 0
        per = (points + rebounds + assists + steals + blocks - (fga - fgm) - (fta - (fta * 0.75)) - turnovers) / (avg_minutes + 1) if avg_minutes > 0 else 0
        game_pace = (points + rebounds + assists + steals + blocks + turnovers) / (avg_minutes + 1) if avg_minutes > 0 else 0
        
        # Prepare features - using straightforward values + advanced stats
        features = {
            'avg_minutes_last_5': avg_minutes,
            'games_played_last_7': games_last_7,
            'back_to_back_games': 1 if back_to_back else 0,
            'usage_rate': usage_rate,
            'points': points,
            'rebounds': rebounds,
            'assists': assists,
            'true_shooting_pct': true_shooting_pct,
            'effective_fg_pct': effective_fg_pct,
            'turnover_rate': turnover_rate,
            'rebound_rate': rebound_rate,
            'assist_rate': assist_rate,
            'defensive_activity': defensive_activity,
            'foul_rate': foul_rate,
            'per': per,
            'game_pace': game_pace,
            'efficiency': efficiency,  # Use user input, not calculated
            'fg_pct': fg_pct
        }
        
        # Make prediction
        try:
            result = predictor.predict(features)
            
            # Display results
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fatigue Probability", f"{result['fatigue_probability']:.1%}")
            
            with col2:
                risk_class = result['risk_level'].lower()
                risk_color = {
                    'high': 'risk-high',
                    'medium': 'risk-medium',
                    'low': 'risk-low'
                }.get(risk_class, '')
                st.markdown(f"<p class='{risk_color}'>Risk Level: {result['risk_level']}</p>", 
                           unsafe_allow_html=True)
            
            with col3:
                st.metric("Fatigue Risk", "Yes" if result['fatigue_risk'] == 1 else "No")
            
            # Visualizations
            st.subheader("Risk Visualization")
            
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['fatigue_probability'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fatigue Risk (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Feature importance (simplified)
            feature_values = pd.DataFrame({
                'Feature': list(features.keys()),
                'Value': list(features.values())
            })
            
            fig_bar = px.bar(
                feature_values,
                x='Feature',
                y='Value',
                title="Input Features",
                labels={'Value': 'Feature Value'}
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")


def show_comparison_page(predictor):
    """Player comparison page"""
    st.header("Player Comparison")
    st.markdown("Compare fatigue risk between two players over time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Player 1")
        p1_minutes = st.slider("Avg Minutes (P1)", 0, 48, 32, key="p1_min")
        p1_games = st.slider("Games Last 7 Days (P1)", 0, 7, 4, key="p1_games")
        p1_points = st.number_input("Points (P1)", value=20, step=1, key="p1_pts")
    
    with col2:
        st.subheader("Player 2")
        p2_minutes = st.slider("Avg Minutes (P2)", 0, 48, 28, key="p2_min")
        p2_games = st.slider("Games Last 7 Days (P2)", 0, 7, 3, key="p2_games")
        p2_points = st.number_input("Points (P2)", value=22, step=1, key="p2_pts")
    
    if st.button("Compare Players", type="primary"):
        if predictor is None:
            st.error("Model not available.")
            return
        
        # Predict for both players - need all features
        # Using default values for missing features in comparison
        features_p1 = {
            'avg_minutes_last_5': p1_minutes,
            'games_played_last_7': p1_games,
            'back_to_back_games': 0,
            'usage_rate': 25.0,
            'points': p1_points,
            'rebounds': 7,
            'assists': 5,
            'true_shooting_pct': 0.50,
            'effective_fg_pct': 0.45,
            'turnover_rate': 2.0,
            'rebound_rate': 0.20,
            'assist_rate': 0.15,
            'defensive_activity': 0.10,
            'foul_rate': 0.10,
            'per': 15.0,
            'game_pace': 1.0,
            'efficiency': 10.0,
            'fg_pct': 0.45
        }
        
        features_p2 = {
            'avg_minutes_last_5': p2_minutes,
            'games_played_last_7': p2_games,
            'back_to_back_games': 0,
            'usage_rate': 25.0,
            'points': p2_points,
            'rebounds': 7,
            'assists': 5,
            'true_shooting_pct': 0.50,
            'effective_fg_pct': 0.45,
            'turnover_rate': 2.0,
            'rebound_rate': 0.20,
            'assist_rate': 0.15,
            'defensive_activity': 0.10,
            'foul_rate': 0.10,
            'per': 15.0,
            'game_pace': 1.0,
            'efficiency': 10.0,
            'fg_pct': 0.45
        }
        
        try:
            result_p1 = predictor.predict(features_p1)
            result_p2 = predictor.predict(features_p2)
            
            # Comparison chart
            comparison_data = pd.DataFrame({
                'Player': ['Player 1', 'Player 2'],
                'Fatigue Probability': [
                    result_p1['fatigue_probability'],
                    result_p2['fatigue_probability']
                ],
                'Risk Level': [result_p1['risk_level'], result_p2['risk_level']]
            })
            
            fig = px.bar(
                comparison_data,
                x='Player',
                y='Fatigue Probability',
                color='Risk Level',
                title="Fatigue Risk Comparison",
                color_discrete_map={
                    'High': 'red',
                    'Medium': 'orange',
                    'Low': 'green'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
            st.subheader("Detailed Comparison")
            st.dataframe(comparison_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Comparison error: {e}")


def show_data_overview():
    """Data overview page"""
    st.header("Data Overview")
    st.markdown("View collected and processed data")
    
    tab1, tab2 = st.tabs(["Raw Data", "Processed Data"])
    
    with tab1:
        st.subheader("Raw Player Statistics")
        try:
            ingestor = DataIngestion()
            # Try to load most recent raw data
            raw_files = [f for f in os.listdir(ingestor.data_dir) if f.endswith('.csv')]
            if raw_files:
                latest_file = max(raw_files, 
                                key=lambda f: os.path.getmtime(
                                    os.path.join(ingestor.data_dir, f)))
                df_raw = pd.read_csv(os.path.join(ingestor.data_dir, latest_file))
                st.dataframe(df_raw.head(100), use_container_width=True)
                st.info(f"Total records: {len(df_raw)}")
            else:
                st.warning("No raw data files found. Run data ingestion first.")
        except Exception as e:
            st.error(f"Error loading raw data: {e}")
    
    with tab2:
        st.subheader("Processed Features")
        try:
            fe = FeatureEngineering()
            processed_files = [f for f in os.listdir(fe.output_dir) if f.endswith('.csv')]
            if processed_files:
                latest_file = max(processed_files,
                                key=lambda f: os.path.getmtime(
                                    os.path.join(fe.output_dir, f)))
                df_processed = pd.read_csv(os.path.join(fe.output_dir, latest_file))
                st.dataframe(df_processed.head(100), use_container_width=True)
                st.info(f"Total records: {len(df_processed)}")
                
                # Show fatigue risk distribution
                if 'fatigue_risk' in df_processed.columns:
                    risk_dist = df_processed['fatigue_risk'].value_counts()
                    fig = px.pie(
                        values=risk_dist.values,
                        names=['No Risk', 'At Risk'],
                        title="Fatigue Risk Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No processed data files found. Run feature engineering first.")
        except Exception as e:
            st.error(f"Error loading processed data: {e}")


def show_model_info():
    """Model information page"""
    st.header("Model Information")
    
    st.subheader("Model Details")
    st.markdown("""
    **Model Type:** Random Forest Classifier
    
    **Features Used:**
    - Average minutes (last 5 games)
    - Games played (last 7 days)
    - Performance differences from average (points, rebounds, assists)
    - Usage rate
    - Back-to-back games indicator
    - Field goal percentage
    - Efficiency metrics
    - PCA components (dimensionality reduction)
    
    **Target Variable:**
    - Fatigue Risk (Binary: 0 = No Risk, 1 = At Risk)
    
    **Risk Levels:**
    - **Low:** Probability < 40%
    - **Medium:** Probability 40-70%
    - **High:** Probability > 70%
    """)
    
    # Check if model exists
    model_path = "models/fatigue_model.pkl"
    if os.path.exists(model_path):
        st.success("✓ Model file found")
        st.info(f"Model location: {model_path}")
    else:
        st.warning("⚠ Model file not found. Please train the model first.")
    
    # MLflow info
    st.subheader("MLflow Tracking")
    mlruns_dir = "./mlruns"
    if os.path.exists(mlruns_dir):
        st.success("✓ MLflow tracking enabled")
        st.info(f"Tracking URI: file:./mlruns")
    else:
        st.info("MLflow tracking will be initialized on first model training")


if __name__ == "__main__":
    main()

