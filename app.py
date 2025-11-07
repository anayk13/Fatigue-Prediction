"""
Streamlit Frontend Application
Basketball Player Fatigue and Injury Risk Monitor Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import FatiguePredictor

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
    
    # Load model
    predictor = load_predictor()
    
    # Show only prediction page
    show_prediction_page(predictor)


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


if __name__ == "__main__":
    main()

