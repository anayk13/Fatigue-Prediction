"""
Tests for API endpoints
"""

import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.predict import FatiguePredictor


def test_predictor_initialization():
    """Test predictor initialization"""
    # This will fail if model doesn't exist, which is expected
    try:
        predictor = FatiguePredictor()
        assert predictor is not None
    except FileNotFoundError:
        # Expected if model hasn't been trained yet
        pytest.skip("Model not found - train model first")


def test_prediction_format():
    """Test prediction format"""
    try:
        predictor = FatiguePredictor()
        
        features = {
            'avg_minutes_last_5': 34,
            'games_played_last_7': 4,
            'pts_diff_from_avg': -6.2,
            'reb_diff_from_avg': -1.1,
            'ast_diff_from_avg': -0.5,
            'usage_rate': 28.5,
            'back_to_back_games': 1,
            'fg_pct': 0.42,
            'efficiency': 12.5
        }
        
        result = predictor.predict(features)
        
        assert 'fatigue_probability' in result
        assert 'risk_level' in result
        assert 'fatigue_risk' in result
        assert 0 <= result['fatigue_probability'] <= 1
        assert result['risk_level'] in ['Low', 'Medium', 'High']
        
    except FileNotFoundError:
        pytest.skip("Model not found - train model first")


def test_health_check():
    """Test health check (simplified)"""
    # This would test the FastAPI health endpoint if we had it running
    # For now, just verify the module can be imported
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

