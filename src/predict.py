"""
Prediction Module
Loads trained model and makes fatigue predictions
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Optional


class FatiguePredictor:
    """Handles fatigue risk predictions"""
    
    def __init__(self, model_path: str = "models/fatigue_model.pkl"):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using train_model.py"
            )
        
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")
        
        # Try to get feature names from model
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = self.model.feature_names_in_
        elif hasattr(self.model, 'feature_importances_'):
            # For models without feature_names_in_, we'll need to match by position
            pass
    
    def predict(self, features: Dict) -> Dict:
        """
        Predict fatigue risk for a player
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Expected features (in order)
        expected_features = [
            'avg_minutes_last_5',
            'games_played_last_7',
            'pts_diff_from_avg',
            'reb_diff_from_avg',
            'ast_diff_from_avg',
            'usage_rate',
            'back_to_back_games',
            'fg_pct',
            'efficiency'
        ]
        
        # Add PCA features if model expects them
        if self.feature_names is not None:
            expected_features = list(self.feature_names)
        
        # Create feature vector
        feature_vector = []
        for feat in expected_features:
            if feat in features:
                feature_vector.append(features[feat])
            elif feat.startswith('pca_'):
                # PCA features might not be in input, use 0
                feature_vector.append(0.0)
            else:
                # Default value for missing features
                feature_vector.append(0.0)
        
        # Convert to numpy array
        X = np.array([feature_vector])
        
        # Make prediction
        probability = self.model.predict_proba(X)[0, 1]
        prediction = self.model.predict(X)[0]
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'fatigue_probability': float(probability),
            'fatigue_risk': int(prediction),
            'risk_level': risk_level
        }
    
    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        """
        Predict fatigue risk for multiple players
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for features in features_list:
            results.append(self.predict(features))
        return results


def main():
    """Example usage"""
    predictor = FatiguePredictor()
    
    # Example prediction
    example_features = {
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
    
    result = predictor.predict(example_features)
    print("\nPrediction Result:")
    print(f"Fatigue Probability: {result['fatigue_probability']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Fatigue Risk: {result['fatigue_risk']}")


if __name__ == "__main__":
    main()

