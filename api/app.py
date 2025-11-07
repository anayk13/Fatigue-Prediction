"""
FastAPI Backend (Optional)
REST API for fatigue predictions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.predict import FatiguePredictor

app = FastAPI(
    title="Basketball Fatigue Monitor API",
    description="API for predicting player fatigue and injury risk",
    version="1.0.0"
)

# Load predictor
try:
    predictor = FatiguePredictor()
except FileNotFoundError:
    predictor = None
    print("Warning: Model not found. Train model first using train_model.py")


class PlayerStats(BaseModel):
    """Player statistics input model"""
    avg_minutes_last_5: float
    games_played_last_7: int
    pts_diff_from_avg: float
    reb_diff_from_avg: float
    ast_diff_from_avg: float
    usage_rate: float
    back_to_back_games: int
    age: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    fg_pct: Optional[float] = None
    efficiency: Optional[float] = None


class PredictionResponse(BaseModel):
    """Prediction response model"""
    fatigue_probability: float
    risk_level: str
    fatigue_risk: int


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Basketball Fatigue Monitor API",
        "version": "1.0.0",
        "endpoints": {
            "/predict_fatigue": "POST - Predict fatigue risk",
            "/retrain_model": "POST - Retrain model (not implemented)",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict_fatigue", response_model=PredictionResponse)
async def predict_fatigue(stats: PlayerStats):
    """
    Predict fatigue risk for a player
    
    Example request:
    ```json
    {
        "avg_minutes_last_5": 34,
        "games_played_last_7": 4,
        "pts_diff_from_avg": -6.2,
        "reb_diff_from_avg": -1.1,
        "ast_diff_from_avg": -0.5,
        "usage_rate": 28.5,
        "back_to_back_games": 1,
        "age": 30,
        "height": 200,
        "weight": 95,
        "fg_pct": 0.42,
        "efficiency": 12.5
    }
    ```
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )
    
    try:
        # Convert Pydantic model to dict
        features = stats.dict(exclude_none=True)
        
        # Ensure required fields have defaults
        if 'fg_pct' not in features:
            features['fg_pct'] = 0.45
        if 'efficiency' not in features:
            features['efficiency'] = 10.0
        
        # Make prediction
        result = predictor.predict(features)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/retrain_model")
async def retrain_model():
    """
    Retrain the model (placeholder)
    In production, this would trigger model retraining
    """
    return {
        "message": "Model retraining not implemented in API",
        "instruction": "Run 'python src/train_model.py' to retrain the model"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

