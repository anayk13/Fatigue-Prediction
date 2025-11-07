#!/bin/bash

# Setup script for Basketball Fatigue Monitor

echo "🏀 Setting up Basketball Fatigue Monitor..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models mlruns

# Run data pipeline
echo "Running data pipeline..."
echo "Step 1: Data ingestion..."
python src/data_ingestion.py

echo "Step 2: Feature engineering..."
python src/feature_engineering.py

echo "Step 3: Model training..."
python src/train_model.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the Streamlit app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "To run the FastAPI backend:"
echo "  cd api"
echo "  uvicorn app:app --reload"

