#!/bin/bash

# Simple Pipeline Runner Script
# Run the complete ML pipeline: data ingestion → feature engineering → model training

set -e  # Exit on error

echo "🏀 Running Basketball Fatigue Monitor Pipeline..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies if needed
echo "📥 Checking dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Step 1: Data Ingestion
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1/3: DATA INGESTION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/data_ingestion.py
echo "✅ Data ingestion complete!"
echo ""

# Step 2: Feature Engineering
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2/3: FEATURE ENGINEERING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/feature_engineering.py
echo "✅ Feature engineering complete!"
echo ""

# Step 3: Model Training
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3/3: MODEL TRAINING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/train_model.py
echo "✅ Model training complete!"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PIPELINE COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Generated Files:"
[ -f data/raw/*.csv ] && ls -lh data/raw/*.csv 2>/dev/null | tail -1 | awk '{print "  • Raw Data: " $9 " (" $5 ")"}' || echo "  • Raw Data: (check data/raw/)"
[ -f data/processed/*.csv ] && ls -lh data/processed/*.csv 2>/dev/null | tail -1 | awk '{print "  • Processed: " $9 " (" $5 ")"}' || echo "  • Processed: (check data/processed/)"
[ -f models/*.pkl ] && ls -lh models/*.pkl 2>/dev/null | tail -1 | awk '{print "  • Model: " $9 " (" $5 ")"}' || echo "  • Model: (check models/)"
echo ""
echo "🚀 To run the Streamlit app:"
echo "   streamlit run app.py"
echo ""

