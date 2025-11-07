#!/bin/bash

# Pipeline Demonstration Script
# Run this to show the complete CI/CD pipeline in action

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     BASKETBALL FATIGUE MONITOR - PIPELINE DEMO              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
source venv/bin/activate

# Step 1: Data Ingestion
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1/5: DATA INGESTION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/data_ingestion.py
echo "✅ Data ingestion complete!"
echo ""

# Step 2: Feature Engineering
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2/5: FEATURE ENGINEERING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/feature_engineering.py
echo "✅ Feature engineering complete!"
echo ""

# Step 3: Model Training
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3/5: MODEL TRAINING"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/train_model.py 2>&1 | grep -E "(Training|Loaded|Accuracy|F1|ROC|Model saved|threshold|Error)" || python src/train_model.py
echo "✅ Model training complete!"
echo ""

# Step 4: Testing
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4/5: RUNNING TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m pytest tests/ -v --tb=short
echo "✅ Tests complete!"
echo ""

# Step 5: Show Results
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5/5: PIPELINE ARTIFACTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Generated Files:"
ls -lh data/raw/*.csv 2>/dev/null | awk '{print "  📊 Raw Data:    " $9 " (" $5 ")"}'
ls -lh data/processed/*.csv 2>/dev/null | awk '{print "  🔧 Processed:   " $9 " (" $5 ")"}'
ls -lh models/*.pkl 2>/dev/null | awk '{print "  🤖 Model:       " $9 " (" $5 ")"}'
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    PIPELINE COMPLETE! ✅                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 View on GitHub Actions:"
echo "   https://github.com/anayk13/Fatigue-Prediction/actions"
echo ""
echo "🚀 To trigger GitHub Actions pipeline:"
echo "   git add . && git commit -m 'Demo run' && git push"
echo ""

