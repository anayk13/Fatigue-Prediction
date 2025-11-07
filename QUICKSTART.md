# 🚀 Quick Start Guide

## ✅ What's Ready

Your project is **fully set up** with:
- ✅ Complete data pipeline (ingestion, feature engineering, training)
- ✅ Streamlit frontend dashboard
- ✅ FastAPI backend (optional)
- ✅ Docker configuration
- ✅ GitHub Actions CI/CD pipeline
- ✅ Tests
- ✅ All necessary scripts and configurations

## 🎯 Getting Started (3 Steps)

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Run the Pipeline

```bash
# Fetch data and train model
python src/data_ingestion.py
python src/feature_engineering.py
python src/train_model.py
```

### Step 3: Launch the App

```bash
streamlit run app.py
```

That's it! The app will open at `http://localhost:8501`

## 📝 Where You Need to Sign In

### ❌ NO SIGN-IN REQUIRED (Primary Method)

The project uses **balldontlie.io API** which is:
- ✅ **100% FREE**
- ✅ **No API key needed**
- ✅ **No account required**
- ✅ **Works immediately**

The code will automatically fetch data from this API. If the API is unavailable, it creates sample data for testing.

### 📊 Optional: Kaggle (Only if you want historical data)

**You DON'T need this** - the project works without it!

If you want historical data from Kaggle:
1. Sign up at https://www.kaggle.com (free)
2. Download "NBA Games Dataset" or "NBA Player Stats"
3. Place CSV files in `data/raw/` folder

### 🚀 Deployment (Optional)

If you want to deploy online:

1. **GitHub** (for CI/CD):
   - Sign up at https://github.com
   - Push your code to a repository
   - GitHub Actions will run automatically

2. **Render.com** (for hosting):
   - Sign up at https://render.com (free tier available)
   - Connect your GitHub repo
   - Enable auto-deploy

3. **Railway.app** (alternative):
   - Sign up at https://railway.app (free tier available)
   - Connect via GitHub integration

## 🐳 Docker (Optional)

```bash
# Build
docker build -t fatigue-monitor .

# Run
docker run -p 8501:8501 fatigue-monitor
```

## 📋 Summary

**To use the project right now:**
1. Install dependencies (`pip install -r requirements.txt`)
2. Run the pipeline (3 Python scripts)
3. Launch Streamlit (`streamlit run app.py`)

**No sign-ins required!** Everything works out of the box with the free balldontlie.io API.

## 🆘 Troubleshooting

**Model not found?**
```bash
python src/train_model.py
```

**Port in use?**
```bash
streamlit run app.py --server.port 8502
```

**Need help?** Check the full README.md for detailed instructions.

