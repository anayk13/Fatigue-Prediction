# 🏀 Basketball Player Fatigue and Injury Risk Monitor

A complete end-to-end ML project that predicts fatigue and injury risk for NBA players using machine learning and DevOps practices.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Start the Application (3 Simple Steps)

1. **Install Dependencies**
   ```bash
   # Create virtual environment (if not already created)
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install packages
   pip install -r requirements.txt
   ```

2. **Run the Pipeline** (to fetch data and train model)
   ```bash
   # Option A: Use the automated script
   bash run_pipeline.sh
   
   # Option B: Run manually
   python src/data_ingestion.py
   python src/feature_engineering.py
   python src/train_model.py
   ```

3. **Start the Streamlit App**
   ```bash
   streamlit run app.py
   ```
   
   The app will automatically open in your browser at `http://localhost:8501`

### Alternative: Quick Start Script
```bash
# Make script executable (first time only)
chmod +x run_pipeline.sh

# Run everything
./run_pipeline.sh && streamlit run app.py
```

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Deployment](#deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Where to Sign In / Get Data](#where-to-sign-in--get-data)

## 🎯 Overview

This project monitors basketball players' fatigue and injury risk based on their match performance and workload data. It uses machine learning models (Random Forest, Logistic Regression) to predict fatigue probability and provides a Streamlit dashboard for visualization.

## ✨ Features

- **Automated Data Collection**: Fetches NBA player statistics from balldontlie.io API
- **Feature Engineering**: Computes rolling averages, performance metrics, and applies PCA
- **ML Model Training**: Random Forest and Logistic Regression models with MLflow tracking
- **Streamlit Dashboard**: Interactive web interface for predictions and visualizations
- **Player Comparison**: Compare fatigue risk between multiple players
- **CI/CD Pipeline**: Automated testing and model retraining via GitHub Actions
- **Docker Support**: Containerized deployment ready

## 🛠 Tech Stack

- **Language**: Python 3.10
- **ML Libraries**: scikit-learn, pandas, numpy, mlflow
- **Frontend**: Streamlit, Plotly
- **DevOps**: GitHub Actions, Docker
- **Data Versioning**: DVC (optional)
- **Model Tracking**: MLflow

## 📁 Project Structure

```
basketball-fatigue-monitor/
│
├── data/
│   ├── raw/              # Raw data from API
│   └── processed/        # Processed features
│
├── models/
│   └── fatigue_model.pkl # Trained model
│
├── src/
│   ├── data_ingestion.py      # Fetch data from API
│   ├── feature_engineering.py # Feature computation + PCA
│   ├── train_model.py         # Model training with MLflow
│   └── predict.py             # Prediction module
│
├── api/
│   └── app.py            # FastAPI backend (optional)
│
├── tests/
│   ├── test_data_pipeline.py
│   └── test_api.py
│
├── app.py                # Streamlit frontend
├── Dockerfile
├── requirements.txt
├── .github/workflows/
│   └── ci_cd.yml         # CI/CD pipeline
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git (for version control)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository** (or navigate to project directory):
   ```bash
   git clone <your-repo-url>
   cd Fatigue-Prediction
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env if needed
   ```

5. **Run the data pipeline**:
   ```bash
   # Option A: Use automated script
   bash run_pipeline.sh
   
   # Option B: Run manually
   python src/data_ingestion.py
   python src/feature_engineering.py
   python src/train_model.py
   ```

6. **Launch Streamlit app**:
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

### Using the Pipeline Script

The `run_pipeline.sh` script automates the entire setup:
- Creates virtual environment if needed
- Installs dependencies
- Runs data ingestion, feature engineering, and model training
- Provides a summary of generated files

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## 📊 Usage

### Streamlit Dashboard

1. **Fatigue Prediction**: Enter player statistics to get fatigue risk prediction
2. **Player Comparison**: Compare fatigue risk between two players
3. **Data Overview**: View raw and processed data
4. **Model Info**: View model details and metrics

### API Usage (FastAPI - Optional)

If you want to use the FastAPI backend instead:

```bash
cd api
uvicorn app:app --reload
```

Then access the API at `http://localhost:8000`

## 📥 Data Sources

### Primary Source: balldontlie.io API

**No sign-up required!** The API is free and open.

- **Base URL**: `https://www.balldontlie.io/api/v1`
- **Example**: `GET https://www.balldontlie.io/api/v1/stats?player_ids[]=237&seasons[]=2023`
- **Rate Limits**: None specified, but we include delays to be respectful

### Optional: Kaggle Dataset

For historical data, you can download:
- **Dataset**: "NBA Player Stats - Seasons 2013-2023" or "NBA Games Dataset"
- **URL**: https://www.kaggle.com/datasets/nathanlauga/nba-games
- **Action Required**: 
  1. Sign up for a free Kaggle account at https://www.kaggle.com
  2. Download the dataset
  3. Place CSV files in `data/raw/` directory

## 🔐 Where to Sign In / Get Data

### ✅ No Sign-In Required (Primary Method)

**balldontlie.io API** - This is the main data source and requires **NO authentication**:
- ✅ Free
- ✅ No API key needed
- ✅ No account required
- ✅ Works immediately

The project is configured to use this API by default.

### 📝 Optional: Kaggle (For Historical Data Only)

If you want to use historical Kaggle datasets:

1. **Sign up** at https://www.kaggle.com (free account)
2. **Download** the dataset: "NBA Games Dataset" or "NBA Player Stats"
3. **Place files** in `data/raw/` directory
4. The code will automatically use these files if available

**Note**: The project works without Kaggle - it will use the balldontlie.io API and create sample data if needed.

### 🚀 Deployment Platforms

For deployment, you'll need accounts on:

1. **Render.com** (Recommended - Free tier available):
   - Sign up at https://render.com
   - Connect your GitHub repository
   - Enable auto-deploy

2. **Railway.app** (Alternative - Free tier available):
   - Sign up at https://railway.app
   - Connect via GitHub integration

3. **GitHub** (For CI/CD):
   - Sign up at https://github.com (if you don't have an account)
   - Push your code to a repository
   - GitHub Actions will run automatically

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t fatigue-monitor:latest .
```

### Run Container

```bash
docker run -p 8501:8501 fatigue-monitor:latest
```

Access the app at `http://localhost:8501`

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci_cd.yml`) includes:

1. **On Push/PR**: Run tests and build Docker image
2. **Weekly Schedule** (Sunday): Automatically retrain model with latest data
3. **Auto-deploy**: Deploy to Render/Railway (when configured)

### Setting Up GitHub Actions

1. Push your code to GitHub
2. GitHub Actions will run automatically
3. For scheduled retraining, ensure the workflow has permission to commit back to the repo

## 📈 Model Training

The model uses:
- **Features**: Rolling averages, performance differences, usage rate, PCA components
- **Target**: Fatigue risk (binary classification)
- **Algorithms**: Random Forest, Logistic Regression
- **Evaluation**: Accuracy, F1 Score, ROC-AUC, Precision, Recall

Model metrics are logged to MLflow (local file store by default).

## 🧪 Testing

Run tests with:

```bash
pytest tests/ -v
```

Or run individual test files:

```bash
pytest tests/test_data_pipeline.py -v
pytest tests/test_api.py -v
```

## 📝 Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
MLFLOW_TRACKING_URI=file:./mlruns
DATA_DIR=data/raw
PROCESSED_DIR=data/processed
MODEL_DIR=models
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is open source and available for educational purposes.

## 🐛 Troubleshooting

### Model Not Found Error

If you see "Model not found", run:
```bash
python src/train_model.py
```

### API Connection Issues

If balldontlie.io API is unavailable, the code will create sample data automatically.

### Port Already in Use

If port 8501 is in use:
```bash
streamlit run app.py --server.port 8502
```

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

**Built with ❤️ using Python, Streamlit, and MLflow**

