You are an expert DevOps + AI engineer.
Create a complete end-to-end project titled:

🏀 “Basketball Player Fatigue and Injury Risk Monitor”

A project that uses AI + DevOps to predict fatigue or injury risk for NBA players based on recent workload and performance data.
It should be simple, free, and fully automated using DevOps pipelines (no AWS).

1️⃣ Project Goals

Fetch daily NBA player stats automatically.

Compute fatigue indicators (e.g., minutes load, efficiency drop).

Train an ML model weekly (Logistic Regression or Random Forest).

Deploy a REST API for predictions using FastAPI.

Automate everything using GitHub Actions + Docker (CI/CD).

Deploy on Render.com or Railway.app (free).

2️⃣ Tech Stack

Language: Python
Libraries: pandas, numpy, scikit-learn, requests, joblib, FastAPI, uvicorn, pytest, python-dotenv
DevOps Tools: GitHub Actions (CI/CD + cron), Docker, GitHub for version control
Hosting: Render or Railway (free-tier app deployment)
Data Source: balldontlie.io API + Kaggle dataset for historical stats

3️⃣ Data Collection Details

Primary Source: https://www.balldontlie.io

API Example:

GET https://www.balldontlie.io/api/v1/stats?player_ids[]=237&seasons[]=2023


No API key required.

You can loop through player IDs to collect stats (minutes, points, rebounds, assists, etc.).

Store data as daily CSV in /data/raw/

For older data or player meta info, use Kaggle dataset:
👉 NBA Player Stats (1996–2023)

4️⃣ Features & Target

Features:

avg_minutes_last_5

games_played_last_7

pts_diff_from_avg

reb_diff_from_avg

ast_diff_from_avg

usage_rate

back_to_back_games

age, height, weight

Target:

fatigue_risk = 1 if performance dropped >20% after heavy minutes in last 7 days; else 0.

5️⃣ Folder Structure
basketball-fatigue-monitor/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   └── fatigue_model.pkl
│
├── src/
│   ├── data_ingestion.py          # fetch data from balldontlie API
│   ├── feature_engineering.py     # compute fatigue indicators
│   ├── train_model.py             # train logistic regression or random forest
│   ├── predict.py                 # load model and predict fatigue
│
├── api/
│   └── app.py                     # FastAPI app (3 endpoints)
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_api.py
│
├── Dockerfile
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci_cd.yml              # CI/CD pipeline
│
├── .env
└── README.md

6️⃣ API Design (FastAPI)

Endpoints to include:

Endpoint	Method	Description
/predict_fatigue	POST	Takes JSON with player stats → returns fatigue probability
/retrain_model	POST	Retrain model manually
/health	GET	Health check

Example payload for /predict_fatigue:

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
  "weight": 95
}


Response:

{"fatigue_probability": 0.72, "risk_level": "High"}

7️⃣ CI/CD Pipeline (GitHub Actions)

On every push: run tests + build Docker image.

On schedule (every Sunday): retrain model using latest data and push new model to models/.

Deploy Dockerized API automatically to Render or Railway.

Example cron job trigger (weekly retrain):

on:
  schedule:
    - cron: "0 0 * * 0"

8️⃣ Dockerfile

Use Python 3.10 base image.

Install dependencies.

Copy project files.

Run FastAPI with uvicorn.

9️⃣ Testing

Use pytest to verify:

Data ingestion (correct columns, no NaN)

Model training (accuracy > 0.6)

API health (/health returns 200)

🔟 Deliverables

Cursor should generate:

All scripts with docstrings and comments.

Pre-written .github/workflows/ci_cd.yml.

Ready-to-deploy Dockerfile.

Example requirements.txt.

README.md with setup + usage instructions.

🧠 Additional Notes

All components must work without AWS.

Use GitHub Actions, Docker, and Render for automation and deployment.

All data and libraries must be free.

Code must be modular, clean, and documented.

You are an expert AI + DevOps engineer. Build a complete end-to-end project titled:

🏀 "Player Fatigue and Injury Risk Monitor using Machine Learning and DevOps"

### 🔍 Problem Statement:
Predict and monitor basketball players' fatigue and injury risk based on their match performance and workload data. The model should give a risk score (0–1) for potential fatigue or injury.

### 🧠 AI/ML Part:
1. **Data Source:**
   - Use free data from [NBA Stats](https://www.kaggle.com/datasets/nathanlauga/nba-games) or [Basketball Reference](https://www.basketball-reference.com/leagues/)
   - Alternative Kaggle dataset: “NBA Player Stats - Seasons 2013-2023”

2. **Data Collected:**
   - Minutes played, average speed, acceleration (if available), points, rebounds, assists, turnovers, plus-minus, age, previous injuries, games played in last 7 days, etc.

3. **Feature Engineering:**
   - Compute rolling averages of workload metrics (e.g., 3-game, 5-game averages)
   - Calculate rate of change in performance (performance drop rate)
   - Label data: “Fatigued” or “At Risk” based on drop in performance or missed games
   - **Apply PCA (Principal Component Analysis)** to reduce correlated performance metrics (retain 95% variance)
   - Train-test split (80/20)

4. **Model:**
   - Try Random Forest, XGBoost, or LightGBM
   - Output: fatigue/injury probability score (0–1)

5. **Evaluation:**
   - Accuracy, F1 score, ROC-AUC, Precision-Recall

6. **Explainability:**
   - Use SHAP or LIME to visualize feature importance.

---

### ⚙️ DevOps Pipeline:
1. **Tools:**
   - GitHub (Version Control)
   - Docker (Containerization)
   - Jenkins or GitHub Actions (CI/CD)
   - MLflow (Model Tracking)
   - DVC (Data Versioning)
   - Prometheus + Grafana (Model performance monitoring)
   - Streamlit or FastAPI (Frontend for visualization)

2. **Pipeline Steps:**
   - **Code Commit** → Trigger CI/CD pipeline
   - **Automated Build:** Dockerize the app
   - **Testing:** Unit tests for data ingestion & model prediction
   - **Model Training:** Triggered automatically on dataset changes
   - **MLflow Integration:** Log model metrics & parameters
   - **Deployment:** Deploy Streamlit/FastAPI app locally (no AWS)
   - **Monitoring:** Prometheus tracks API latency & request count, Grafana dashboard displays live performance

---

### 🧰 Stack Summary:
- **Language:** Python
- **ML Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, shap, mlflow
- **DevOps Tools:** GitHub Actions, Docker, DVC, MLflow, Prometheus, Grafana, Streamlit
- **Data Source:** Kaggle NBA dataset (free)

---

### 🧾 Deliverables:
- `data_ingestion.py` → Loads and cleans data
- `feature_engineering.py` → Rolling stats + PCA + feature creation
- `train_model.py` → ML model training + evaluation + MLflow logging
- `app.py` → Streamlit app with visual fatigue/injury dashboard
- `.github/workflows/ci.yml` → CI/CD pipeline for auto-testing & deployment
- `Dockerfile` → For containerized deployment
- `prometheus.yml` + Grafana dashboard → For monitoring

---

### 🧠 Bonus:
Add a “player comparison” tab on the Streamlit dashboard comparing fatigue risk over time between two players.

