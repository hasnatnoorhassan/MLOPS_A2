# MLOps Assignment #02 — Titanic Survival Prediction Pipeline
**Student:** Hasnat Noor | **ID:** i222049 | **Course:** MLOps (BS DS)

End-to-end ML pipeline using **Apache Airflow** for orchestration and **MLflow** for experiment tracking, trained on the Titanic dataset to predict passenger survival.

---

## Tech Stack
| Tool | Version | Purpose |
|------|---------|---------|
| Apache Airflow | 2.8.1 | DAG orchestration & task scheduling |
| MLflow | 2.10.0 | Experiment tracking & model registry |
| scikit-learn | 1.4.0 | Model training (RandomForest, LogisticRegression) |
| pandas | 2.1.4 | Data processing |
| Python | 3.8+ | Runtime |

---

## Project Structure
```
mlops_assignment2/
├── airflow/
│   └── dags/
│       └── mlops_airflow_mlflow_pipeline.py   # Main DAG file
├── data/
│   └── titanic.csv                            # Dataset (891 rows)
├── requirements.txt                           # Python dependencies
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/hasnatnoor/mlops-assignment2.git
cd mlops-assignment2
```

### 2. Create and activate virtual environment
```bash
python3 -m venv mlops_env
source mlops_env/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialize Airflow
```bash
export AIRFLOW_HOME=~/mlops_assignment2/airflow
airflow db init
```

### 5. Create Airflow admin user
```bash
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

---

## Running the Pipeline

Open **3 separate terminals** and run the following:

### Terminal 1 — Airflow Webserver
```bash
cd ~/mlops_assignment2
source mlops_env/bin/activate
export AIRFLOW_HOME=~/mlops_assignment2/airflow
airflow webserver --port 8080
```

### Terminal 2 — Airflow Scheduler
```bash
cd ~/mlops_assignment2
source mlops_env/bin/activate
export AIRFLOW_HOME=~/mlops_assignment2/airflow
airflow scheduler
```

### Terminal 3 — MLflow UI
```bash
cd ~/mlops_assignment2
source mlops_env/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## Access UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow | http://localhost:8080 | admin / admin |
| MLflow  | http://localhost:5000 | — |

---

## Triggering DAG Runs

Go to **http://localhost:8080** → find `mlops_titanic_pipeline` → click **▶ Trigger DAG w/ config** → paste the JSON below.

### Run 1 — RandomForest (shallow)
```json
{"model_type": "RandomForest", "n_estimators": 10, "max_depth": 2, "random_state": 42}
```

### Run 2 — RandomForest (deep) ✅ Best Model
```json
{"model_type": "RandomForest", "n_estimators": 200, "max_depth": 10, "random_state": 42}
```

### Run 3 — Logistic Regression
```json
{"model_type": "LogisticRegression", "n_estimators": 100, "max_depth": "None", "random_state": 42}
```

> Wait for each run to fully complete (all tasks green) before triggering the next one.

---

## Pipeline DAG Structure

```
ingest_data
     │
validate_data          ← retries=2, retry_delay=10s (retry demo on attempt 1)
     │
┌────┴────┐
│         │            ← PARALLEL execution
handle_  feature_
missing  engineering
└────┬────┘
encode_data
     │
train_model            ← logs params + model artifact to MLflow
     │
evaluate_model         ← logs accuracy, precision, recall, F1 to MLflow
     │
branch_model           ← acc >= 0.80 → register | acc < 0.80 → reject
┌────┴────┐
│         │
register_ reject_
model     model
└────┬────┘
    end
```

---

## Experiment Results

| Run | Model | n_estimators | max_depth | Accuracy | F1-Score | Decision |
|-----|-------|-------------|-----------|----------|----------|----------|
| 01  | RandomForest | 10 | 2 | 0.7765 | 0.6825 | Rejected |
| 02  | RandomForest | 200 | 10 | **0.8324** | **0.7857** | **Registered ✓** |
| 03  | LogisticRegression | — | None | 0.7989 | 0.7465 | Rejected |

**Best Model:** Run 02 — RandomForest (n=200, depth=10) registered as `TitanicSurvivalModel` in MLflow Model Registry.

---

## Key Features Demonstrated

- **Task Orchestration** — 11-task DAG with clear dependencies
- **Parallel Execution** — handle_missing and feature_engineering run simultaneously
- **XCom Communication** — file paths, MLflow run_id, and metrics passed between tasks
- **Retry Mechanism** — validate_data fails on attempt 1, auto-retries and passes on attempt 2
- **Branching Logic** — BranchPythonOperator routes model based on accuracy threshold (0.80)
- **MLflow Tracking** — all parameters and metrics logged per run
- **Model Registry** — approved model registered with version control

---

## requirements.txt
```
apache-airflow==2.8.1
mlflow==2.10.0
scikit-learn==1.4.0
pandas==2.1.4
numpy==1.26.3
```
