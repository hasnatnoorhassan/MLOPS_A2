from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# ─────────────────────────────────────────────
# CONFIGURATION — Change these per run
# ─────────────────────────────────────────────
import os

DATA_PATH    = "/home/hamza/mlops_assignment2/data/titanic.csv"
MLFLOW_URI   = "sqlite:////home/hamza/mlops_assignment2/mlflow.db"
EXPERIMENT   = "Titanic_Survival"

# Read from environment variables so changes take effect immediately
MODEL_TYPE   = os.environ.get("MODEL_TYPE",   "RandomForest")
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "50"))
MAX_DEPTH_STR = os.environ.get("MAX_DEPTH", "3")
MAX_DEPTH    = None if MAX_DEPTH_STR == "None" else int(MAX_DEPTH_STR)

# ─────────────────────────────────────────────
# DEFAULT ARGS
# ─────────────────────────────────────────────
default_args = {
    "owner": "Hasnat_Noor",
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}

# ══════════════════════════════════════════════
# TASK 2 — Data Ingestion
# ══════════════════════════════════════════════
def ingest_data(**kwargs):
    print("=" * 50)
    print("TASK 2: DATA INGESTION")
    print("=" * 50)
    
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"\nColumn Names: {list(df.columns)}")
    
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\nMissing Values Count:\n{missing}")
    
    kwargs["ti"].xcom_push(key="data_path", value=DATA_PATH)
    print(f"\nData path pushed to XCom: {DATA_PATH}")
    return DATA_PATH

# ══════════════════════════════════════════════
# TASK 3 — Data Validation
# ══════════════════════════════════════════════
def validate_data(**kwargs):
    print("=" * 50)
    print("TASK 3: DATA VALIDATION")
    print("=" * 50)

    # Demonstrate retry: fail only on attempt 1
    attempt = kwargs["ti"].try_number
    print(f"Current attempt: {attempt}")
    if attempt == 1:
        print("Attempt 1: Simulating intentional failure for retry demo...")
        raise ValueError("Intentional failure on attempt 1 — will retry automatically!")

    print("Attempt 2: Proceeding with real validation...")

    path = kwargs["ti"].xcom_pull(key="data_path", task_ids="ingest_data")
    df   = pd.read_csv(path)

    age_missing      = df["Age"].isnull().mean() * 100
    embarked_missing = df["Embarked"].isnull().mean() * 100

    print(f"Age missing:      {age_missing:.2f}%")
    print(f"Embarked missing: {embarked_missing:.2f}%")

    if age_missing > 30:
        raise ValueError(f"Age missing {age_missing:.1f}% exceeds 30% threshold!")

    if embarked_missing > 30:
        raise ValueError(f"Embarked missing {embarked_missing:.1f}% exceeds 30% threshold!")

    print("\nValidation PASSED ✓")
    print("All columns within acceptable missing value range.")

# ══════════════════════════════════════════════
# TASK 4a — Handle Missing Values (Parallel)
# ══════════════════════════════════════════════
def handle_missing(**kwargs):
    print("=" * 50)
    print("TASK 4a: HANDLE MISSING VALUES")
    print("=" * 50)
    
    path = kwargs["ti"].xcom_pull(key="data_path", task_ids="ingest_data")
    df   = pd.read_csv(path)
    
    print(f"Before - Age nulls:      {df['Age'].isnull().sum()}")
    print(f"Before - Embarked nulls: {df['Embarked'].isnull().sum()}")
    print(f"Before - Fare nulls:     {df['Fare'].isnull().sum()}")
    
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    
    print(f"\nAfter - Age nulls:      {df['Age'].isnull().sum()}")
    print(f"After - Embarked nulls: {df['Embarked'].isnull().sum()}")
    print(f"After - Fare nulls:     {df['Fare'].isnull().sum()}")
    
    out = DATA_PATH.replace(".csv", "_missing_handled.csv")
    df.to_csv(out, index=False)
    
    kwargs["ti"].xcom_push(key="missing_handled_path", value=out)
    print(f"\nSaved to: {out}")

# ══════════════════════════════════════════════
# TASK 4b — Feature Engineering (Parallel)
# ══════════════════════════════════════════════
def feature_engineering(**kwargs):
    print("=" * 50)
    print("TASK 4b: FEATURE ENGINEERING")
    print("=" * 50)
    
    path = kwargs["ti"].xcom_pull(key="data_path", task_ids="ingest_data")
    df   = pd.read_csv(path)
    
    df["Age"].fillna(df["Age"].median(), inplace=True)
    
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)
    
    print(f"FamilySize - min: {df['FamilySize'].min()}, max: {df['FamilySize'].max()}")
    print(f"IsAlone    - 0s: {(df['IsAlone']==0).sum()}, 1s: {(df['IsAlone']==1).sum()}")
    print(f"\nSample rows:\n{df[['SibSp','Parch','FamilySize','IsAlone']].head()}")
    
    out = DATA_PATH.replace(".csv", "_features.csv")
    df.to_csv(out, index=False)
    
    kwargs["ti"].xcom_push(key="features_path", value=out)
    print(f"\nSaved to: {out}")

# ══════════════════════════════════════════════
# TASK 5 — Data Encoding
# ══════════════════════════════════════════════
def encode_data(**kwargs):
    print("=" * 50)
    print("TASK 5: DATA ENCODING")
    print("=" * 50)
    
    ti  = kwargs["ti"]
    p1  = ti.xcom_pull(key="missing_handled_path", task_ids="handle_missing")
    p2  = ti.xcom_pull(key="features_path",        task_ids="feature_engineering")
    
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)[["PassengerId", "FamilySize", "IsAlone"]]
    df  = df1.merge(df2, on="PassengerId", how="left")
    
    print(f"Columns before encoding: {list(df.columns)}")
    
    le = LabelEncoder()
    df["Sex"]      = le.fit_transform(df["Sex"])
    df["Embarked"].fillna("S", inplace=True)
    df["Embarked"] = le.fit_transform(df["Embarked"])
    
    print(f"\nSex encoded:      {df['Sex'].unique()}")
    print(f"Embarked encoded: {df['Embarked'].unique()}")
    
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    print(f"\nColumns after encoding & dropping: {list(df.columns)}")
    print(f"Final shape: {df.shape}")
    
    out = DATA_PATH.replace(".csv", "_encoded.csv")
    df.to_csv(out, index=False)
    
    ti.xcom_push(key="encoded_path", value=out)
    print(f"\nSaved to: {out}")

# ══════════════════════════════════════════════
# TASK 6 — Model Training with MLflow
# ══════════════════════════════════════════════
def train_model(**kwargs):
    print("=" * 50)
    print("TASK 6: MODEL TRAINING WITH MLFLOW")
    print("=" * 50)

    # Read hyperparameters from dag_run conf (passed at trigger time)
    conf         = kwargs.get("dag_run").conf or {}
    model_type   = conf.get("model_type",   os.environ.get("MODEL_TYPE",   "RandomForest"))
    n_estimators = int(conf.get("n_estimators", os.environ.get("N_ESTIMATORS", 50)))
    max_depth_val = conf.get("max_depth",   os.environ.get("MAX_DEPTH",   "3"))
    max_depth    = None if str(max_depth_val) == "None" else int(max_depth_val)
    random_state = int(conf.get("random_state", os.environ.get("RANDOM_STATE", 42)))

    ti   = kwargs["ti"]
    path = ti.xcom_pull(key="encoded_path", task_ids="encode_data")
    df   = pd.read_csv(path)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training size: {X_train.shape}")
    print(f"Testing size:  {X_test.shape}")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run() as run:
        mlflow.log_param("model_type",   model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth",    max_depth)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size",   len(X_train))
        mlflow.log_param("test_size",    len(X_test))
        mlflow.log_param("random_state", random_state)

        print(f"\nModel Type:   {model_type}")
        print(f"N_Estimators: {n_estimators}")
        print(f"Max_Depth:    {max_depth}")

        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            model = LogisticRegression(max_iter=200, random_state=42)

        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("Model artifact logged to MLflow ✓")

        ti.xcom_push(key="run_id",  value=run_id)
        ti.xcom_push(key="X_test",  value=X_test.to_json())
        ti.xcom_push(key="y_test",  value=y_test.tolist())

# ══════════════════════════════════════════════
# TASK 7 — Model Evaluation
# ══════════════════════════════════════════════
def evaluate_model(**kwargs):
    print("=" * 50)
    print("TASK 7: MODEL EVALUATION")
    print("=" * 50)
    
    ti     = kwargs["ti"]
    run_id = ti.xcom_pull(key="run_id",  task_ids="train_model")
    X_test = pd.read_json(ti.xcom_pull(key="X_test", task_ids="train_model"))
    y_test = ti.xcom_pull(key="y_test",  task_ids="train_model")
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    preds = model.predict(X_test)
    
    acc  = accuracy_score(y_test,  preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test,    preds)
    f1   = f1_score(y_test,        preds)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Log metrics to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)
    
    print("\nAll metrics logged to MLflow ✓")
    
    ti.xcom_push(key="accuracy", value=acc)

# ══════════════════════════════════════════════
# TASK 8 — Branching Logic
# ══════════════════════════════════════════════
def branch_model(**kwargs):
    print("=" * 50)
    print("TASK 8: BRANCHING LOGIC")
    print("=" * 50)
    
    acc = kwargs["ti"].xcom_pull(key="accuracy", task_ids="evaluate_model")
    
    print(f"Model Accuracy: {acc:.4f}")
    print(f"Threshold:      0.80")
    
    if acc >= 0.80:
        print("Decision: REGISTER MODEL ✓")
        return "register_model"
    else:
        print("Decision: REJECT MODEL ✗")
        return "reject_model"

# ══════════════════════════════════════════════
# TASK 9a — Register Model
# ══════════════════════════════════════════════
def register_model(**kwargs):
    print("=" * 50)
    print("TASK 9: MODEL REGISTRATION")
    print("=" * 50)
    
    ti     = kwargs["ti"]
    run_id = ti.xcom_pull(key="run_id", task_ids="train_model")
    acc    = ti.xcom_pull(key="accuracy", task_ids="evaluate_model")
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    
    model_uri = f"runs:/{run_id}/model"
    result    = mlflow.register_model(model_uri, "TitanicSurvivalModel")
    
    print(f"Model Name:    {result.name}")
    print(f"Model Version: {result.version}")
    print(f"Run ID:        {run_id}")
    print(f"Accuracy:      {acc:.4f}")
    print("\nModel successfully registered in MLflow Registry ✓")

# ══════════════════════════════════════════════
# TASK 9b — Reject Model
# ══════════════════════════════════════════════
def reject_model(**kwargs):
    print("=" * 50)
    print("TASK 9: MODEL REJECTED")
    print("=" * 50)
    
    ti     = kwargs["ti"]
    acc    = ti.xcom_pull(key="accuracy", task_ids="evaluate_model")
    run_id = ti.xcom_pull(key="run_id",   task_ids="train_model")
    
    reason = f"Accuracy {acc:.4f} is below threshold of 0.80"
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("rejection_reason", reason)
        mlflow.log_param("status", "REJECTED")
    
    print(f"Rejection Reason: {reason}")
    print("Rejection reason logged to MLflow ✓")

# ══════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════
with DAG(
    dag_id="mlops_titanic_pipeline",
    default_args=default_args,
    description="End-to-end MLOps: Airflow + MLflow on Titanic dataset",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "titanic", "mlflow"],
) as dag:

    t_ingest   = PythonOperator(task_id="ingest_data",          python_callable=ingest_data)
    t_validate = PythonOperator(task_id="validate_data",        python_callable=validate_data)
    t_missing  = PythonOperator(task_id="handle_missing",       python_callable=handle_missing)
    t_features = PythonOperator(task_id="feature_engineering",  python_callable=feature_engineering)
    t_encode   = PythonOperator(task_id="encode_data",          python_callable=encode_data)
    t_train    = PythonOperator(task_id="train_model",          python_callable=train_model)
    t_evaluate = PythonOperator(task_id="evaluate_model",       python_callable=evaluate_model)
    t_branch   = BranchPythonOperator(task_id="branch_model",   python_callable=branch_model)
    t_register = PythonOperator(task_id="register_model",       python_callable=register_model)
    t_reject   = PythonOperator(task_id="reject_model",         python_callable=reject_model)
    t_end      = EmptyOperator(
                    task_id="end",
                    trigger_rule="none_failed_min_one_success"
                 )

    # ── Task Dependencies ──────────────────────
    #
    #  ingest_data
    #       │
    #  validate_data
    #       │
    #  ┌────┴────┐
    #  │         │   ← PARALLEL
    # handle   feature
    # missing  engineering
    #  └────┬────┘
    #  encode_data
    #       │
    #  train_model
    #       │
    # evaluate_model
    #       │
    #  branch_model
    #  ┌────┴────┐
    #  │         │
    # register  reject
    #  └────┬────┘
    #      end
    #
    t_ingest >> t_validate >> [t_missing, t_features] >> t_encode
    t_encode >> t_train >> t_evaluate >> t_branch
    t_branch >> [t_register, t_reject] >> t_end