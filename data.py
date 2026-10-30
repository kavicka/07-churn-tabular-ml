"""Download the IBM Telco Customer Churn dataset (the same CSV as on Kaggle) into data/.

Run:  uv run python data.py
"""

import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
PATH = Path("data/telco_churn.csv")

SERVICES = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "n_services", "avg_monthly", "charge_change"]
CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def download() -> Path:
    if not PATH.exists():
        PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, PATH)
    return PATH


def load() -> tuple[pd.DataFrame, pd.Series]:
    """Features X and target y (1 = churned). customerID is dropped: an ID carries no signal, only a way to memorise."""
    df = pd.read_csv(download())
    # 11 brand-new customers (tenure 0) have TotalCharges " ": not billed yet, so NaN and let the imputer handle it
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df.pop("Churn") == "Yes").astype(int)
    X = add_features(df.drop(columns="customerID"))
    return X, y


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """A few engineered columns. Each uses only that customer's own row, so it cannot leak across customers."""
    df = df.copy()
    df["n_services"] = (df[SERVICES] == "Yes").sum(axis=1) + (df["InternetService"] != "No")
    df["avg_monthly"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    # > 0: pays more now than on average in the past (a recent price rise)
    df["charge_change"] = df["MonthlyCharges"] - df["avg_monthly"]
    return df


if __name__ == "__main__":
    X, y = load()
    print(f"{len(X)} customers, {X.shape[1]} columns, churn rate {y.mean():.2%}")
