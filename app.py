"""FastAPI endpoint for the churn model saved by churn.py.

Run:  uv run uvicorn app:app --reload
      curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @example_customer.json
Docs: http://localhost:8000/docs
"""

from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from data import add_features

BUNDLE = joblib.load("models/churn_model.joblib")
YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    """One customer as it appears in the CRM. Literal types reject typos with a 422 instead of a silent wrong answer."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(None, ge=0, description="empty for brand-new customers")


class Prediction(BaseModel):
    churn_probability: float
    send_offer: bool
    threshold: float
    reason: str


app = FastAPI(title="Churn model", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "threshold": BUNDLE["threshold"], "costs": BUNDLE["costs"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    X = add_features(pd.DataFrame([customer.model_dump()]))[BUNDLE["columns"]]
    p = float(BUNDLE["pipeline"].predict_proba(X)[0, 1])
    t = BUNDLE["threshold"]
    c = BUNDLE["costs"]
    reason = (
        f"expected loss without offer {p * c['lost_customer']:.0f} € vs offer {c['offer']:.0f} € + "
        f"{p * (1 - c['save_rate']) * c['lost_customer']:.0f} € remaining risk"
    )
    return Prediction(churn_probability=round(p, 4), send_offer=p >= t, threshold=t, reason=reason)
