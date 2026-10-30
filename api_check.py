"""Check the FastAPI app end to end: every test customer through POST /predict must give the same
probability as churn.py computed, a typo must be rejected, and one request must be fast.

Run:  uv run python api_check.py   → results_api.json
"""

import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from app import app
from churn import SEED
from data import load

client = TestClient(app)
raw_columns = [f for f in app.openapi()["components"]["schemas"]["Customer"]["properties"]]


def main() -> None:
    X, y = load()
    _, Xte, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    expected = np.array(json.loads(Path("results.json").read_text())["final"]["test_proba"])
    got, times = [], []
    for _, row in Xte[raw_columns].iterrows():
        body = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.to_dict().items()}
        t = time.perf_counter()
        r = client.post("/predict", json=body)
        times.append(time.perf_counter() - t)
        got.append(r.json()["churn_probability"])
    bad = client.post("/predict", json=json.loads(Path("example_customer.json").read_text()) | {"Contract": "monthly"})
    example = client.post("/predict", json=json.loads(Path("example_customer.json").read_text())).json()
    res = {
        "requests": len(got),
        "max_abs_diff_vs_churn_py": float(np.max(np.abs(np.array(got) - expected))),
        "latency_ms_median": float(np.median(times) * 1000),
        "latency_ms_p95": float(np.percentile(times, 95) * 1000),
        "typo_status_code": bad.status_code,
        "example_response": example,
        "note": "in-process TestClient, no network; measured on the machine that ran it",
    }
    print(json.dumps(res, indent=1))
    Path("results_api.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
