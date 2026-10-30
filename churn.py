"""P2 · Tabular ML end-to-end: who is about to cancel, and is it worth calling them?

Steps (every number in the README comes from results.json written here):
1. EDA numbers: churn rate and churn by contract / tenure.
2. Models on one stratified 80/20 split, compared with 5-fold stratified CV on the training part:
   naive "nobody churns" · rule "month-to-month contract" · logistic regression · random forest ·
   LightGBM · LightGBM tuned with Optuna.
3. Leakage: oversampling churners BEFORE the CV split vs inside each fold.
4. Threshold from a cost matrix (retention offer vs lost customer), chosen on out-of-fold predictions.
5. Calibration: reliability curves and Brier score; why a class-weighted model breaks the cost threshold.
6. SHAP vs LightGBM's built-in feature importance.
7. Save the final pipeline (calibrated tuned LightGBM + threshold) for the FastAPI app.

Run:  uv run python churn.py   → results.json, models/churn_model.joblib
"""

import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL, NUMERIC, load

SEED = 42
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
OPTUNA_TRIALS = 40

# Cost matrix in € per customer. A retention offer (discount) costs OFFER whether or not they would have left.
# It keeps a real churner with probability SAVE_RATE; a churner we lose costs LOST (about a year of revenue).
OFFER, LOST, SAVE_RATE = 100.0, 500.0, 0.5
# Offer when p · LOST > OFFER + p · (1 − SAVE_RATE) · LOST  ⇔  p > OFFER / (SAVE_RATE · LOST)
THEORY_THRESHOLD = OFFER / (SAVE_RATE * LOST)


def cost_per_customer(y: np.ndarray, alarm: np.ndarray) -> float:
    cost = np.where(alarm, OFFER + y * (1 - SAVE_RATE) * LOST, y * LOST)
    return float(cost.mean())


def preprocess() -> ColumnTransformer:
    """Fitted inside the Pipeline, so medians, scales and categories are learned from the training fold only."""
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)], verbose_feature_names_out=False)


def lgbm(**params) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(random_state=SEED, verbose=-1, **params)


def pipe(model) -> Pipeline:
    return Pipeline([("prep", preprocess()), ("model", model)])


class MonthToMonthRule:
    """Rule baseline a human would write: 'month-to-month contract = will churn'. Score 1/0, no learning."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = (X["Contract"] == "Month-to-month").to_numpy(float)
        return np.column_stack([1 - p, p])


class Prior:
    """Naive baseline: everybody gets the training churn rate. Ranks nobody above anybody."""

    def fit(self, X, y):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def oof_proba(model, X, y) -> np.ndarray:
    """Out-of-fold probabilities: every training row is scored by a model that never saw it."""
    out = np.zeros(len(y))
    for tr, va in CV.split(X, y):
        m = clone_model(model).fit(X.iloc[tr], y.iloc[tr])
        out[va] = m.predict_proba(X.iloc[va])[:, 1]
    return out


def clone_model(model):
    from sklearn.base import clone

    return model.__class__() if isinstance(model, (MonthToMonthRule, Prior)) else clone(model)


def fold_scores(model, X, y) -> dict:
    pr, roc = [], []
    for tr, va in CV.split(X, y):
        p = clone_model(model).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
        pr.append(average_precision_score(y.iloc[va], p))
        roc.append(roc_auc_score(y.iloc[va], p))
    return {
        "cv_pr_auc_mean": float(np.mean(pr)),
        "cv_pr_auc_std": float(np.std(pr)),
        "cv_roc_auc_mean": float(np.mean(roc)),
    }


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [cost_per_customer(y, p >= t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    alarm = p >= t
    tp = int((alarm & (y == 1)).sum())
    return {
        "threshold": t,
        "alarms": int(alarm.sum()),
        "true_alarms": tp,
        "recall": tp / int(y.sum()),
        "precision": tp / max(int(alarm.sum()), 1),
        "cost_per_customer": cost_per_customer(y, alarm),
    }


# --- 1. EDA ----------------------------------------------------------------------------
def eda(X, y) -> dict:
    by_contract = y.groupby(X["Contract"]).agg(["mean", "size"])
    bins = [-1, 6, 12, 24, 48, 72]
    tenure = y.groupby(np.digitize(X["tenure"], bins[1:-1], right=True)).agg(["mean", "size"])
    labels = ["0–6", "7–12", "13–24", "25–48", "49–72"]
    return {
        "customers": len(y),
        "churn_rate": float(y.mean()),
        "columns": int(X.shape[1]),
        "missing_total_charges": int(X["TotalCharges"].isna().sum()),
        "churn_by_contract": {k: {"rate": float(r["mean"]), "n": int(r["size"])} for k, r in by_contract.iterrows()},
        "churn_by_tenure_months": {
            labels[i]: {"rate": float(r["mean"]), "n": int(r["size"])} for i, r in tenure.iterrows()
        },
        "churn_by_internet": {k: float(v) for k, v in y.groupby(X["InternetService"]).mean().items()},
    }


# --- 2. Optuna ---------------------------------------------------------------------------
def tune(X, y) -> tuple[dict, list[float]]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 150, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        }
        return fold_scores(pipe(lgbm(**params)), X, y)["cv_pr_auc_mean"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    best = dict(study.best_params) | {"subsample_freq": 1}
    return best, [t.value for t in study.trials]


# --- 3. Leakage ----------------------------------------------------------------------------
def oversample(X, y, rng):
    """Copy random churners until both classes are the same size (a popular 'fix' for imbalance)."""
    pos = np.flatnonzero(y.to_numpy() == 1)
    extra = rng.choice(pos, (y == 0).sum() - len(pos), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def leakage(make, Xtr, ytr, Xte, yte) -> dict:
    rng = np.random.default_rng(SEED)
    # WRONG: oversample first, then split. Copies of the same churner land in the training AND validation fold.
    Xo, yo = oversample(Xtr, ytr, rng)
    wrong_pr, wrong_roc, dup_share = [], [], []
    rows = pd.util.hash_pandas_object(Xo, index=False).to_numpy()  # one hash per row: equal rows, equal hash
    for tr, va in CV.split(Xo, yo):
        p = make().fit(Xo.iloc[tr], yo.iloc[tr]).predict_proba(Xo.iloc[va])[:, 1]
        wrong_pr.append(average_precision_score(yo.iloc[va], p))
        wrong_roc.append(roc_auc_score(yo.iloc[va], p))
        seen = set(rows[tr])
        dup_share.append(np.mean([r in seen for r in rows[va][yo.iloc[va].to_numpy() == 1]]))
    p_wrong = make().fit(Xo, yo).predict_proba(Xte)[:, 1]
    # RIGHT: split first, oversample only the training part of each fold; validate on untouched real rows.
    right_pr, right_roc = [], []
    for tr, va in CV.split(Xtr, ytr):
        Xf, yf = oversample(Xtr.iloc[tr], ytr.iloc[tr], rng)
        p = make().fit(Xf, yf).predict_proba(Xtr.iloc[va])[:, 1]
        right_pr.append(average_precision_score(ytr.iloc[va], p))
        right_roc.append(roc_auc_score(ytr.iloc[va], p))
    return {
        "wrong": {
            "cv_pr_auc": float(np.mean(wrong_pr)),
            "cv_roc_auc": float(np.mean(wrong_roc)),
            "test_pr_auc": float(average_precision_score(yte, p_wrong)),
            "test_roc_auc": float(roc_auc_score(yte, p_wrong)),
            "val_churners_with_copy_in_train": float(np.mean(dup_share)),
        },
        "right": {"cv_pr_auc": float(np.mean(right_pr)), "cv_roc_auc": float(np.mean(right_roc))},
    }


# --- 6. SHAP ---------------------------------------------------------------------------------
def explain(fitted: Pipeline, Xte, top: int = 12) -> dict:
    prep, model = fitted.named_steps["prep"], fitted.named_steps["model"]
    Z = prep.transform(Xte)
    names = list(prep.get_feature_names_out())
    sv = shap.TreeExplainer(model).shap_values(Z)
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv).mean(0)
    gain = model.booster_.feature_importance("gain")
    split = model.booster_.feature_importance("split")
    order = np.argsort(-mean_abs)[:top]
    # one customer: the highest-risk one in the test set
    i = int(np.argmax(fitted.predict_proba(Xte)[:, 1]))
    contrib = np.argsort(-np.abs(sv[i]))[:8]
    return {
        "features": [names[j] for j in order],
        "mean_abs_shap": [float(mean_abs[j]) for j in order],
        "gain_share": [float(gain[j] / gain.sum()) for j in order],
        "split_share": [float(split[j] / split.sum()) for j in order],
        "top_by_split": [names[j] for j in np.argsort(-split)[:top]],
        "beeswarm": {"shap": sv[:, order].round(4).tolist(), "value": Z[:, order].round(3).tolist()},
        "example": {
            "row": i,
            "proba": float(fitted.predict_proba(Xte.iloc[[i]])[0, 1]),
            "base_log_odds": float(np.ravel(shap.TreeExplainer(model).expected_value)[-1]),
            "contributions": [
                {
                    "feature": names[j],
                    "shap": float(sv[i, j]),
                    "raw": str(Xte.iloc[i][names[j]]) if names[j] in Xte.columns else None,
                }
                for j in contrib
            ],
        },
    }


def main() -> None:
    t0 = time.time()
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    res: dict = {
        "seed": SEED,
        "split": {"train": len(ytr), "test": len(yte), "test_churners": int(yte.sum())},
        "costs": {"offer": OFFER, "lost_customer": LOST, "save_rate": SAVE_RATE, "theory_threshold": THEORY_THRESHOLD},
        "eda": eda(X, y),
    }
    print(f"{len(y)} customers, churn {y.mean():.2%}")

    best, trials = tune(Xtr, ytr)
    res["optuna"] = {"trials": OPTUNA_TRIALS, "best_params": best, "trial_cv_pr_auc": trials}
    print(f"optuna best CV PR-AUC {max(trials):.4f}  {best}")

    models = {
        "naive: nobody churns": Prior(),
        "rule: month-to-month = churn": MonthToMonthRule(),
        "logistic regression": pipe(LogisticRegression(max_iter=2000)),
        "random forest": pipe(RandomForestClassifier(300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
        "LightGBM (default)": pipe(lgbm()),
        "LightGBM (Optuna)": pipe(lgbm(**best)),
    }
    res["models"], fitted, curves = {}, {}, {}
    for name, m in models.items():
        r = fold_scores(m, Xtr, ytr)
        oof = oof_proba(m, Xtr, ytr)
        t = best_threshold(ytr.to_numpy(), oof)  # chosen on training data only, never on the test set
        m = clone_model(m).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        r |= {
            "test_pr_auc": float(average_precision_score(yte, p)),
            "test_roc_auc": float(roc_auc_score(yte, p)),
            "test_accuracy_at_0.5": float(((p >= 0.5) == yte).mean()),
            "test_brier": float(brier_score_loss(yte, p)),
            "at_cost_threshold": at_threshold(yte.to_numpy(), p, t),
        }
        res["models"][name], fitted[name] = r, m
        prec, rec, _ = precision_recall_curve(yte, p)
        step = max(1, len(rec) // 300)
        curves[name] = {"recall": rec[::step].tolist(), "precision": prec[::step].tolist()}
        a = r["at_cost_threshold"]
        print(
            f"{name:30s} CV PR-AUC {r['cv_pr_auc_mean']:.3f}±{r['cv_pr_auc_std']:.3f}  test PR-AUC "
            f"{r['test_pr_auc']:.3f}  ROC {r['test_roc_auc']:.3f}  t={a['threshold']:.2f} recall {a['recall']:.3f} "
            f"precision {a['precision']:.3f}  cost {a['cost_per_customer']:.1f} €"
        )
    res["pr_curves"] = curves
    yt = yte.to_numpy()
    res["policies"] = {
        "never offer": cost_per_customer(yt, np.zeros(len(yt), bool)),
        "offer everyone": cost_per_customer(yt, np.ones(len(yt), bool)),
        "perfect knowledge": cost_per_customer(yt, yt == 1),
    }

    res["leakage"] = leakage(lambda: pipe(lgbm()), Xtr, ytr, Xte, yte)
    print("leakage", json.dumps(res["leakage"], indent=1))

    # --- 5. Calibration
    calibrated = CalibratedClassifierCV(pipe(lgbm(**best)), method="isotonic", cv=CV)
    balanced = pipe(lgbm(**best, class_weight="balanced"))
    cal_models = {
        "logistic regression": fitted["logistic regression"],
        "LightGBM (Optuna)": fitted["LightGBM (Optuna)"],
        "LightGBM (Optuna) + isotonic": calibrated.fit(Xtr, ytr),
        "LightGBM (Optuna), class_weight=balanced": balanced.fit(Xtr, ytr),
    }
    res["calibration"] = {}
    for name, m in cal_models.items():
        p = m.predict_proba(Xte)[:, 1]
        frac, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        res["calibration"][name] = {
            "brier": float(brier_score_loss(yte, p)),
            "mean_predicted": float(p.mean()),
            "curve": {"predicted": mean_pred.tolist(), "observed": frac.tolist()},
            "at_theory_threshold": at_threshold(yt, p, THEORY_THRESHOLD),
        }
        c = res["calibration"][name]
        print(
            f"{name:42s} Brier {c['brier']:.4f}  mean p {c['mean_predicted']:.3f}  at t={THEORY_THRESHOLD}: "
            f"alarms {c['at_theory_threshold']['alarms']} cost {c['at_theory_threshold']['cost_per_customer']:.1f} €"
        )

    # --- 7. Final model: calibrated tuned LightGBM, threshold from its out-of-fold probabilities
    oof = cross_val_predict(
        CalibratedClassifierCV(pipe(lgbm(**best)), method="isotonic", cv=CV), Xtr, ytr, cv=CV, method="predict_proba"
    )[:, 1]
    t_final = best_threshold(ytr.to_numpy(), oof)
    p_final = calibrated.predict_proba(Xte)[:, 1]
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    res["final"] = {
        "model": "LightGBM (Optuna) + isotonic calibration",
        "threshold": t_final,
        "test_pr_auc": float(average_precision_score(yte, p_final)),
        "test_roc_auc": float(roc_auc_score(yte, p_final)),
        "at_threshold": at_threshold(yt, p_final, t_final),
        "at_0.5": at_threshold(yt, p_final, 0.5),
        "cost_curve": {
            "threshold": grid.tolist(),
            "train_oof": [cost_per_customer(ytr.to_numpy(), oof >= t) for t in grid],
            "test": [cost_per_customer(yt, p_final >= t) for t in grid],
        },
        "test_proba": p_final.round(4).tolist(),
        "test_y": yt.tolist(),
    }
    f = res["final"]
    # error analysis: who are the churners we catch vs the ones we miss?
    churner = yt == 1
    groups = {"caught": churner & (p_final >= t_final), "missed": churner & (p_final < t_final)}
    f["error_analysis"] = {
        k: {
            "n": int(m.sum()),
            "median_tenure": float(Xte["tenure"][m].median()),
            "contract_share": {c: float(v) for c, v in Xte["Contract"][m].value_counts(normalize=True).items()},
            "fiber_share": float((Xte["InternetService"][m] == "Fiber optic").mean()),
        }
        for k, m in groups.items()
    }
    print(f"final: threshold {t_final:.2f} (theory {THEORY_THRESHOLD:.2f})  {json.dumps(f['at_threshold'])}")

    res["shap"] = explain(fitted["LightGBM (Optuna)"], Xte)
    print("SHAP top:", res["shap"]["features"][:6], " split top:", res["shap"]["top_by_split"][:6])

    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": calibrated, "threshold": t_final, "costs": res["costs"], "columns": list(X.columns)},
        "models/churn_model.joblib",
    )
    res["runtime_s"] = round(time.time() - t0, 1)
    Path("results.json").write_text(json.dumps(res, indent=1))
    print(f"done in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
