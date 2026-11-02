# Tutorial · 07 · P2: Customer churn, tabular ML end-to-end

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial follows the workflow in the order it was built: data → EDA → baselines → pipeline → models and tuning → a deliberate leakage → threshold from costs → calibration → explanations → API. Each step says **what** was done, **why**, and **how** it works. The README holds the final numbers.

**Goal:** not just "train a model", but answer a business question: *which customers should get a 100 € retention offer?* And do it without fooling yourself.

**You need:** project 06 (why PR-AUC, not accuracy or ROC-AUC). Libraries: pandas, scikit-learn, LightGBM, Optuna, SHAP, FastAPI. On a Mac LightGBM also needs `brew install libomp`.

---

## Step 1 · Load the data and engineer a few features ([`data.py`](data.py))

```python
df = pd.read_csv(download())                               # IBM Telco, 7043 customers
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
y = (df.pop("Churn") == "Yes").astype(int)
X = add_features(df.drop(columns="customerID"))
```

**Why each line:**
- `TotalCharges` is stored as text, and 11 brand-new customers have `" "` there (never billed). `errors="coerce"` turns that into NaN, and an imputer handles it later. Silently filling 0 here would be a decision made before looking.
- **Drop `customerID`:** an ID carries no signal about churn, only a way for a model to memorise individual rows.
- `y` is 1 for churned. 26.5 % of customers churned: imbalanced, so accuracy is misleading (project 06).

**Engineered features** (`add_features`): number of services, average past monthly charge (`TotalCharges / tenure`), and `charge_change` = today's charge minus that average (a recent price rise). **Why they are safe:** each uses only the customer's own row, so no information flows between customers or from the test set.

---

## Step 2 · Split once, and keep the test set locked

```python
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
CV = StratifiedKFold(5, shuffle=True, random_state=42)
```

**Why:** every decision (model, hyperparameters, threshold) is made with 5-fold cross-validation **on the training part only**. The 1409-customer test set is used once per model, at the end. If you choose anything by looking at the test set, its score stops being an honest estimate.

**Why stratified:** every fold keeps the same 26.5 % churners, so fold scores are comparable.

---

## Step 3 · EDA: look before modelling (`eda` in [`churn.py`](churn.py))

Churn rate per group of the most important columns:
- month-to-month contract 43 % vs two-year 3 %,
- first 6 months 53 % vs after 4 years 10 %,
- fiber 42 % vs DSL 19 %.

**Why this matters:** three columns already explain most of the story. It sets expectations (a model will mostly learn these) and suggests a human baseline.

---

## Step 4 · Two baselines before any model

- **Naive:** give everyone the training churn rate. Its PR-AUC equals the share of churners, **0.265**. That is the floor (project 06: a random model's PR-AUC = base rate).
- **Rule:** "month-to-month = will churn", what a human would write first. PR-AUC **0.406**.

**Why both:** a model is only interesting if it beats a simple rule. Otherwise you have built complexity for nothing.

---

## Step 5 · Put all preprocessing inside a `Pipeline`

```python
def preprocess():
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

def pipe(model):
    return Pipeline([("prep", preprocess()), ("model", model)])
```

**How it works:** numeric columns get the median filled in and are standardised; text categories become one-hot columns. The pipeline glues preprocessing and model into one object with `fit` and `predict_proba`.

**Why it matters:** the median, the scaling statistics and the list of categories are *learned from data*. If you compute them on all data before splitting, the validation fold has influenced training: a small leak. Inside a pipeline, cross-validation refits them on each fold's training part automatically.

**Why `handle_unknown="ignore"`:** a category never seen in training (e.g. a new payment method in the API) becomes all zeros instead of crashing.

---

## Step 6 · Compare models with cross-validation

Logistic regression, random forest and LightGBM all go through `pipe(...)` and get scored on 5-fold CV PR-AUC (mean ± standard deviation), then once on the test set.

**Why start with logistic regression:** it is fast, interpretable and a strong reference for tabular data. Here it scores 0.633 test PR-AUC, close to everything else.

**Why the standard deviation:** with ±0.02 between folds, a 0.01 difference between models is noise. Project 06's √n lesson applied.

---

## Step 7 · Tune LightGBM with Optuna

```python
def objective(trial):
    params = {"n_estimators": trial.suggest_int(...), "learning_rate": trial.suggest_float(..., log=True),
              "num_leaves": ..., "reg_lambda": ..., ...}
    return mean CV PR-AUC of pipe(lgbm(**params))
study.optimize(objective, n_trials=40)
```

**How it works:** Optuna proposes hyperparameters, learns from earlier trials which regions are promising, and proposes better ones (smarter than grid search). The objective is CV PR-AUC on training data only.

**What it found:** 550 tiny trees with **4 leaves**, a low learning rate and strong L2 regularisation. **What that tells you:** the data has little complex structure; the signal is mostly a few additive effects. That is also why logistic regression is nearly as good. Best test PR-AUC: **0.664**.

---

## Step 8 · Make a leakage on purpose and catch it (`leakage`)

```python
# WRONG: oversample first, then split
Xo, yo = oversample(Xtr, ytr, rng)          # copy churners until classes are equal
for tr, va in CV.split(Xo, yo): ...

# RIGHT: split first, oversample only the training part of each fold
for tr, va in CV.split(Xtr, ytr):
    Xf, yf = oversample(Xtr.iloc[tr], ytr.iloc[tr], rng)
    p = make().fit(Xf, yf).predict_proba(Xtr.iloc[va])[:, 1]
```

**What goes wrong:** copying churners before CV puts copies of the *same* customer into both the training fold and the validation fold. The model is then validated on rows it has memorised.

**How it is proven:** each row is hashed (`hash_pandas_object`); 89 % of validation churners had an exact copy in the training fold. CV PR-AUC: **0.893 leaky vs 0.638 correct**; the real test says 0.655.

**Lesson:** a suspiciously good result is a reason to look for a leak. Everything fitted to data (imputer, scaler, oversampling) belongs *inside* the fold. And correct oversampling did not help anyway (0.638 vs 0.640 without it).

---

## Step 9 · Choose the threshold from money, not 0.5

```python
OFFER, LOST, SAVE_RATE = 100, 500, 0.5
THEORY_THRESHOLD = OFFER / (SAVE_RATE * LOST)       # 0.40

def best_threshold(y, p):
    grid = np.arange(0.01, 1.0, 0.01)
    costs = [cost_per_customer(y, p >= t) for t in grid]
    return grid[np.argmin(costs)]
```

**The cost matrix:** an offer costs 100 €; a lost customer 500 €; an offer saves half of the would-be churners. Sending an offer is worth it when `100 + p·250 < p·500`, i.e. **p > 0.40**.

**Why also search empirically:** the 0.40 rule only holds if `p` is a true probability. So the script picks the cost-minimising threshold on **out-of-fold predictions on the training set** (`cross_val_predict`), never on the test set. Result **0.37**, close to theory.

**Result on test:** 118.2 € per customer vs 132.7 € "never offer" and 166.4 € "offer everyone". The cost curve is flat between 0.3 and 0.5, so avoiding an extreme threshold matters more than the exact value.

---

## Step 10 · Check calibration

A model is calibrated if "30 % risk" really means 30 of 100 such customers leave. `calibration_curve` bins predictions and compares the mean prediction with the observed rate; the **Brier score** is the mean squared error of the probabilities.

**Findings:**
- LightGBM and logistic regression are trained on log-loss, so they are already calibrated (mean predicted 0.266 vs true 0.265). Isotonic calibration (`CalibratedClassifierCV`) changes nothing here; the final model keeps it as a guard if data changes.
- **`class_weight="balanced"`** weights churners ~2.8× more. Every probability goes up (mean 0.41), the 0.40 rule sends 674 offers instead of 420, and the cost rises to 124.1 €.

**Lesson:** rebalancing tricks don't add information, they distort probabilities. If you rebalance, you must re-choose the threshold; better, don't, and choose the threshold from costs.

---

## Step 11 · Explain the model with SHAP (`explain`)

- **LightGBM's built-in importance** (split counts) favours continuous columns (`tenure`, charges) because they offer many split points, even for tiny gains.
- **SHAP** measures how much each feature moves each prediction (in log-odds). Top: month-to-month contract, tenure, no online security.
- The beeswarm plot shows the **direction**: short tenure pushes towards churn.

**Why both:** "how often did the model use this column" and "how much did this column change the answer" are different questions. SHAP answers the one you usually care about.

---

## Step 12 · Error analysis: who does it miss?

119 of 374 churners got no offer. The caught ones are almost all month-to-month with short tenure. The missed ones look like loyal customers (median tenure 28 months, many on long contracts). **Nothing in these columns says they are about to leave**; you would need data like complaints or competitor offers. This tells you the next improvement is better data, not a bigger model.

---

## Step 13 · Serve it with FastAPI ([`app.py`](app.py))

```python
class Customer(BaseModel):
    Contract: Literal["Month-to-month", "One year", "Two year"]
    tenure: int = Field(ge=0, le=100)
    ...

@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    X = add_features(pd.DataFrame([customer.model_dump()]))[BUNDLE["columns"]]
    p = float(BUNDLE["pipeline"].predict_proba(X)[0, 1])
    return Prediction(churn_probability=p, send_offer=p >= BUNDLE["threshold"], ...)
```

**Why each part:**
- **The saved bundle** contains the pipeline, the threshold, the column order and the costs: everything needed to reproduce a decision.
- **The same `add_features` function** as in training, so features are computed identically (a classic production bug is computing them differently).
- **`Literal` types:** a typo like `"Contract": "monthly"` returns HTTP 422 instead of a silently wrong prediction.
- **[`api_check.py`](api_check.py)** sends all 1409 test customers through the endpoint and compares with the offline predictions: max difference 0.0. That proves the API serves the same model.

---

## Step 14 · A visual example that fits ([`threshold_gif.py`](threshold_gif.py))

Churn has no natural car analogy, so instead of a Racing Lab chapter the GIF shows test customers as dots by predicted probability, the threshold sliding across them, and the cost updating until it stops at the threshold chosen from training data.

```bash
uv run python data.py && uv run python churn.py && uv run python plot.py
uv run uvicorn app:app --reload
```

---

## Try it yourself

1. Change `SAVE_RATE` to 0.3. Recompute the theory threshold (`100 / (0.3 · 500)`) and check that the empirical one follows.
2. Remove `charge_change` and `avg_monthly`. Does PR-AUC drop, or were they redundant?
3. Fit the `StandardScaler` on all of `X` before splitting and compare CV scores. Is this leak big or small here, and why?

## Recap

| Step | Idea |
|---|---|
| 2 | Lock the test set; all choices via CV on training data |
| 4 | Naive and human-rule baselines first |
| 5 | Everything learned from data goes inside the `Pipeline` |
| 7 | Tuning found tiny trees: the signal is simple |
| 8 | Oversampling before the split = leak; 0.89 looked great and was false |
| 9 | Threshold = business decision, derived from costs |
| 10 | Rebalancing distorts probabilities |
| 11 | SHAP ≠ split-count importance |
| 13 | Serve the same features and model, validate inputs, verify end to end |
