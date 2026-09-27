https://fintech-ai-wuit-hackathon.streamlit.app/

# Fintech AI Hackathon: Financial Signal Escalation Prediction

## Project Overview

This project was developed for the Westminster Hackathon, Fintech AI track.

The goal is to predict the probability that a financial signal will be **escalated** based on historical transaction behavior.

The dataset is synthetic and localized for the hackathon. It contains financial signals and associated transactions.

The task is a **binary classification problem**, where:

* `eskalatsiya = 1` means the signal was escalated
* `eskalatsiya = 0` means the signal was dismissed

The final model produces an escalation probability for every signal in the test dataset.

---

## Dataset

The dataset contains four main files:

```text
data/
└── raw/
    ├── train_signals.csv
    ├── test_signals.csv
    ├── train_transactions.parquet
    └── test_transactions.parquet
```

### Signals

Training signals:

* 14,000 signals
* 11,595 non-escalated signals
* 2,405 escalated signals
* Positive class rate: **17.18%**

Test signals:

* 6,000 signals

### Transactions

Training transactions:

* ~7.0 million transactions

Test transactions:

* ~3.0 million transactions

Main transaction fields:

* `signal_id`
* `tranzaktsiya_vaqti`
* `kirim_chiqim`
* `tranzaktsiya_turi`
* `miqdor_indeksi`

---

## Project Structure

```text
Fintech-AI-WUIT_hackathon/
│
├── data/
│   ├── raw/
│   │   ├── train_signals.csv
│   │   ├── test_signals.csv
│   │   ├── train_transactions.parquet
│   │   └── test_transactions.parquet
│   │
│   └── processed/
│       ├── train_features.csv
│       └── test_features.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model.ipynb
│
├── submission/
│   └── submission.csv
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Methodology

The project consists of four main stages:

1. Exploratory Data Analysis
2. Feature Engineering
3. Model Training and Validation
4. Test Prediction and Submission

---

# 1. Exploratory Data Analysis

The EDA investigates:

* target distribution
* signal timeline
* transaction directions
* transaction types
* transaction activity per signal
* amount index distribution
* relationship between transaction behavior and escalation

The main findings include:

* The dataset is imbalanced, with approximately 17.18% positive signals.
* Most transactions are outgoing (`chiqim`).
* `karta` and `bank_otkazmasi` are the most common transaction types.
* Signals have different levels of transaction activity.
* Transaction amount statistics show substantial variation between signals.
* Temporal analysis is important because the dataset covers 2025-2026.

---

# 2. Leakage Prevention

A critical part of the feature engineering process was preventing temporal data leakage.

For each signal, only transactions that occurred **on or before the signal date** were used.

Formally:

```text
transaction_time <= signal_date
```

Transactions occurring after the signal date were excluded.

This removed **840 transactions** from the training data and prevented future information from being used to predict the signal.

The `days_before_signal` feature was then calculated as:

```text
signal_date - transaction_time
```

This allows transaction activity to be analyzed within different historical windows.

---

# 3. Feature Engineering

Transaction-level data was aggregated into signal-level features.

The final model uses 22 features:

```text
transaction_count

miqdor_mean
miqdor_median
miqdor_std
miqdor_min
miqdor_max

chiqim_count
kirim_count
kirim_ratio

bank_otkazmasi_count
karta_count
naqd_count
xalqaro_count

tx_1d
tx_3d
tx_7d
tx_30d

miqdor_q10
miqdor_q25
miqdor_q75
miqdor_q90
miqdor_iqr
```

### Feature Categories

#### Transaction volume

```text
transaction_count
tx_1d
tx_3d
tx_7d
tx_30d
```

These describe how active the signal was and how recent the transactions were.

#### Transaction direction

```text
chiqim_count
kirim_count
kirim_ratio
```

These describe incoming and outgoing transaction behavior.

#### Transaction type

```text
bank_otkazmasi_count
karta_count
naqd_count
xalqaro_count
```

These capture the distribution of transaction types.

#### Amount statistics

```text
miqdor_mean
miqdor_median
miqdor_std
miqdor_min
miqdor_max
miqdor_q10
miqdor_q25
miqdor_q75
miqdor_q90
miqdor_iqr
```

These describe the distribution of `miqdor_indeksi` for each signal.

---

# 4. Model Selection

Several models were tested using a chronological validation split.

The main models included:

* Logistic Regression
* Random Forest
* XGBoost
* CatBoost
* HistGradientBoostingClassifier

The evaluation metric was **ROC-AUC**.

### Validation Results

| Model                   | Validation AUC |
| ----------------------- | -------------: |
| Logistic Regression     |         0.5684 |
| Random Forest           |         0.6011 |
| XGBoost                 |         0.6012 |
| CatBoost                |         0.6035 |
| HistGradientBoosting v2 |         0.6080 |
| HistGradientBoosting v3 |     **0.6212** |

The final model was:

```python
HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.03,
    max_leaf_nodes=15,
    random_state=42
)
```

---

# 5. Temporal Validation

Because this is a time-dependent financial problem, a random train/validation split was not used for the final evaluation.

Instead, the data was split chronologically.

For the main validation:

```text
Training period:      before 2026-06-06
Validation period:    from 2026-06-06
```

Results:

```text
Training samples:     11,183
Validation samples:    2,817

ROC-AUC:               0.6212
```

Additional temporal validation showed variation over time:

| Validation Period       | ROC-AUC |
| ----------------------- | ------: |
| 2026-03-30 → 2026-05-31 |  0.5883 |
| 2026-06-01 → 2026-08-31 |  0.5926 |
| 2026-09-01 → 2026-12-31 |  0.6340 |

This indicates that model performance varies across time, which is important when interpreting the validation score.

---

# 6. Feature Importance

Permutation importance was used to understand which features contributed most to validation performance.

The most influential features were:

1. `miqdor_min`
2. `miqdor_max`
3. `chiqim_count`
4. `kirim_ratio`
5. `naqd_count`
6. `bank_otkazmasi_count`
7. `miqdor_std`
8. `miqdor_q25`

The strongest feature was `miqdor_min`.

Feature importance was measured using the validation set and should be interpreted as model-specific rather than as a causal explanation of escalation.

---

# 7. Final Training

After model selection, the final HistGradientBoosting model was trained on the complete training feature dataset.

```python
final_model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.03,
    max_leaf_nodes=15,
    random_state=42
)

final_model.fit(X, y)
```

Predictions were generated using:

```python
test_pred = final_model.predict_proba(X_test)[:, 1]
```

The output represents the predicted probability of escalation.

---

# 8. Submission

The final submission contains:

```text
signal_id,ehtimollik
```

where:

* `signal_id` identifies the signal
* `ehtimollik` is the predicted escalation probability

Example:

```text
signal_id,ehtimollik
12345,0.1732
12346,0.2148
12347,0.1287
```

The final submission contains:

```text
Rows:                 6000
Unique signal IDs:    6000
Missing predictions:  0
Unknown IDs:          0
Values outside [0,1]: 0
```

All submission validation checks passed.

---

# 9. Streamlit Dashboard

The project includes a Streamlit dashboard for public presentation of the EDA and modeling process.

The dashboard contains:

1. Dataset Overview
2. Target Distribution
3. Signal Timeline
4. Transaction Directions
5. Transaction Types
6. Transaction Activity per Signal
7. Target vs Transaction Activity
8. Amount Index Distribution
9. Temporal Leakage Investigation
10. Feature Engineering
11. Model Validation
12. Key Findings

Run the dashboard locally with:

```bash
python -m streamlit run app.py
```

The application can then be opened in a browser.

---

# 10. Installation

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 11. Reproducing the Project

Run the notebooks in the following order:

### Step 1: EDA

```text
notebooks/01_eda.ipynb
```

This explores the raw dataset and identifies important patterns.

### Step 2: Feature Engineering

```text
notebooks/02_feature_engineering.ipynb
```

This creates:

```text
data/processed/train_features.csv
data/processed/test_features.csv
```

### Step 3: Modeling

```text
notebooks/03_model.ipynb
```

This:

* loads processed features
* performs temporal validation
* compares models
* evaluates ROC-AUC
* trains the final model
* generates test predictions
* creates the final submission

The final file is:

```text
submission/submission.csv
```

---

# 12. Technologies

The project uses:

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* CatBoost
* Matplotlib
* Jupyter Notebook
* Streamlit
* PyArrow

---

# 13. Key Conclusions

The main conclusions from the experiments are:

* Transaction history contains predictive information about signal escalation.
* Aggregating transactions into signal-level behavioral features provides useful predictive signal.
* Temporal leakage must be explicitly controlled because transactions after the signal date cannot be used as historical evidence.
* Gradient boosting models performed better than the baseline linear model.
* HistGradientBoosting achieved a validation ROC-AUC of **0.6212** on the main chronological split.
* Model performance changes across different time periods, indicating temporal variation in the underlying data.
* Transaction amount distribution and outgoing transaction activity were among the most important predictive factors in the final model.

The final system therefore combines **EDA, leakage-safe feature engineering, temporal validation, gradient boosting, and probability-based submission generation**.
