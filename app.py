from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Fintech AI: Signal Escalation Prediction",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path("data/raw")


# ============================================================
# TITLE
# ============================================================

st.title("Fintech AI: Signal Escalation Prediction")

st.markdown(
    """
This interactive dashboard presents the exploratory data analysis
of a synthetic financial monitoring dataset.

The objective is to predict the probability that a monitoring signal
will be escalated.
"""
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    train_signals = pd.read_csv(
        DATA_DIR / "train_signals.csv"
    )

    test_signals = pd.read_csv(
        DATA_DIR / "test_signals.csv"
    )

    train_signals["signal_sanasi"] = pd.to_datetime(
        train_signals["signal_sanasi"]
    )

    test_signals["signal_sanasi"] = pd.to_datetime(
        test_signals["signal_sanasi"]
    )

    # --------------------------------------------------------
    # Transactions
    # --------------------------------------------------------

    train_transactions = pd.read_parquet(
        DATA_DIR / "train_transactions.parquet"
    )


    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    def normalize_columns(df):

        df = df.copy()

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        # Possible alternative names
        aliases = {
            "transaction_time": "tranzaktsiya_vaqti",
            "transaction_datetime": "tranzaktsiya_vaqti",
            "tranzaksiya_vaqti": "tranzaktsiya_vaqti",

            "transaction_type": "tranzaktsiya_turi",
            "transaction_kind": "tranzaktsiya_turi",
            "tranzaksiya_turi": "tranzaktsiya_turi",

            "direction": "kirim_chiqim",
            "transaction_direction": "kirim_chiqim",
            "kirim_chiqim": "kirim_chiqim",

            "amount_index": "miqdor_indeksi",
            "amount": "miqdor_indeksi",
            "miqdor_indeksi": "miqdor_indeksi",

            "signal": "signal_id",
            "signal_identifier": "signal_id",
            "signal_id": "signal_id",
        }

        df = df.rename(
            columns={
                column: aliases.get(column, column)
                for column in df.columns
            }
        )

        return df

    train_transactions = normalize_columns(
        train_transactions
    )

    test_transactions = normalize_columns(
        test_transactions
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_transaction_columns = {
        "signal_id",
        "tranzaktsiya_vaqti",
        "kirim_chiqim",
        "tranzaktsiya_turi",
        "miqdor_indeksi",
    }

    train_missing = (
        required_transaction_columns
        - set(train_transactions.columns)
    )

    test_missing = (
        required_transaction_columns
        - set(test_transactions.columns)
    )

    if train_missing:
        raise ValueError(
            "Missing columns in train_transactions: "
            + str(sorted(train_missing))
            + "\n\nActual columns: "
            + str(train_transactions.columns.tolist())
        )

    if test_missing:
        raise ValueError(
            "Missing columns in test_transactions: "
            + str(sorted(test_missing))
            + "\n\nActual columns: "
            + str(test_transactions.columns.tolist())
        )

    return (
        train_signals,
        test_signals,
        train_transactions,
        test_transactions,
    )


# ============================================================
# LOAD
# ============================================================

(
    train_signals,
    test_signals,
    train_transactions,
    test_transactions,
) = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dataset")

st.sidebar.metric(
    "Training Signals",
    f"{len(train_signals):,}",
)

st.sidebar.metric(
    "Test Signals",
    f"{len(test_signals):,}",
)

st.sidebar.metric(
    "Training Transactions",
    f"{len(train_transactions):,}",
)

st.sidebar.metric(
    "Test Transactions",
    f"{len(test_transactions):,}",
)


# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Train Signals",
        f"{len(train_signals):,}",
    )

with col2:
    st.metric(
        "Test Signals",
        f"{len(test_signals):,}",
    )

with col3:
    st.metric(
        "Train Transactions",
        f"{len(train_transactions):,}",
    )

with col4:
    st.metric(
        "Test Transactions",
        f"{len(test_transactions):,}",
    )


st.subheader("Training Signals")

st.dataframe(
    train_signals.head(10),
    use_container_width=True,
)


st.subheader("Training Transactions")

st.dataframe(
    train_transactions.head(10),
    use_container_width=True,
)


# ============================================================
# 2. TARGET DISTRIBUTION
# ============================================================

st.header("2. Target Distribution")

target_counts = (
    train_signals["eskalatsiya"]
    .value_counts()
    .sort_index()
)

target_percentages = (
    target_counts
    / len(train_signals)
    * 100
)

target_table = pd.DataFrame(
    {
        "Count": target_counts,
        "Percentage": target_percentages.round(2),
    }
)

st.dataframe(
    target_table,
    use_container_width=True,
)


fig, ax = plt.subplots()

ax.bar(
    ["Dismissed (0)", "Escalated (1)"],
    [
        target_counts.get(0, 0),
        target_counts.get(1, 0),
    ],
)

ax.set_ylabel("Number of signals")
ax.set_title("Target Distribution")

st.pyplot(fig)

plt.close(fig)


escalation_rate = (
    train_signals["eskalatsiya"].mean()
    * 100
)

st.metric(
    "Escalation Rate",
    f"{escalation_rate:.2f}%",
)


# ============================================================
# 3. SIGNAL TIMELINE
# ============================================================

st.header("3. Signal Timeline")

monthly_signals = (
    train_signals
    .set_index("signal_sanasi")
    .resample("ME")
    .size()
)

fig, ax = plt.subplots()

ax.plot(
    monthly_signals.index,
    monthly_signals.to_numpy(),
)

ax.set_xlabel("Date")
ax.set_ylabel("Number of signals")
ax.set_title("Signals Over Time")

plt.xticks(rotation=45)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 4. TRANSACTION DIRECTIONS
# ============================================================

st.header("4. Transaction Directions")

direction_counts = (
    train_transactions[
        "kirim_chiqim"
    ]
    .value_counts()
)

direction_table = (
    direction_counts
    .rename("Count")
    .to_frame()
)

st.dataframe(
    direction_table,
    use_container_width=True,
)


fig, ax = plt.subplots()

ax.bar(
    direction_counts.index.astype(str),
    direction_counts.values,
)

ax.set_xlabel("Transaction direction")
ax.set_ylabel("Number of transactions")
ax.set_title("Transaction Directions")

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 5. TRANSACTION TYPES
# ============================================================

st.header("5. Transaction Types")

type_counts = (
    train_transactions[
        "tranzaktsiya_turi"
    ]
    .value_counts()
)

type_table = (
    type_counts
    .rename("Count")
    .to_frame()
)

st.dataframe(
    type_table,
    use_container_width=True,
)


fig, ax = plt.subplots()

ax.bar(
    type_counts.index.astype(str),
    type_counts.values,
)

ax.set_xlabel("Transaction type")
ax.set_ylabel("Number of transactions")
ax.set_title("Transaction Types")

plt.xticks(rotation=30)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 6. TRANSACTION ACTIVITY PER SIGNAL
# ============================================================

st.header("6. Transaction Activity per Signal")

transaction_counts = (
    train_transactions
    .groupby("signal_id")
    .size()
)

activity_stats = pd.DataFrame(
    {
        "Statistic": [
            "Mean",
            "Median",
            "Minimum",
            "Maximum",
        ],
        "Transactions per signal": [
            transaction_counts.mean(),
            transaction_counts.median(),
            transaction_counts.min(),
            transaction_counts.max(),
        ],
    }
)

activity_stats[
    "Transactions per signal"
] = (
    activity_stats[
        "Transactions per signal"
    ].round(2)
)

st.dataframe(
    activity_stats,
    use_container_width=True,
    hide_index=True,
)


fig, ax = plt.subplots()

ax.hist(
    transaction_counts,
    bins=50,
)

ax.set_xlabel("Transactions per signal")
ax.set_ylabel("Number of signals")
ax.set_title(
    "Distribution of Transaction Activity"
)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 7. TARGET VS TRANSACTION ACTIVITY
# ============================================================

st.header("7. Target vs Transaction Activity")

activity_with_target = (
    train_signals[
        [
            "signal_id",
            "eskalatsiya",
        ]
    ]
    .merge(
        transaction_counts.rename(
            "transaction_count"
        ),
        left_on="signal_id",
        right_index=True,
        how="left",
    )
)

activity_with_target[
    "transaction_count"
] = (
    activity_with_target[
        "transaction_count"
    ].fillna(0)
)

activity_summary = (
    activity_with_target
    .groupby("eskalatsiya")[
        "transaction_count"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

st.dataframe(
    activity_summary.round(2),
    use_container_width=True,
)


fig, ax = plt.subplots()

for target_value in [0, 1]:

    values = activity_with_target.loc[
        activity_with_target["eskalatsiya"]
        == target_value,
        "transaction_count",
    ]

    ax.hist(
        values,
        bins=40,
        alpha=0.5,
        label=f"eskalatsiya = {target_value}",
    )

ax.set_xlabel("Transactions per signal")
ax.set_ylabel("Number of signals")
ax.set_title(
    "Transaction Activity by Target"
)

ax.legend()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 8. AMOUNT INDEX DISTRIBUTION
# ============================================================

st.header("8. Amount Index Distribution")

st.markdown(
    """
`miqdor_indeksi` is a synthetic numerical indicator representing
transaction amount information in the dataset.
"""
)

amount_stats = (
    train_transactions[
        "miqdor_indeksi"
    ]
    .describe()
)

st.dataframe(
    amount_stats.to_frame("Value"),
    use_container_width=True,
)


fig, ax = plt.subplots()

ax.hist(
    train_transactions[
        "miqdor_indeksi"
    ],
    bins=50,
)

ax.set_xlabel("miqdor_indeksi")
ax.set_ylabel("Number of transactions")
ax.set_title(
    "Distribution of Amount Index"
)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 9. TEMPORAL LEAKAGE INVESTIGATION
# ============================================================

st.header("9. Temporal Leakage Investigation")

st.markdown(
    """
Transactions occurring after the signal date cannot be used for
prediction because they contain information that would only become
available after the prediction time.
"""
)


# IMPORTANT:
# Make a completely separate dataframe.
# This prevents accidental modification of
# train_transactions.

leakage_df = train_transactions[
    [
        "signal_id",
        "tranzaktsiya_vaqti",
    ]
].copy()


signal_dates = train_signals[
    [
        "signal_id",
        "signal_sanasi",
    ]
].copy()


leakage_df = leakage_df.merge(
    signal_dates,
    on="signal_id",
    how="left",
)


future_transactions = leakage_df[
    leakage_df[
        "tranzaktsiya_vaqti"
    ]
    > leakage_df[
        "signal_sanasi"
    ]
]


future_count = len(
    future_transactions
)


st.metric(
    "Transactions occurring after the signal",
    f"{future_count:,}",
)


if future_count > 0:

    st.warning(
        f"""
{future_count:,} training transactions occur after their associated
signal date.

These transactions are excluded from the final feature engineering
pipeline to prevent temporal leakage.
"""
    )

else:

    st.success(
        "No future transactions were detected."
    )


st.markdown(
    """
The final feature engineering pipeline uses only transactions that
occurred on or before the signal date.
"""
)


# ============================================================
# 10. FEATURE ENGINEERING
# ============================================================

st.header("10. Feature Engineering")

st.markdown(
    """
The final model uses aggregated transaction-level features calculated
for each monitoring signal.

All features are constructed from historical transactions available
at the signal timestamp.
"""
)


feature_groups = {

    "Transaction activity": [
        "transaction_count",
        "tx_1d",
        "tx_3d",
        "tx_7d",
        "tx_30d",
    ],

    "Amount statistics": [
        "miqdor_mean",
        "miqdor_median",
        "miqdor_std",
        "miqdor_min",
        "miqdor_max",
        "miqdor_q10",
        "miqdor_q25",
        "miqdor_q75",
        "miqdor_q90",
        "miqdor_iqr",
    ],

    "Transaction direction": [
        "kirim_count",
        "chiqim_count",
        "kirim_ratio",
    ],

    "Transaction types": [
        "bank_otkazmasi_count",
        "karta_count",
        "naqd_count",
        "xalqaro_count",
    ],
}


for group_name, features in feature_groups.items():

    st.subheader(group_name)

    descriptions = []

    for feature in features:

        if feature == "transaction_count":

            description = (
                "Total number of transactions"
            )

        elif feature.startswith("tx_"):

            description = (
                "Number of transactions in "
                "the corresponding time window"
            )

        elif feature.startswith("miqdor_"):

            description = (
                "Statistical summary of "
                "the amount indicator"
            )

        elif feature in [
            "kirim_count",
            "chiqim_count",
            "kirim_ratio",
        ]:

            description = (
                "Transaction direction feature"
            )

        else:

            description = (
                "Number of transactions "
                "of the corresponding type"
            )

        descriptions.append(description)

    feature_df = pd.DataFrame(
        {
            "Feature": features,
            "Description": descriptions,
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 11. MODEL VALIDATION
# ============================================================

st.header("11. Model Validation")

st.markdown(
    """
The final model is a `HistGradientBoostingClassifier`.

Chronological validation was used because the task involves predicting
future signals from historical transaction information.
"""
)


validation_results = pd.DataFrame(
    {
        "Validation split": [
            "70% train / 30% validation",
            "80% train / 20% validation",
            "90% train / 10% validation",
        ],

        "Split date": [
            "2026-03-30",
            "2026-06-06",
            "2026-09-03",
        ],

        "Train samples": [
            9781,
            11183,
            12594,
        ],

        "Validation samples": [
            4219,
            2817,
            1406,
        ],

        "ROC-AUC": [
            0.5916,
            0.6212,
            0.6340,
        ],
    }
)

st.dataframe(
    validation_results,
    use_container_width=True,
    hide_index=True,
)


st.metric(
    "80/20 chronological validation ROC-AUC",
    "0.6212",
)


st.subheader("Walk-forward validation")

walk_forward = pd.DataFrame(
    {
        "Validation period": [
            "2026-03-30 → 2026-05-31",
            "2026-06-01 → 2026-08-31",
            "2026-09-01 → 2026-12-31",
        ],
        "Train samples": [
            9781,
            11080,
            12560,
        ],
        "Validation samples": [
            1299,
            1480,
            1440,
        ],
        "ROC-AUC": [
            0.5883,
            0.5926,
            0.6340,
        ],
    }
)

st.dataframe(
    walk_forward,
    use_container_width=True,
    hide_index=True,
)


st.info(
    """
Validation performance varies across time periods. Therefore,
the 0.6212 ROC-AUC score should be interpreted as the result of one
chronological validation split rather than a guarantee of future
performance.
"""
)


# ============================================================
# 12. KEY FINDINGS
# ============================================================

st.header("12. Key Findings")

st.markdown(
    """
### Dataset

- **14,000** training signals
- **6,000** test signals
- **6,987,663** training transactions
- **3,027,575** test transactions
- **17.18%** of training signals were escalated

### Transaction behavior

- Mean transaction count per signal: approximately **499**
- Median transaction count per signal: approximately **461**
- `kirim` is the dominant transaction direction
- `karta` and `bank_otkazmasi` are the most common transaction types

### Temporal leakage

Transactions occurring after the signal date cannot be used as
predictive features.

The final feature engineering pipeline therefore uses only historical
transactions available at the signal timestamp.

### Machine learning

The final model is a
`HistGradientBoostingClassifier`.

The chronological 80/20 validation ROC-AUC was **0.6212**.

Performance varied across chronological validation periods, indicating
that temporal variation is relevant to model evaluation.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fintech AI Hackathon | Synthetic financial monitoring dataset"
)