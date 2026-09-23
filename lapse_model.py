import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_recall_curve, make_scorer,
    precision_score, recall_score, f1_score,
)

RANDOM_STATE = 42
N_FOLDS = 5

# Operating-point choice (see docstring #7). Lower = catch more lapsers but more
# false alarms; higher = fewer, higher-confidence flags. Given that lapse risk
# sits in low-value customers, a higher target (e.g. 0.50) is also defensible.
RECALL_TARGET = 0.70

FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "avg_order_value",
    "tenure_days",
    "delivery_share",
    "avg_items_per_order",
    "item_diversity",
    "tip_rate",
    "modifiers_per_order",
]
TARGET = "lapsed"


def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    # Any customer with genuinely no valid product lines can have NaN features;
    # fill numerically. (tip_rate can be NaN if monetary_total was 0.)
    df[FEATURES] = df[FEATURES].fillna(0)
    return df


def report_cv(name, estimator, X, y, cv):
    """Run stratified CV and print mean +/- std for each metric."""
    # zero_division=0 keeps the majority-class baseline (which never predicts a
    # positive, so precision/recall are undefined) from spamming warnings.
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring)
    print(f"\n=== {name} ({N_FOLDS}-fold stratified CV) ===")
    for m in scoring:
        vals = scores[f"test_{m}"]
        print(f"  {m:10s}: {vals.mean():.3f} +/- {vals.std():.3f}")
    return scores


def main(csv_path):
    df = load_data(csv_path)
    X = df[FEATURES].values
    y = df[TARGET].values

    n, pos = len(y), int(y.sum())
    print(f"Rows: {n}   Lapsed (positive): {pos} ({100 * pos / n:.1f}%)")
    print(f"Test-fold positives per fold: ~{pos // N_FOLDS}  "
          f"(small -> expect wide confidence intervals)")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ----- 1. Dumb baseline: always predict the majority class -----
    baseline = DummyClassifier(strategy="most_frequent")
    report_cv("Baseline (predict majority = 'stays')", baseline, X, y, cv)
    print("  ^ Any real model MUST beat this baseline's accuracy to be useful.")

    # ----- 2. Primary model: Logistic Regression -----
    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    report_cv("Logistic Regression (balanced)", logreg, X, y, cv)

    # ----- 3. Comparison model: Random Forest -----
    # At this sample size we expect little-to-no reliable lift over logistic;
    # reporting that is a legitimate, mature conclusion.
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)
    report_cv("Random Forest (balanced)", rf, X, y, cv)

    # ----- 4. Logistic coefficients (standardized => comparable magnitudes) -----
    logreg.fit(X, y)
    coefs = logreg.named_steps["clf"].coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    print("\n=== Logistic coefficients (sorted by |effect|) ===")
    print("  positive => pushes toward LAPSE, negative => toward STAYING")
    for i in order:
        print(f"  {FEATURES[i]:22s}: {coefs[i]:+.3f}")

    # ----- 5. Threshold analysis from out-of-fold probabilities -----
    oof_proba = cross_val_predict(
        logreg, X, y, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y, oof_proba)
    print("\n=== Threshold trade-off (out-of-fold) ===")
    print("  thresh   precision   recall")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        idx = np.searchsorted(thr, t)
        idx = min(idx, len(prec) - 1)
        print(f"  {t:.2f}     {prec[idx]:.3f}       {rec[idx]:.3f}")

    # Pick the lowest threshold that reaches the recall target (see RECALL_TARGET
    # at top). This is a deliberate business choice, not the arbitrary 0.5.
    ok = np.where(rec[:-1] >= RECALL_TARGET)[0]
    chosen_t = thr[ok[-1]] if len(ok) else 0.5
    flagged = oof_proba >= chosen_t
    print(f"\n  Chosen threshold for recall >= {RECALL_TARGET}: {chosen_t:.3f}")
    print(f"  Flags {flagged.sum()} of {n} customers as at-risk.")

    # ----- 6. Dollar translation -----
    if "monetary_total" in df.columns:
        total_rev = df["monetary_total"].sum()
        flagged_rev = df.loc[flagged, "monetary_total"].sum()
        print("\n=== Business framing ===")
        print(f"  At-risk customers flagged: {flagged.sum()} "
              f"({100 * flagged.sum() / n:.1f}% of the cohort)")
        print(f"  Feature-window revenue they represent: "
              f"{100 * flagged_rev / total_rev:.1f}% of cohort revenue")
        print("  NOTE: if this revenue share is well below the customer share,")
        print("  lapse risk sits in LOW-VALUE customers -> triage retention")
        print("  spend rather than blanket-targeting the whole flagged list.")

    print("\nCaveat to keep in the writeup: the data spans 16 months (a single "
          "seasonal cycle), so a model trained on the 12-month feature window "
          "predicting the 4-month window (Mar-Jun 2026) cannot separate true "
          "lapse from a seasonal lull. Results are correlational, not causal.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "features.csv"
    main(path)