"""
ml_model.py
-----------
Trains an unsupervised anomaly detector (Isolation Forest) on the
behavioral feature vectors from features.py, then scores every
request in the log.

Why unsupervised: in a real deployment we will not have labeled
attack data up front -- the whole point is to catch attacks we have
not seen labeled examples of yet. Isolation Forest only needs to learn
what "normal" looks like from the bulk of (mostly legitimate) traffic.

We deliberately do NOT use is_attack / user_type as a training input --
those columns exist only so we can evaluate the model afterwards.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS

CONTAMINATION = 0.1  # assumed upper bound on fraction of traffic that's malicious
RANDOM_STATE = 42


def score_requests(feat_df: pd.DataFrame):
    X = feat_df[FEATURE_COLUMNS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(X_scaled)

    # decision_function: higher = more normal, lower = more anomalous.
    # We flip sign so higher risk_score = more suspicious (more intuitive
    # for feeding into a smart contract's penalty logic).
    raw_scores = model.decision_function(X_scaled)
    risk_score = -raw_scores
    risk_score_norm = (risk_score - risk_score.min()) / (
        risk_score.max() - risk_score.min()
    )

    predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
    flagged_ml = (predictions == -1).astype(int)

    out = feat_df.copy()
    out["risk_score"] = risk_score_norm
    out["flagged_ml"] = flagged_ml
    return out, model, scaler


if __name__ == "__main__":
    feats = pd.read_csv("features.csv")
    scored, _, _ = score_requests(feats)
    scored.to_csv("ml_scored.csv", index=False)

    summary = scored.groupby("user_type")["flagged_ml"].mean()
    print("Fraction of requests flagged by ML model, per user type:")
    print(summary)

    print("\nMean risk score per user type:")
    print(scored.groupby("user_type")["risk_score"].mean())
