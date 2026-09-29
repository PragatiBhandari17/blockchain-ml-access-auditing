"""
features.py
-----------
Turns the raw event log into a per-request feature matrix that a
learning model can use to judge "does this request look like normal
behaviour for this user, or not" -- richer than the base paper's single
inter-arrival check.

Features (per request, computed only from that user's PAST requests --
no lookahead / no label leakage):
    inter_arrival          seconds since this user's previous request
    roll_mean_gap          rolling mean of last W inter-arrival times
    roll_std_gap           rolling std of last W inter-arrival times
    req_count_5min         # requests by this user in the last 5 minutes
    req_count_1hr          # requests by this user in the last 1 hour
    keyword_entropy        Shannon entropy of keywords in last W requests
                            (low entropy => systematic/scripted sweeping)
    policy_fail_rate       fraction of last W requests that failed policy
                            match (probing attributes the user lacks)
"""

import numpy as np
import pandas as pd

WINDOW = 10          # rolling window size (# of past requests)


def _entropy(values):
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    feature_rows = []

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        timestamps = group["timestamp"].values
        keywords = group["keyword"].values
        policy_matches = group["policy_match"].values

        prev_gaps = []  # history of inter-arrival times for this user

        for i in range(len(group)):
            t = timestamps[i]

            inter_arrival = t - timestamps[i - 1] if i > 0 else np.nan

            window_gaps = prev_gaps[-WINDOW:]
            roll_mean_gap = np.mean(window_gaps) if window_gaps else np.nan
            roll_std_gap = np.std(window_gaps) if len(window_gaps) > 1 else 0.0

            # counts in trailing time windows
            past_times = timestamps[:i]
            req_count_5min = int(np.sum(past_times > t - 300))
            req_count_1hr = int(np.sum(past_times > t - 3600))

            window_keywords = keywords[max(0, i - WINDOW):i]
            keyword_entropy = _entropy(window_keywords)

            window_policy = policy_matches[max(0, i - WINDOW):i]
            policy_fail_rate = (
                1.0 - np.mean(window_policy) if len(window_policy) > 0 else 0.0
            )

            feature_rows.append(
                {
                    "timestamp": t,
                    "user_id": user_id,
                    "inter_arrival": inter_arrival,
                    "roll_mean_gap": roll_mean_gap,
                    "roll_std_gap": roll_std_gap,
                    "req_count_5min": req_count_5min,
                    "req_count_1hr": req_count_1hr,
                    "keyword_entropy": keyword_entropy,
                    "policy_fail_rate": policy_fail_rate,
                }
            )

            if i > 0:
                prev_gaps.append(inter_arrival)

    feat_df = pd.DataFrame(feature_rows)

    # merge ground-truth labels back in (for evaluation only)
    merged = pd.merge(
        feat_df,
        df[["timestamp", "user_id", "user_type", "is_attack"]],
        on=["timestamp", "user_id"],
        how="left",
    )
    merged.sort_values(["user_id", "timestamp"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # first request per user has no inter_arrival / rolling stats yet
    merged.fillna(
        {
            "inter_arrival": merged["inter_arrival"].median(),
            "roll_mean_gap": merged["inter_arrival"].median(),
            "roll_std_gap": 0.0,
        },
        inplace=True,
    )
    return merged


FEATURE_COLUMNS = [
    "inter_arrival",
    "roll_mean_gap",
    "roll_std_gap",
    "req_count_5min",
    "req_count_1hr",
    "keyword_entropy",
    "policy_fail_rate",
]

if __name__ == "__main__":
    raw = pd.read_csv("traffic_log.csv")
    feats = build_features(raw)
    feats.to_csv("features.csv", index=False)
    print(feats.head())
    print(f"Feature rows: {len(feats)}")
