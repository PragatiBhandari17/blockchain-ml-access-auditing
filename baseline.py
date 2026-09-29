"""
baseline.py
-----------
Faithful reimplementation of the base paper's Algorithm 4 (Search)
smart-contract audit logic:

    if nowTime - lastTime < interval:
        lockNum += 1
        lockTime = nowTime + 2 ** (lockNum - 1)
        # request flagged illegal / locked
    else:
        lastTime = nowTime
        # request treated as legal

interval = 10 seconds, matching the paper's own worked example (Fig. 3).

This module walks the event log per user, in chronological order,
replaying the smart contract's exact per-user state machine
(lastTime, lockNum, lockTime) and produces one flag per request:
    flagged_baseline = 1  -> contract would treat this as illegal access
    flagged_baseline = 0  -> contract would treat this as legal access

Note: a user whose request arrives while lockTime > nowTime is also
denied service (still locked out from a previous violation), which we
count as "flagged" too, matching the contract's require() semantics.
"""

import numpy as np
import pandas as pd

INTERVAL = 10  # seconds, as used in the paper's own example


def apply_baseline_rule(df: pd.DataFrame, interval: int = INTERVAL) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    flags = []
    lock_times_out = []

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        last_time = None
        lock_num = 0
        lock_time = -np.inf

        for t in group["timestamp"]:
            still_locked = t < lock_time

            if last_time is not None and (t - last_time) < interval:
                lock_num += 1
                lock_time = t + 2 ** (lock_num - 1)
                flagged = 1
            else:
                last_time = t
                flagged = 1 if still_locked else 0

            flags.append(flagged)
            lock_times_out.append(lock_time)

    df = df.copy()
    df["flagged_baseline"] = flags
    df["lock_time_baseline"] = lock_times_out
    return df


if __name__ == "__main__":
    raw = pd.read_csv("traffic_log.csv")
    out = apply_baseline_rule(raw)
    out.to_csv("baseline_flags.csv", index=False)

    summary = out.groupby("user_type")["flagged_baseline"].mean()
    print("Fraction of requests flagged by BASELINE rule, per user type:")
    print(summary)
