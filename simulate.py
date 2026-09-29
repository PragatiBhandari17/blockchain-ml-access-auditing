"""
simulate.py
-----------
Generates synthetic blockchain access-request traffic for a fine-grained
access control scheme (as in the base paper's smart contract Search()
function). Produces a labeled event log with:

    timestamp, user_id, user_type, keyword, policy_match

user_type is ground truth (legit / slow_low / burst / enumeration) and is
ONLY used for evaluation later -- neither the baseline rule nor the ML
model gets to see it.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

SIM_DURATION_SEC = 24 * 60 * 60          # simulate a 24 hour window
N_LEGIT_USERS = 45

# vocabulary of "keywords" a data user might search for (stand-in for
# attribute / keyword search space in the base paper)
KEYWORD_POOL = [f"kw_{i}" for i in range(40)]


def _gen_legit_user(user_id):
    """
    A legitimate user's requests follow a log-normal inter-arrival
    distribution -- occasional bursts of activity (e.g. downloading a
    batch of files) but mostly spaced out, human-like usage.
    Each user has their own mean request rate.
    """
    mean_gap = RNG.uniform(60, 400)     # seconds, differs per user
    sigma = 0.7

    events = []
    t = RNG.uniform(0, mean_gap)
    while t < SIM_DURATION_SEC:
        # each user has a small favorite subset of keywords (diverse but
        # not random every time -- realistic search behaviour)
        fav_keywords = RNG.choice(KEYWORD_POOL, size=5, replace=False)
        keyword = RNG.choice(fav_keywords)
        policy_match = RNG.random() > 0.03   # legit users rarely fail policy

        events.append((t, user_id, "legit", keyword, policy_match))
        gap = RNG.lognormal(mean=np.log(mean_gap), sigma=sigma)
        t += gap
    return events


def _gen_slow_low_attacker(user_id, interval_threshold=10, epsilon_range=(0.5, 3.0)):
    """
    Attacker who has read the smart contract's lockout rule and paces
    requests just ABOVE the fixed interval threshold, continuously,
    to enumerate ciphertext/attributes without ever tripping
    nowTime - lastTime < interval.
    Low keyword diversity (systematic enumeration) and elevated
    policy-fail rate (probing attributes it doesn't hold).
    """
    events = []
    t = RNG.uniform(0, 50)
    n_requests = 700
    for _ in range(n_requests):
        if t > SIM_DURATION_SEC:
            break
        # systematic sweep through keyword pool -> low entropy in any window
        keyword = KEYWORD_POOL[int(t) % len(KEYWORD_POOL)]
        policy_match = RNG.random() > 0.4   # frequently fails (probing)
        events.append((t, user_id, "slow_low", keyword, policy_match))
        gap = interval_threshold + RNG.uniform(*epsilon_range)
        t += gap
    return events


def _gen_burst_attacker(user_id):
    """
    Naive attacker: rapid-fire requests, far below the interval
    threshold. The base paper's rule SHOULD catch this easily -- it is
    a positive control to confirm the baseline isn't broken, only
    incomplete.
    """
    events = []
    burst_start = RNG.uniform(0, SIM_DURATION_SEC - 600)
    t = burst_start
    n_requests = 150
    for _ in range(n_requests):
        keyword = RNG.choice(KEYWORD_POOL)
        policy_match = RNG.random() > 0.5
        events.append((t, user_id, "burst", keyword, policy_match))
        gap = RNG.uniform(0.2, 1.5)
        t += gap
    return events


def _gen_enumeration_attacker(user_id, interval_threshold=10):
    """
    Attacker paces moderately (sometimes above, sometimes just under
    threshold -- mixed, to look 'noisy' rather than obviously scripted)
    but systematically sweeps the attribute/keyword space with a very
    high policy-fail rate. Represents a keyword-guessing / attribute
    inference attack from the paper's threat model.
    """
    events = []
    t = RNG.uniform(0, 50)
    n_requests = 400
    idx = 0
    for _ in range(n_requests):
        if t > SIM_DURATION_SEC:
            break
        keyword = KEYWORD_POOL[idx % len(KEYWORD_POOL)]
        idx += 1
        policy_match = RNG.random() > 0.6      # fails policy most of the time
        events.append((t, user_id, "enumeration", keyword, policy_match))
        gap = RNG.uniform(interval_threshold * 0.6, interval_threshold * 1.8)
        t += gap
    return events


def generate_traffic():
    all_events = []

    for uid in range(N_LEGIT_USERS):
        all_events.extend(_gen_legit_user(f"user_{uid}"))

    # a handful of independent malicious accounts per attack strategy,
    # rather than one giant single account -- more realistic and gives
    # the evaluation more than one attacker instance per class
    for i in range(2):
        all_events.extend(_gen_slow_low_attacker(f"attacker_slow_low_{i}"))
    all_events.extend(_gen_burst_attacker("attacker_burst_0"))
    for i in range(2):
        all_events.extend(_gen_enumeration_attacker(f"attacker_enum_{i}"))

    df = pd.DataFrame(
        all_events,
        columns=["timestamp", "user_id", "user_type", "keyword", "policy_match"],
    )
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["is_attack"] = (df["user_type"] != "legit").astype(int)
    return df


if __name__ == "__main__":
    df = generate_traffic()
    df.to_csv("traffic_log.csv", index=False)
    print(df["user_type"].value_counts())
    print(f"Total events: {len(df)}")
