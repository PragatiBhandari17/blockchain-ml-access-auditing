# ML-Augmented Access Audit — Simulation Pipeline

This is a self-contained simulation that reproduces the base paper's
smart-contract audit rule (Yan et al., 2023, *Algorithm 4: Search*),
demonstrates its blind spot, and shows how an unsupervised ML model
(Isolation Forest) closes that gap.

## Files

| File | Purpose |
|---|---|
| `simulate.py` | Generates 24h of synthetic access-request traffic: legitimate users + 3 attacker profiles (`slow_low`, `burst`, `enumeration`) |
| `features.py` | Computes rolling behavioral features per request (no label leakage — only past data used) |
| `baseline.py` | Faithful reimplementation of the paper's fixed-interval lockout rule (`interval=10s`, `lockTime = now + 2^(lockNum-1)`) |
| `ml_model.py` | Trains an unsupervised Isolation Forest on the feature vectors and outputs a continuous risk score per request |
| `evaluate.py` | Orchestrates the full pipeline, computes metrics, and generates all plots |

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib
python3 evaluate.py
```

Everything is written to `./outputs/`:
- `results_summary.txt` — headline numbers (precision/recall/F1/AUC, per-type breakdown)
- `metrics_by_type.csv` — the per-attack-type table
- `full_results.csv` — every simulated request with all features, flags, and risk scores (raw data for further analysis)
- `fig1`–`fig6` PNGs — the charts described below

## The core finding

The base paper's rule flags a request as illegal only if:
```
nowTime - lastTime < interval    (interval = 10 seconds)
```
An attacker who paces requests just *above* 10 seconds apart (the
`slow_low` profile) evades this rule by construction — every single
request individually satisfies the timing check, even while the
attacker sweeps the entire keyword/attribute space over time.

The ML model instead scores each request using a **rolling window of
behavioral features** (request volume, keyword entropy, policy-fail
rate, timing regularity) rather than a single inter-arrival check, so
sustained anomalous *patterns* are visible even when no individual
request looks suspicious in isolation.

## What each figure shows

- **Fig 1** — the raw simulated traffic timeline (sanity check that all 4 traffic types are present and distributed as intended)
- **Fig 2** — inter-arrival time histograms: legit vs. `slow_low`, with the 10s threshold marked — shows why the fixed rule structurally cannot separate them well
- **Fig 3** — the headline bar chart: detection rate per attack type, baseline vs. ML. Baseline: 0% on `slow_low`. ML: ~27% direct flag rate, but far higher relative risk score (see summary text)
- **Fig 4** — case study of one `slow_low` attacker instance over its whole session: baseline flag stays at 0 throughout; ML risk score repeatedly exceeds the legitimate population's 95th percentile
- **Fig 5** — ROC and Precision-Recall curves for the ML risk score (threshold-independent evaluation) — ROC-AUC ≈ 0.96
- **Fig 6** — false-lockout rate on legitimate users, baseline vs. ML (the cost side of the comparison — ML has a higher false-flag rate than the near-zero baseline, which is an honest trade-off to discuss in your report)

## How to extend this for your thesis

1. **Try other models** — swap `IsolationForest` in `ml_model.py` for `OneClassSVM`, or build an LSTM autoencoder on the sequence of a user's raw inter-arrival times (needs restructuring into sequences rather than flat feature rows).
2. **Add more attack profiles** — e.g., a "distributed slow" attacker that uses multiple fake accounts, each individually slow, to further evade even the ML model — good stress test for a "limitations" section.
3. **Tune the threshold, don't just hard-flag** — right now `flagged_ml` uses Isolation Forest's built-in `contamination`-based cutoff. For the smart contract integration chapter, replace this with a percentile-based or adaptive threshold and show how it trades off precision vs. recall (use `fig5` data for this).
4. **Feed `risk_score` into a graduated penalty**, not a binary lock — e.g. `lockTime = now + 2^(risk_bucket - 1)` where `risk_bucket` depends on the score, rather than only on `lockNum`. This is the actual smart-contract-side contribution the next chapter of your thesis should describe (Solidity pseudocode + an off-chain oracle that submits the score on-chain).
5. **Replace synthetic data with real access logs** if you can get any (even from a mock file-sharing app you build) to strengthen external validity — synthetic-only results are a limitation you should state explicitly.
