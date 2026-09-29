"""
evaluate.py
-----------
Ties the whole pipeline together:
    simulate traffic -> extract features -> apply baseline rule
    -> score with ML model -> evaluate both -> generate plots

Run this single file to reproduce everything end to end.

Outputs (written to ./outputs/):
    results_summary.txt        headline numbers for the report
    metrics_by_type.csv        per-attack-type recall for both systems
    fig1_traffic_timeline.png
    fig2_interarrival_hist.png
    fig3_detection_by_type.png
    fig4_slowlow_case_study.png
    fig5_roc_pr_curves.png
    fig6_false_lockout_rate.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from simulate import generate_traffic
from features import build_features, FEATURE_COLUMNS
from baseline import apply_baseline_rule, INTERVAL
from ml_model import score_requests

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

TYPE_COLORS = {
    "legit": "#4C72B0",
    "slow_low": "#DD8452",
    "burst": "#C44E52",
    "enumeration": "#8172B2",
}
TYPE_ORDER = ["legit", "slow_low", "enumeration", "burst"]


def run_pipeline():
    print("Step 1: generating synthetic traffic...")
    raw = generate_traffic()

    print("Step 2: extracting behavioral features...")
    feats = build_features(raw)

    print("Step 3: applying baseline fixed-interval rule...")
    baseline_out = apply_baseline_rule(raw)

    print("Step 4: scoring requests with ML anomaly model...")
    ml_out, model, scaler = score_requests(feats)

    # merge everything into one master dataframe keyed on (user_id, timestamp)
    merged = pd.merge(
        ml_out,
        baseline_out[["user_id", "timestamp", "flagged_baseline", "lock_time_baseline"]],
        on=["user_id", "timestamp"],
        how="left",
    )
    merged.sort_values("timestamp", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged.to_csv(os.path.join(OUT_DIR, "full_results.csv"), index=False)
    return merged


def compute_metrics(df):
    y_true = df["is_attack"].values

    metrics = {}
    for name, col in [("baseline", "flagged_baseline"), ("ml", "flagged_ml")]:
        y_pred = df[col].values
        metrics[name] = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "false_flag_rate_on_legit": df.loc[df["is_attack"] == 0, col].mean(),
        }

    # threshold-independent metrics for the ML risk score
    fpr, tpr, _ = roc_curve(y_true, df["risk_score"].values)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, df["risk_score"].values)
    metrics["ml"]["roc_auc"] = roc_auc
    metrics["ml"]["pr_auc"] = ap

    # per-attack-type recall breakdown -- the key evidence table
    per_type = (
        df.groupby("user_type")[["flagged_baseline", "flagged_ml", "risk_score"]]
        .mean()
        .reindex(TYPE_ORDER)
    )
    per_type.rename(
        columns={
            "flagged_baseline": "baseline_flag_rate",
            "flagged_ml": "ml_flag_rate",
            "risk_score": "ml_mean_risk_score",
        },
        inplace=True,
    )
    return metrics, per_type


def write_summary(metrics, per_type):
    lines = []
    lines.append("=" * 70)
    lines.append("ML-AUGMENTED AUDIT vs BASE PAPER FIXED-THRESHOLD RULE")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Fixed interval threshold used by baseline: {INTERVAL} seconds")
    lines.append("")
    lines.append("-- Overall attack detection (all attack types pooled) --")
    for name in ["baseline", "ml"]:
        m = metrics[name]
        lines.append(f"[{name.upper()}]")
        lines.append(f"  Precision              : {m['precision']:.3f}")
        lines.append(f"  Recall                 : {m['recall']:.3f}")
        lines.append(f"  F1 score               : {m['f1']:.3f}")
        lines.append(f"  False-flag rate (legit): {m['false_flag_rate_on_legit']:.4f}")
        if "roc_auc" in m:
            lines.append(f"  ROC-AUC (risk score)   : {m['roc_auc']:.3f}")
            lines.append(f"  PR-AUC  (risk score)   : {m['pr_auc']:.3f}")
        lines.append("")

    lines.append("-- Per attack-type flag rate (the key result) --")
    lines.append(per_type.to_string(float_format=lambda x: f"{x:.4f}"))
    lines.append("")
    lines.append(
        "Interpretation: the baseline's fixed 10s-interval rule flags\n"
        "'slow_low' attackers (paced just above the threshold) at a rate\n"
        "close to zero, because each individual request legitimately\n"
        "satisfies (now - last) >= interval. The ML model instead looks at\n"
        "aggregate behaviour (request volume, keyword diversity, policy-fail\n"
        "rate) over a rolling window, so it assigns this attacker a\n"
        "substantially higher risk score than ordinary users even though no\n"
        "single request would trip the paper's rule."
    )

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "results_summary.txt"), "w") as f:
        f.write(text)

    per_type.to_csv(os.path.join(OUT_DIR, "metrics_by_type.csv"))


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_traffic_timeline(df):
    fig, ax = plt.subplots(figsize=(11, 4))
    for utype in TYPE_ORDER:
        sub = df[df["user_type"] == utype]
        ax.scatter(
            sub["timestamp"] / 3600,
            [utype] * len(sub),
            s=4,
            alpha=0.5,
            color=TYPE_COLORS[utype],
            label=utype,
        )
    ax.set_xlabel("Simulated time (hours)")
    ax.set_ylabel("Traffic source")
    ax.set_title("Fig 1: Simulated 24-Hour Access Request Timeline")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_traffic_timeline.png"))
    plt.close(fig)


def plot_interarrival_hist(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    legit = df[df["user_type"] == "legit"]["inter_arrival"]
    slow = df[df["user_type"] == "slow_low"]["inter_arrival"]

    ax.hist(legit.clip(upper=120), bins=40, alpha=0.6, label="legit",
            color=TYPE_COLORS["legit"], density=True)
    ax.hist(slow, bins=40, alpha=0.6, label="slow_low attacker",
            color=TYPE_COLORS["slow_low"], density=True)
    ax.axvline(INTERVAL, color="black", linestyle="--",
               label=f"baseline threshold ({INTERVAL}s)")
    ax.set_xlabel("Inter-arrival time (seconds, clipped at 120s for readability)")
    ax.set_ylabel("Density")
    ax.set_title("Fig 2: Inter-arrival Times — Legit vs Slow-and-Low Attacker")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_interarrival_hist.png"))
    plt.close(fig)


def plot_detection_by_type(per_type):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(TYPE_ORDER))
    width = 0.35

    ax.bar(x - width / 2, per_type["baseline_flag_rate"], width,
           label="Baseline (fixed threshold)", color="#4C72B0")
    ax.bar(x + width / 2, per_type["ml_flag_rate"], width,
           label="ML (Isolation Forest)", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_ORDER)
    ax.set_ylabel("Fraction of requests flagged")
    ax.set_title("Fig 3: Detection Rate by Traffic Type — Baseline vs ML")
    ax.legend()
    for i, v in enumerate(per_type["baseline_flag_rate"]):
        ax.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(per_type["ml_flag_rate"]):
        ax.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_detection_by_type.png"))
    plt.close(fig)


def plot_slowlow_case_study(df):
    """
    The 'money shot': pick one slow_low attacker instance and show, over
    time, that the baseline NEVER flags it while the ML risk score sits
    persistently elevated above the legit population's typical range.
    """
    attacker_id = sorted(
        df.loc[df["user_type"] == "slow_low", "user_id"].unique()
    )[0]
    sub = df[df["user_id"] == attacker_id].sort_values("timestamp")

    legit_risk_95 = df.loc[df["user_type"] == "legit", "risk_score"].quantile(0.95)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].step(sub["timestamp"] / 3600, sub["flagged_baseline"],
                 where="post", color="#4C72B0")
    axes[0].set_ylabel("Baseline flag\n(0=allowed,1=locked)")
    axes[0].set_title(
        f"Fig 4: Case Study — attacker '{attacker_id}' "
        f"(paced just above the {INTERVAL}s threshold)"
    )
    axes[0].set_yticks([0, 1])

    axes[1].plot(sub["timestamp"] / 3600, sub["risk_score"], color="#DD8452",
                 label="ML risk score")
    axes[1].axhline(legit_risk_95, color="black", linestyle="--",
                     label="95th percentile of legit users' risk score")
    axes[1].set_ylabel("ML risk score")
    axes[1].set_xlabel("Simulated time (hours)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_slowlow_case_study.png"))
    plt.close(fig)


def plot_roc_pr(df):
    y_true = df["is_attack"].values
    scores = df["risk_score"].values

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(fpr, tpr, color="#4C72B0", label=f"ROC (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — ML Risk Score")
    axes[0].legend()

    axes[1].plot(rec, prec, color="#DD8452", label=f"PR (AP={pr_auc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — ML Risk Score")
    axes[1].legend()

    fig.suptitle("Fig 5: Threshold-Independent Evaluation of ML Risk Score")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_roc_pr_curves.png"))
    plt.close(fig)


def plot_false_lockout(metrics):
    fig, ax = plt.subplots(figsize=(6, 5))
    names = ["Baseline", "ML"]
    values = [
        metrics["baseline"]["false_flag_rate_on_legit"],
        metrics["ml"]["false_flag_rate_on_legit"],
    ]
    bars = ax.bar(names, values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Fraction of legitimate requests wrongly flagged")
    ax.set_title("Fig 6: False-Lockout Rate on Legitimate Users")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_false_lockout_rate.png"))
    plt.close(fig)


def main():
    df = run_pipeline()
    metrics, per_type = compute_metrics(df)
    write_summary(metrics, per_type)

    print("\nGenerating plots...")
    plot_traffic_timeline(df)
    plot_interarrival_hist(df)
    plot_detection_by_type(per_type)
    plot_slowlow_case_study(df)
    plot_roc_pr(df)
    plot_false_lockout(metrics)

    print(f"\nAll outputs written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
