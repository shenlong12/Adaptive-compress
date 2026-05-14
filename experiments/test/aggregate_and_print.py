"""
aggregate_and_print.py
======================
If the main sweep already ran, call this to reprint tables from the CSV
without re-running inference.

Usage:
    python aggregate_and_print.py [--csv results/ablation_tau_results.csv]
"""

import argparse, json, math
from pathlib import Path
import pandas as pd


def print_table(rows):
    header = (
        f"{'Threshold τ':>14} | "
        f"{'Dense Routing (%)':>17} | "
        f"{'Average Acc. (%)':>16} | "
        f"{'GSM8K (%)':>10} | "
        f"{'WikiText2 PPL':>13} | "
        f"{'Gain over Static':>16}"
    )
    sep = "-" * len(header)
    print(f"\nSADS Threshold Ablation  (Sparsity = 40%)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        tau = r["tau"]
        dense_r = r.get("avg_fb_rate", 0)
        if tau == "inf":
            dense_r = 0.00
        elif tau == "0.0":
            dense_r = 100.00
        gain = r["gain_over_static"]
        gain_str = f"{gain:+.2f}" if tau != "inf" else " 0.00"
        print(
            f"{tau:>14} | "
            f"{dense_r:>17.2f} | "
            f"{r['avg_acc']:>16.2f} | "
            f"{r['gsm8k_acc']:>10.2f} | "
            f"{r['wikitext2_ppl']:>13.2f} | "
            f"{gain_str:>16}"
        )
    print(sep)


def print_latex(rows):
    print("\n% ── LaTeX Table ──────────────────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{SADS Threshold Ablation at 40\% Sparsity (LLaMA-3-8B). "
          r"Average Acc.\ is the mean over ARC-e, ARC-c, HellaSwag, and OBQA. "
          r"Gain over Static compares to $\tau{=}\infty$ (static sparse inference).}")
    print(r"\label{tab:ablation_tau}")
    print(r"\resizebox{\linewidth}{!}{%")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"$\tau$ & Dense Routing (\%) & Avg.\ Acc.\ (\%) & GSM8K (\%) "
          r"& WikiText2 PPL $\downarrow$ & Gain over Static \\")
    print(r"\midrule")
    for r in rows:
        tau = r["tau"]
        tau_str = r"$\infty$" if tau == "inf" else f"${tau}$"
        dense_r = r.get("avg_fb_rate", 0)
        if tau == "inf":
            dense_r = 0.00
        elif tau == "0.0":
            dense_r = 100.00
        gain = r["gain_over_static"]
        gain_str = f"$0.00$" if tau == "inf" else f"${gain:+.2f}$"
        print(
            f"{tau_str} & {dense_r:.2f} & {r['avg_acc']:.2f} & "
            f"{r['gsm8k_acc']:.2f} & {r['wikitext2_ppl']:.2f} & {gain_str} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")


def analyze(rows):
    static_row  = next(r for r in rows if r["tau"] == "inf")
    dense_row   = next((r for r in rows if r["tau"] == "0.0"), None)
    inner_rows  = [r for r in rows if r["tau"] not in ("inf", "0.0")]

    print("\n── Analysis ──────────────────────────────────────────────────────")

    # Q1: monotone recovery?
    gains = [r["gain_over_static"] for r in inner_rows]
    taus  = [float(r["tau"]) for r in inner_rows]
    sorted_pairs = sorted(zip(taus, gains), reverse=True)  # high tau → low tau
    is_monotone = all(sorted_pairs[i][1] <= sorted_pairs[i+1][1]
                      for i in range(len(sorted_pairs)-1))
    print(f"\n1. Monotone quality recovery as τ decreases: {'YES ✓' if is_monotone else 'NOT strictly monotone'}")

    # Q2: best working point (max gain with min dense routing)
    if inner_rows:
        best = max(inner_rows, key=lambda r: r["gain_over_static"] / max(r.get("avg_fb_rate",1), 1))
        print(f"\n2. Recommended working point: τ = {best['tau']}  "
              f"(avg_acc={best['avg_acc']:.2f}%, dense_routing={best.get('avg_fb_rate',0):.1f}%, "
              f"gain={best['gain_over_static']:+.2f}%)")

    # Q3: GSM8K vs avg
    print(f"\n3. GSM8K vs Average Acc. gains:")
    static_gsm  = static_row["gsm8k_acc"]
    static_avg  = static_row["avg_acc"]
    for r in inner_rows:
        gsm_gain = r["gsm8k_acc"] - static_gsm
        avg_gain = r["gain_over_static"]
        stronger = "GSM8K" if gsm_gain > avg_gain else "Avg MC"
        print(f"   τ={r['tau']}: GSM8K gain={gsm_gain:+.2f}%  Avg gain={avg_gain:+.2f}%  → {stronger} benefits more")

    # Q4: supports the paper claim?
    max_gain = max(r["gain_over_static"] for r in inner_rows) if inner_rows else 0
    any_gain = max_gain > 0
    print(f"\n4. Supports claim 'controllable trade-off without recovery fine-tuning': "
          f"{'YES ✓' if any_gain else 'WEAK — gains not observed'}")
    if any_gain:
        print(f"   (Max avg accuracy recovery: {max_gain:+.2f}% over static sparse, "
              f"at τ={max(inner_rows, key=lambda r: r['gain_over_static'])['tau']})")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/ablation_tau_results.csv")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    # ensure tau is stored as string
    for r in rows:
        r["tau"] = str(r["tau"])

    print_table(rows)
    print_latex(rows)
    analyze(rows)


if __name__ == "__main__":
    main()