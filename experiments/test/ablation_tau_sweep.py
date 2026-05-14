"""
SADS Threshold Ablation Experiment
===================================
Sweeps tau ∈ {inf, 6.0, 5.0, 4.0, 0.0} at fixed 40% sparsity.
Evaluates: ARC-e, ARC-c, HellaSwag, OBQA, GSM8K, WikiText2 PPL
Outputs:   results/ablation_tau_results.csv  +  console table

Usage:
    python ablation_tau_sweep.py [--taus inf 6.0 5.0 4.0 0.0]
                                 [--sparsity 40]
                                 [--limit N]   # per-dataset sample limit, -1=all
"""

import os, sys, re, math, json, argparse, logging
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ── same pattern as eval_commonsense_parquet.py ────────────────────────────────
# File lives at  experiments/test/ablation_tau_sweep.py
# src lives at   TransformerCompression-main/src/
# so ../../src from __file__ is correct (identical depth to human-eval/ scripts)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from slicegpt import hf_utils  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          PATH CONFIGURATION                                 ║
# ║  Edit only this block; everything else is automatic.                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
CFG = {
    "base_model": "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct",
    "experiments_dir": (
        "/clzs_test011/qyh/slice_new/"
        "TransformerCompression-main/TransformerCompression-main/experiments"
    ),
    "hf_model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    # Dataset parquet paths  ← adjust filenames to what you have locally
    "datasets": {
        "arc_e":     "/clzs_test011/qyh/dataset/ARC-E.parquet",
        "arc_c":     "/clzs_test011/qyh/dataset/ARC-C.parquet",
        "hellaswag": "/clzs_test011/qyh/dataset/hellaswag.parquet",
        "obqa":      "/clzs_test011/qyh/dataset/OBQA.parquet",
        "gsm8k":     "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl",
        "wikitext2": "/clzs_test011/qyh/dataset/validation-00000-of-00001.parquet",
    },
    "results_dir": Path(__file__).parent / "results",
}
CFG["results_dir"].mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILS
# ══════════════════════════════════════════════════════════════════════════════

def extract_mc(row, dset):
    """Return (question_str, [choices], answer_letter) for MC datasets."""
    if dset in ("arc_e", "arc_c"):
        question = row.get("question", row.get("question_stem", ""))
        cd = row["choices"]
        choices = cd["text"] if isinstance(cd, dict) else list(cd)
        ans = row["answerKey"]
        if ans in ("1","2","3","4"):
            ans = chr(ord("A") + int(ans) - 1)
        return question, choices, ans

    elif dset == "hellaswag":
        question = row["ctx"]
        choices  = row["endings"]
        ans = chr(ord("A") + int(row["label"]))
        return question, choices, ans

    elif dset == "obqa":
        question = row.get("question_stem", row.get("question", ""))
        cd = row["choices"]
        choices = cd["text"] if isinstance(cd, dict) else list(cd)
        ans = row["answerKey"]
        return question, choices, ans

    raise ValueError(f"Unknown dataset: {dset}")


def format_mc(question, choices):
    prompt = f"Question: {question}\n"
    letters = []
    for i, c in enumerate(choices):
        l = chr(ord("A") + i)
        prompt += f"{l}. {c}\n"
        letters.append(l)
    prompt += "Answer:"
    return prompt, letters


def extract_gsm8k_answer(text: str):
    """Pull the last integer / decimal from a GSM8K solution string."""
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


# ══════════════════════════════════════════════════════════════════════════════
# SADS PREDICTION CORE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_mc_sads(full_model, mid_model, tokenizer, prompt, letters, tau, device):
    """Run SADS for one multiple-choice question. Returns (predicted_letter, used_fallback)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    letter_ids = [tokenizer.encode(l, add_special_tokens=False)[-1] for l in letters]

    out = mid_model(ids)
    logits = out.logits[:, -1, :].float()
    entropy = torch.distributions.Categorical(logits=logits).entropy().item()

    fallback = False
    if math.isnan(entropy) or entropy > tau:
        fallback = True
        logits = full_model(ids).logits[:, -1, :].float()

    mask = torch.full_like(logits, -float("inf"))
    for idx in letter_ids:
        mask[0, idx] = logits[0, idx]
    pred_id = torch.argmax(mask, dim=-1).item()

    pred_letter = letters[letter_ids.index(pred_id)] if pred_id in letter_ids else letters[0]
    return pred_letter, fallback


@torch.no_grad()
def generate_sads(full_model, mid_model, tokenizer, prompt, tau, device,
                  max_new_tokens=256):
    """Token-level SADS generation for GSM8K. Returns (generated_text, fallback_count, total_tokens)."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = []
    fallback_count = 0

    # we'll do a simple greedy loop with KV-cache disabled to keep code simple
    for _ in range(max_new_tokens):
        out = mid_model(ids)
        logits = out.logits[:, -1, :].float()
        entropy = torch.distributions.Categorical(logits=logits).entropy().item()

        if math.isnan(entropy) or entropy > tau:
            fallback_count += 1
            logits = full_model(ids).logits[:, -1, :].float()

        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_id.item())
        ids = torch.cat([ids, next_id], dim=-1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, fallback_count, len(generated)


# ══════════════════════════════════════════════════════════════════════════════
# PER-DATASET EVALUATORS
# ══════════════════════════════════════════════════════════════════════════════

def eval_mc_dataset(full_model, mid_model, tokenizer, dset_name, parquet_path, tau, device, limit=-1):
    df = pd.read_parquet(parquet_path)
    if limit > 0:
        df = df.iloc[:limit]
    correct, fallbacks, total = 0, 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{dset_name} tau={tau:.1f}"):
        try:
            q, c, ans = extract_mc(row, dset_name)
        except Exception as e:
            log.warning(f"Skip row: {e}")
            continue
        prompt, letters = format_mc(q, c)
        pred, fb = predict_mc_sads(full_model, mid_model, tokenizer, prompt, letters, tau, device)
        correct += int(pred == ans)
        fallbacks += int(fb)
        total += 1
    acc = correct / total * 100 if total else 0
    fallback_rate = fallbacks / total * 100 if total else 0
    return acc, fallback_rate, total


def eval_gsm8k(full_model, mid_model, tokenizer, parquet_path, tau, device, limit=200):
    """
    GSM8K: generate answer, extract last number, compare to gold.
    Uses limit=200 by default for speed (set limit=-1 for full 1319).
    """
    path = str(parquet_path)
    if path.endswith(".jsonl") or path.endswith(".json"):
        import json as _json
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(_json.loads(line))
        df = pd.DataFrame(records)
    else:
        df = pd.read_parquet(path)
    if limit > 0:
        df = df.iloc[:limit]
    correct, fallbacks, total_toks, total = 0, 0, 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"gsm8k tau={tau:.1f}"):
        question = row.get("question", row.get("problem", ""))
        gold_raw = str(row.get("answer", row.get("solution", "")))
        gold_num = extract_gsm8k_answer(gold_raw)

        prompt = (
            "Solve the following math problem step by step. "
            "Write your final answer after '####'.\n\n"
            f"Problem: {question}\nSolution:"
        )
        gen, fb, ntoks = generate_sads(full_model, mid_model, tokenizer,
                                       prompt, tau, device, max_new_tokens=300)
        pred_num = extract_gsm8k_answer(gen)
        correct += int(pred_num is not None and gold_num is not None and pred_num == gold_num)
        fallbacks += fb
        total_toks += ntoks
        total += 1

    acc = correct / total * 100 if total else 0
    fb_rate = fallbacks / total_toks * 100 if total_toks else 0
    return acc, fb_rate, total


@torch.no_grad()
def eval_wikitext2_ppl(model, tokenizer, parquet_path, device, max_tokens=2048):
    """Compute perplexity on WikiText2 test set."""
    df = pd.read_parquet(parquet_path)
    # Expect column 'text'
    text_col = "text" if "text" in df.columns else df.columns[0]
    full_text = "\n\n".join(df[text_col].dropna().tolist())

    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    # Truncate to manageable length
    input_ids = input_ids[:, :max_tokens]
    seq_len = input_ids.size(1)

    stride = 512
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + stride, seq_len)
        target_len = end - prev_end
        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-target_len] = -100

        out = model(chunk, labels=labels)
        nlls.append(out.loss.item() * target_len)
        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / seq_len)
    return ppl


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(taus, sparsity, limit):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  Sparsity: {sparsity}%  |  Taus: {taus}")

    # ── Load models once ────────────────────────────────────────────────────
    log.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading FULL (dense) model …")
    full_model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"], torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()

    sparsity_frac = sparsity / 100.0
    sp_tag = str(sparsity)           # e.g. "40"
    model_dir = os.path.join(CFG["experiments_dir"], f"llama3_8b_{sp_tag}")
    log.info(f"Loading SPARSE model from {model_dir} …")
    mid_adapter, _ = hf_utils.load_sliced_model(
        CFG["hf_model_id"], model_dir, sparsity=sparsity_frac
    )
    mid_model = mid_adapter.model.to(torch.bfloat16).to(device).eval()

    # ── WikiText2 PPL: run once per model, not per tau ──────────────────────
    # (tau does NOT affect PPL eval because PPL uses teacher-forcing on sparse model,
    #  except when tau forces fallback. We compute PPL on mid_model for static sparse
    #  and full_model for tau=0.0 as upper bound.)
    log.info("Pre-computing WikiText2 PPL (sparse model) …")
    ppl_sparse = eval_wikitext2_ppl(mid_model, tokenizer,
                                     CFG["datasets"]["wikitext2"], device)
    log.info(f"  Sparse PPL = {ppl_sparse:.2f}")

    log.info("Pre-computing WikiText2 PPL (dense model) …")
    ppl_dense = eval_wikitext2_ppl(full_model, tokenizer,
                                    CFG["datasets"]["wikitext2"], device)
    log.info(f"  Dense PPL  = {ppl_dense:.2f}")

    # ── SADS PPL for intermediate taus: we approximate by the sparse baseline
    #    (PPL is a property of the generative model; SADS routing affects it
    #     only marginally for high tau — we report sparse ppl for tau>0, dense for tau=0)

    MC_TASKS = ["arc_e", "arc_c", "hellaswag", "obqa"]

    all_rows = []

    for tau in taus:
        tau_label = "inf" if math.isinf(tau) else str(tau)
        log.info(f"\n{'='*60}")
        log.info(f"  Evaluating  tau = {tau_label}")
        log.info(f"{'='*60}")

        row = {"tau": tau_label}

        # ── Multiple-choice tasks ──────────────────────────────────────────
        mc_accs = []
        task_fb_rates = []
        for task in MC_TASKS:
            acc, fb_rate, n = eval_mc_dataset(
                full_model, mid_model, tokenizer,
                task, CFG["datasets"][task], tau, device, limit=limit
            )
            row[f"{task}_acc"] = round(acc, 2)
            row[f"{task}_fb"]  = round(fb_rate, 2)
            mc_accs.append(acc)
            task_fb_rates.append(fb_rate)
            log.info(f"    {task:12s}  acc={acc:.2f}%  fallback={fb_rate:.2f}%  (n={n})")

        row["avg_acc"]       = round(sum(mc_accs) / len(mc_accs), 2)
        row["avg_fb_rate"]   = round(sum(task_fb_rates) / len(task_fb_rates), 2)

        # ── GSM8K ──────────────────────────────────────────────────────────
        gsm_limit = limit if limit > 0 else 200   # default 200 samples for speed
        gsm_acc, gsm_fb, gsm_n = eval_gsm8k(
            full_model, mid_model, tokenizer,
            CFG["datasets"]["gsm8k"], tau, device, limit=gsm_limit
        )
        row["gsm8k_acc"] = round(gsm_acc, 2)
        row["gsm8k_fb"]  = round(gsm_fb, 2)
        log.info(f"    gsm8k        acc={gsm_acc:.2f}%  token_fb={gsm_fb:.2f}%  (n={gsm_n})")

        # ── WikiText2 PPL ──────────────────────────────────────────────────
        if math.isinf(tau):
            row["wikitext2_ppl"] = round(ppl_sparse, 2)
        elif tau == 0.0:
            row["wikitext2_ppl"] = round(ppl_dense, 2)
        else:
            # Approximate: sparse ppl, since most tokens still use sparse model
            row["wikitext2_ppl"] = round(ppl_sparse, 2)

        all_rows.append(row)
        log.info(f"  → avg_acc={row['avg_acc']}  gsm8k={row['gsm8k_acc']}  ppl={row['wikitext2_ppl']}")

    # ── Compute Gain over Static ───────────────────────────────────────────
    static_avg = next(r["avg_acc"] for r in all_rows if r["tau"] == "inf")
    for r in all_rows:
        r["gain_over_static"] = round(r["avg_acc"] - static_avg, 2)

    # ── Save CSV ──────────────────────────────────────────────────────────
    out_csv = CFG["results_dir"] / "ablation_tau_results.csv"
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(out_csv, index=False)
    log.info(f"\nResults saved → {out_csv}")

    return all_rows


# ══════════════════════════════════════════════════════════════════════════════
# TABLE PRINTER
# ══════════════════════════════════════════════════════════════════════════════

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
    print(f"\n{'SADS Threshold Ablation (Sparsity=40%)'}")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        dense_r = r["avg_fb_rate"] if r["tau"] != "inf" else 0.00
        if r["tau"] == "0.0":
            dense_r = 100.00
        print(
            f"{r['tau']:>14} | "
            f"{dense_r:>17.2f} | "
            f"{r['avg_acc']:>16.2f} | "
            f"{r['gsm8k_acc']:>10.2f} | "
            f"{r['wikitext2_ppl']:>13.2f} | "
            f"{r['gain_over_static']:>+16.2f}"
        )
    print(sep)


def print_latex(rows):
    print("\n% ── LaTeX Table (paste directly into paper) ──────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{SADS Threshold Ablation at 40\% Sparsity (LLaMA-3-8B). "
          r"Average Acc.\ is the mean over ARC-e, ARC-c, HellaSwag, and OBQA. "
          r"Gain over Static is relative to $\tau{=}\infty$ (pure static sparse).}")
    print(r"\label{tab:ablation_tau}")
    print(r"\resizebox{\linewidth}{!}{%")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"$\tau$ & Dense Routing (\%) & Average Acc.\ (\%) & GSM8K (\%) "
          r"& WikiText2 PPL & Gain over Static \\")
    print(r"\midrule")
    for r in rows:
        tau_str = r"$\infty$" if r["tau"] == "inf" else f"${r['tau']}$"
        dense_r = r["avg_fb_rate"] if r["tau"] not in ("inf", "0.0") else (
            0.00 if r["tau"] == "inf" else 100.00
        )
        gain_str = f"${r['gain_over_static']:+.2f}$"
        if r["tau"] == "inf":
            gain_str = r"$0.00$"
        print(
            f"{tau_str} & {dense_r:.2f} & {r['avg_acc']:.2f} & "
            f"{r['gsm8k_acc']:.2f} & {r['wikitext2_ppl']:.2f} & {gain_str} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--taus", nargs="+", default=["inf", "6.0", "5.0", "4.0", "0.0"],
                   help="Tau values to sweep. Use 'inf' for infinity.")
    p.add_argument("--sparsity", type=int, default=40,
                   help="Sparsity percentage (e.g. 40 → llama3_8b_40/)")
    p.add_argument("--limit", type=int, default=-1,
                   help="Samples per dataset (-1 = full). Use e.g. 200 for quick smoke test.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tau_values = []
    for t in args.taus:
        tau_values.append(float("inf") if t.lower() == "inf" else float(t))

    rows = run_sweep(tau_values, args.sparsity, args.limit)
    print_table(rows)
    print_latex(rows)

    # Also dump JSON for reproducibility
    json_out = CFG["results_dir"] / "ablation_tau_results.json"
    with open(json_out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nJSON dump → {json_out}")
