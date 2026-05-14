import argparse
import gc
import json
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Use the project src path directly to avoid import issues in detached runs.
SRC_PATH = "/clzs_test011/qyh/slice_new/TransformerCompression-main/TransformerCompression-main/src"
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from slicegpt import hf_utils  # noqa: E402

CFG = {
    "base_model": "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct",
    "experiments_dir": "/clzs_test011/qyh/slice_new/TransformerCompression-main/TransformerCompression-main/experiments",
    "dataset_path": "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl",
    "results_file": "gsm8k_main_results_persistent.jsonl",
    "tau_sads": 4.0,
    "limit": 200,  # set to -1 for full evaluation
    "max_new_tokens": 300,
    "sparsities": [20, 40, 60],
}


def extract_gsm8k_answer(text: str):
    """Prefer the official #### answer marker, then fall back to the last number."""
    text = text.replace(",", "")
    marker = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if marker:
        return marker.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def build_prompt(tokenizer, question: str):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful math reasoning assistant. "
                "Solve the problem step by step, and put the final answer after '####'."
            ),
        },
        {"role": "user", "content": f"Problem: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_sads_persistent(
        full_model,
        mid_model,
        tokenizer,
        question,
        tau,
        device,
        max_new_tokens,
):
    """
    GSM8K generation with persistent fallback.

    Modes:
    - tau < 0: always dense
    - tau == inf: always sparse
    - otherwise: start sparse, switch to dense permanently once entropy > tau

    This is closer to the paper's "move to denser tier and stay there" behavior
    than the prior one-step dense correction script.
    """
    prompt = build_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated_ids = []
    sparse_cache = None
    dense_cache = None
    current_input = prompt_ids

    active_mode = "dense" if tau < 0 else "sparse"
    switched_once = False
    dense_token_count = 0

    eos_token_ids = {tokenizer.eos_token_id}
    for special_id in (128001, 128009):
        eos_token_ids.add(special_id)

    for _ in range(max_new_tokens):
        if active_mode == "dense":
            outputs = full_model(current_input, past_key_values=dense_cache, use_cache=True)
            logits = outputs.logits[:, -1, :].float()
            dense_cache = outputs.past_key_values
        else:
            outputs = mid_model(current_input, past_key_values=sparse_cache, use_cache=True)
            logits = outputs.logits[:, -1, :].float()
            sparse_cache = outputs.past_key_values

            entropy = torch.distributions.Categorical(logits=logits).entropy().item()
            if math.isnan(entropy) or entropy > tau:
                switched_once = True
                active_mode = "dense"

                # Rebuild dense state from the full context once, then continue densely.
                if generated_ids:
                    gen_tensor = torch.tensor([generated_ids], dtype=prompt_ids.dtype, device=device)
                    full_context = torch.cat([prompt_ids, gen_tensor], dim=1)
                else:
                    full_context = prompt_ids

                dense_outputs = full_model(full_context, use_cache=True)
                logits = dense_outputs.logits[:, -1, :].float()
                dense_cache = dense_outputs.past_key_values
                sparse_cache = None

        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = next_id.item()
        generated_ids.append(token_id)
        if active_mode == "dense":
            dense_token_count += 1

        current_input = next_id
        if token_id in eos_token_ids:
            break

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    total_generated = len(generated_ids)
    dense_token_rate = (dense_token_count / total_generated * 100.0) if total_generated else 0.0
    return text, switched_once, dense_token_rate, total_generated


def load_gsm8k(path: str, limit: int):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    if limit > 0:
        df = df.iloc[:limit]
    return df


def evaluate_gsm8k(full_model, mid_model, tokenizer, method_name, tau, device, limit, max_new_tokens):
    print(f"\n[Evaluating] {method_name} | tau={tau}")
    df = load_gsm8k(CFG["dataset_path"], limit)

    correct = 0
    switched_questions = 0
    dense_token_rates = []
    total = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=method_name):
        question = row.get("question", row.get("problem", ""))
        gold_raw = str(row.get("answer", row.get("solution", "")))
        gold_num = extract_gsm8k_answer(gold_raw)

        pred_text, switched_once, dense_token_rate, _ = generate_sads_persistent(
            full_model=full_model,
            mid_model=mid_model,
            tokenizer=tokenizer,
            question=question,
            tau=tau,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        pred_num = extract_gsm8k_answer(pred_text)

        if pred_num is not None and gold_num is not None and pred_num == gold_num:
            correct += 1
        switched_questions += int(switched_once)
        dense_token_rates.append(dense_token_rate)
        total += 1

    acc = (correct / total * 100.0) if total else 0.0
    question_level_switch_rate = (switched_questions / total * 100.0) if total else 0.0
    avg_dense_token_rate = (sum(dense_token_rates) / len(dense_token_rates)) if dense_token_rates else 0.0

    print(
        f"  -> Acc: {acc:.2f}% | "
        f"Question Switch Rate: {question_level_switch_rate:.2f}% | "
        f"Avg Dense Token Rate: {avg_dense_token_rate:.2f}%"
    )
    return acc, question_level_switch_rate, avg_dense_token_rate


def run_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], local_files_only=True)

    print("[Loading dense model]")
    full_model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"],
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()

    results = []

    dense_acc, dense_switch_rate, dense_token_rate = evaluate_gsm8k(
        full_model=full_model,
        mid_model=None,
        tokenizer=tokenizer,
        method_name="Dense",
        tau=-1.0,
        device=device,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    results.append(
        {
            "method": "Dense",
            "sparsity": 0,
            "tau": -1.0,
            "gsm8k_acc": round(dense_acc, 2),
            "question_switch_rate": round(dense_switch_rate, 2),
            "avg_dense_token_rate": round(dense_token_rate, 2),
        }
    )

    for sp in args.sparsities:
        print(f"\n{'=' * 60}\nEvaluating sparsity = {sp}%\n{'=' * 60}")
        model_dir = os.path.join(CFG["experiments_dir"], f"llama3_8b_{sp}")
        if not os.path.exists(model_dir):
            print(f"[Skip] Missing sparse model directory: {model_dir}")
            continue

        print(f"[Loading sparse model] {model_dir}")
        mid_adapter, _ = hf_utils.load_sliced_model(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            model_dir,
            sparsity=sp / 100.0,
        )
        mid_model = mid_adapter.model.to(torch.bfloat16).to(device).eval()

        static_acc, static_switch_rate, static_token_rate = evaluate_gsm8k(
            full_model=full_model,
            mid_model=mid_model,
            tokenizer=tokenizer,
            method_name=f"Static Sparse {sp}%",
            tau=float("inf"),
            device=device,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(
            {
                "method": "Static Sparse",
                "sparsity": sp,
                "tau": "inf",
                "gsm8k_acc": round(static_acc, 2),
                "question_switch_rate": round(static_switch_rate, 2),
                "avg_dense_token_rate": round(static_token_rate, 2),
            }
        )

        sads_acc, sads_switch_rate, sads_token_rate = evaluate_gsm8k(
            full_model=full_model,
            mid_model=mid_model,
            tokenizer=tokenizer,
            method_name=f"SADS {sp}%",
            tau=args.tau_sads,
            device=device,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(
            {
                "method": "SADS",
                "sparsity": sp,
                "tau": args.tau_sads,
                "gsm8k_acc": round(sads_acc, 2),
                "question_switch_rate": round(sads_switch_rate, 2),
                "avg_dense_token_rate": round(sads_token_rate, 2),
            }
        )

        del mid_model, mid_adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = Path(args.results_file)
    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[Done] Results saved to: {out_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-sads", type=float, default=CFG["tau_sads"])
    parser.add_argument("--limit", type=int, default=CFG["limit"])
    parser.add_argument("--max-new-tokens", type=int, default=CFG["max_new_tokens"])
    parser.add_argument("--results-file", type=str, default=CFG["results_file"])
    parser.add_argument("--sparsities", nargs="+", type=int, default=CFG["sparsities"])
    return parser.parse_args()


if __name__ == "__main__":
    run_main(parse_args())
