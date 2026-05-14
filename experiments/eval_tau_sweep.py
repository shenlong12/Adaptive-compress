import os
import torch
import math
import gc
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils


def load_text_data(file_path, max_chars=10000):
    text = ""
    try:
        df = pd.read_parquet(file_path)
        text_list = df['text'].dropna().astype(str).tolist()
        text = "\n\n".join(text_list)
        return text[:max_chars]
    except Exception:
        return ""


def sweep_tau_thresholds():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    wiki_path = "/clzs_test011/qyh/dataset/wikitext"

    wiki_text = load_text_data(wiki_path, max_chars=10000)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

    print("\n[加载模型...]")
    full_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)

    # 我们固定使用 15% 的小模型来做这次消融实验
    sparsity = 0.15
    mid_adapter, _ = hf_utils.load_sliced_model("meta-llama/Meta-Llama-3-8B-Instruct",
                                                os.path.join(DIR_PATH, "llama3_8b_15"), sparsity=sparsity)
    mid_model = mid_adapter.model.to(torch.bfloat16).to(device)

    full_model.eval()
    mid_model.eval()

    encodings = tokenizer(wiki_text, return_tensors="pt").input_ids[:, :2048].to(device)
    seq_len = encodings.size(1)
    warmup_len = 128

    # 预先跑一次小模型的预热，拿到基础状态
    prompt_ids = encodings[:, :warmup_len]
    with torch.no_grad():
        outputs_mid_base = mid_model(prompt_ids, use_cache=True)
        past_key_values_base = outputs_mid_base.past_key_values
    current_input_ids_base = encodings[:, warmup_len - 1].unsqueeze(1)

    # 我们要扫描的阈值
    tau_list = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
    results = []

    print("\n" + "=" * 65)
    print(" 🚀 开启 SADS 阈值敏感度与理论加速分析 (Tau Sweep)")
    print("=" * 65)

    for tau in tau_list:
        nlls = []
        fallback_count = 0

        # 恢复初始状态
        past_key_values_mid = past_key_values_base
        current_input_ids = current_input_ids_base

        for i in tqdm(range(warmup_len - 1, seq_len - 1), desc=f"评测 Tau={tau}", leave=False):
            target_id = encodings[:, i + 1]

            with torch.no_grad():
                outputs_mid = mid_model(current_input_ids, past_key_values=past_key_values_mid, use_cache=True)
                logits_mid = outputs_mid.logits[:, -1, :].float()
                past_key_values_mid = outputs_mid.past_key_values

                entropy = torch.distributions.Categorical(logits=logits_mid).entropy().item()

                if math.isnan(entropy) or entropy > tau:
                    fallback_count += 1
                    current_context_ids = encodings[:, :i + 1]
                    outputs_full = full_model(current_context_ids, use_cache=True)
                    logits = outputs_full.logits[:, -1, :].float()
                else:
                    logits = logits_mid

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                true_log_prob = log_probs[0, target_id.item()]
                nlls.append(-true_log_prob.item())

                current_input_ids = target_id.unsqueeze(1)

        total_tokens = seq_len - warmup_len
        ppl = math.exp(sum(nlls) / total_tokens) if total_tokens > 0 else float('inf')
        fallback_rate = fallback_count / total_tokens

        # 👑 核心理论计算：
        # 满血 FLOPs = 1.0
        # 小模型 FLOPs = (1.0 - sparsity) = 0.85
        # 实际消耗 = 小模型一直算(0.85) + 触发率 * 大模型额外算(1.0)
        flops_consumed = (1.0 - sparsity) + fallback_rate * 1.0
        theoretical_speedup = 1.0 / flops_consumed

        print(
            f"🔹 [Tau = {tau:4.1f}] PPL: {ppl:5.2f} | 救场: {fallback_count:>3} 次 ({fallback_rate * 100:5.1f}%) | 理论加速: {theoretical_speedup:4.2f}x")

        results.append({
            "Tau": tau,
            "PPL": round(ppl, 2),
            "Fallback_Rate": round(fallback_rate * 100, 1),
            "Speedup": round(theoretical_speedup, 2)
        })

    print("\n\n" + "🏆 Tau 敏感度分析结果 (用于绘制 Pareto 曲线) 🏆".center(65))
    print("| 阈值 (Tau) | SADS PPL | 触发率 (%) | 理论 FLOPs 加速比 |")
    print("|:---|:---|:---|:---|")
    for r in results:
        # 加速的用加粗显示
        speed_str = f"**{r['Speedup']}x**" if r['Speedup'] > 1.0 else f"{r['Speedup']}x"
        print(f"| {r['Tau']} | {r['PPL']} | {r['Fallback_Rate']}% | {speed_str} |")


if __name__ == "__main__":
    sweep_tau_thresholds()