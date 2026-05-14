"""
SADS 熵阈值标定脚本
===================
逻辑：
  1. 用 Dense 模型跑 calibration set → 得到健康基线熵分布 → 确定 τ₉₅
  2. 用各 Sparse 模型跑同样数据 → 统计超标率
  3. 结果直接支撑论文 Section 3.2 的 claim

用法：
  python calibrate_entropy.py
"""

import os
import sys
import math
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # 服务器无显示器，用非交互后端
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

# ── 加载 slicegpt 工具 ──────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils

# ── 离线模式，绝不连外网 ────────────────────────────────────────────────────
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


# ===========================================================================
# 工具函数
# ===========================================================================

def load_calibration_prompts(gsm8k_path, wiki_path, n_math=25, n_wiki=25, seed=42):
    """
    从 GSM8K 和 WikiText 各取一部分，混合成 calibration set。
    涵盖"难"和"易"两类文本，让熵分布更有代表性。
    """
    random.seed(seed)
    prompts = []

    # ── GSM8K（数学，偏难） ──
    if os.path.exists(gsm8k_path):
        with open(gsm8k_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f if l.strip()]
        math_prompts = [d['question'] for d in lines if 'question' in d]
        sampled = random.sample(math_prompts, min(n_math, len(math_prompts)))
        prompts.extend(sampled)
        print(f"  ✅ GSM8K 载入 {len(sampled)} 条数学题")
    else:
        print(f"  ⚠️  找不到 GSM8K: {gsm8k_path}，跳过")

    def load_wikitext_from_dir(wiki_dir, n_samples=25, seed=42):
        random.seed(seed)
        texts = []

        if not os.path.exists(wiki_dir):
            print(f"  ⚠️  找不到目录: {wiki_dir}")
            return texts

        files = os.listdir(wiki_dir)
        print(f"  📁 WikiText 目录内容: {files}")

        for fname in files:
            fpath = os.path.join(wiki_dir, fname)

            # parquet 格式
            if fname.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(fpath)
                col = "text" if "text" in df.columns else df.columns[0]
                texts += [str(t).strip() for t in df[col].dropna() if len(str(t).strip()) > 80]

            # jsonl 格式
            elif fname.endswith(".jsonl") or fname.endswith(".json"):
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            t = d.get("text", d.get("prompt", ""))
                            if len(t) > 80:
                                texts.append(t.strip())
                        except:
                            pass

            # 纯文本格式
            elif fname.endswith(".txt") or fname.endswith(".test") or "wiki" in fname.lower():
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) > 80:
                            texts.append(line)

            if len(texts) >= n_samples * 10:
                break

        sampled = random.sample(texts, min(n_samples, len(texts)))
        print(f"  ✅ WikiText 载入 {len(sampled)} 条")
        return sampled
    return prompts


def collect_entropy(model, tokenizer, prompts, device, max_new_tokens=80, desc="采集熵值"):
    """
    对每条 prompt 用模型自回归生成，记录每个生成 token 的香农熵。
    使用 use_cache=True + 每步只传新 token，效率最高。
    """
    model.eval()
    all_entropies = []
    eos_ids = {tokenizer.eos_token_id}
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None:
        eos_ids.add(eot_id)

    for prompt in tqdm(prompts, desc=desc):
        # 格式化为 chat template
        msg = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = prompt

        input_ids = tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(device)

        past_key_values = None
        current_ids = input_ids   # 第一步传完整 prompt

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    current_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()   # (1, vocab)

            # 香农熵：H = -Σ p·log(p)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()

            if not math.isnan(entropy) and not math.isinf(entropy):
                all_entropies.append(entropy)

            # 贪婪解码，只传新 token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)   # (1,1)
            current_ids = next_token

            if next_token.item() in eos_ids:
                break

    return np.array(all_entropies)


def print_stats(name, entropies, tau=None):
    print(f"\n{'='*55}")
    print(f"  📊 {name}")
    print(f"{'='*55}")
    print(f"  样本 Token 数  : {len(entropies)}")
    print(f"  均值           : {np.mean(entropies):.4f}")
    print(f"  中位数         : {np.median(entropies):.4f}")
    print(f"  标准差         : {np.std(entropies):.4f}")
    print(f"  90th pct       : {np.percentile(entropies, 90):.4f}")
    print(f"  95th pct (τ₉₅) : {np.percentile(entropies, 95):.4f}  ← 推荐阈值")
    print(f"  99th pct       : {np.percentile(entropies, 99):.4f}")
    if tau is not None:
        exceed = (entropies > tau).mean() * 100
        print(f"\n  ⚡ 相对于 Dense τ₉₅={tau:.4f}，本模型超标率: {exceed:.2f}%")
        if exceed > 5:
            print(f"  ✅ 超标率 > 5%，说明稀疏模型确实在部分 token 上力不从心")
            print(f"     → 熵信号有效，支撑论文 Section 3.2")
        else:
            print(f"  ℹ️  超标率 ≤ 5%，该稀疏度对此数据集影响较小")
    print(f"{'='*55}")


def plot_distributions(results, tau_dense, save_path="entropy_distributions.png"):
    """
    画熵分布对比图，可以直接放进论文。
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for ax, (name, entropies), color in zip(axes, results.items(), colors):
        ax.hist(entropies, bins=60, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(tau_dense, color='red', linestyle='--', linewidth=1.5, label=f'Dense τ₉₅={tau_dense:.2f}')
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Shannon Entropy", fontsize=9)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Token Count", fontsize=9)
    plt.suptitle("Entropy Distributions: Dense vs Sparse Models", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  📈 分布图已保存: {save_path}")


# ===========================================================================
# 主程序
# ===========================================================================

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}\n")

    # ── 路径配置（按你的服务器实际路径修改） ──────────────────────────────
    BASE_MODEL_PATH  = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    GSM8K_PATH       = "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl"
    WIKI_PATH = "/clzs_test011/qyh/dataset/wikitext"

    DIR = os.path.dirname(os.path.abspath(__file__))
    TIER_CONFIGS = [
        {"name": "Sparse 10%", "path": os.path.join(DIR, "llama3_8b_10"), "sparsity": 0.10},
        {"name": "Sparse 15%", "path": os.path.join(DIR, "llama3_8b_15"), "sparsity": 0.15},
        {"name": "Sparse 30%", "path": os.path.join(DIR, "llama3_8b_30"), "sparsity": 0.30},
    ]
    # ───────────────────────────────────────────────────────────────────────

    # ── Step 0: 准备校准集 ──────────────────────────────────────────────────
    print("=" * 55)
    print("  Step 0 / 加载校准集")
    print("=" * 55)
    calibration_prompts = load_calibration_prompts(
        gsm8k_path=GSM8K_PATH,
        wiki_path=WIKI_PATH,
        n_math=25,
        n_wiki=25,
    )

    if len(calibration_prompts) == 0:
        print("❌ 校准集为空，请检查数据路径！")
        sys.exit(1)

    # ── Step 1: Dense 模型标定 τ₉₅ ─────────────────────────────────────────
    print("=" * 55)
    print("  Step 1 / Dense 模型采集基线熵分布")
    print("=" * 55)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dense_adapter, _ = hf_utils.get_model_and_tokenizer(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        model_path=BASE_MODEL_PATH,
        dtype=torch.bfloat16
    )
    dense_model = dense_adapter.model.to(device).eval()

    dense_entropies = collect_entropy(
        dense_model, tokenizer, calibration_prompts, device, desc="Dense 模型"
    )

    tau_95 = float(np.percentile(dense_entropies, 95))
    print_stats("Dense Model (0% sparsity)", dense_entropies)
    print(f"\n  🎯 论文使用的 τ₉₅ = {tau_95:.4f}  （写进 Section 3.2）\n")

    # 保存结果，后续不用重跑
    np.save("entropy_dense.npy", dense_entropies)

    del dense_adapter, dense_model
    torch.cuda.empty_cache()

    # ── Step 2: 各稀疏模型超标率统计 ───────────────────────────────────────
    all_results = {"Dense (0%)": dense_entropies}
    tau_summary = []

    for cfg in TIER_CONFIGS:
        print(f"\n{'='*55}")
        print(f"  Step 2 / 加载 {cfg['name']} 稀疏模型")
        print(f"{'='*55}")

        if not os.path.exists(cfg["path"]):
            print(f"  ⚠️  路径不存在: {cfg['path']}，跳过")
            continue

        adapter, _ = hf_utils.load_sliced_model(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            cfg["path"],
            sparsity=cfg["sparsity"]
        )
        sparse_model = adapter.model.to(torch.bfloat16).to(device).eval()

        sparse_entropies = collect_entropy(
            sparse_model, tokenizer, calibration_prompts, device,
            desc=cfg["name"]
        )

        np.save(f"entropy_{cfg['name'].replace(' ', '_').replace('%', 'pct')}.npy", sparse_entropies)
        print_stats(cfg["name"], sparse_entropies, tau=tau_95)

        exceed_rate = float((sparse_entropies > tau_95).mean() * 100)
        tau_summary.append({
            "model": cfg["name"],
            "mean_entropy": float(np.mean(sparse_entropies)),
            "tau_95_self": float(np.percentile(sparse_entropies, 95)),
            "exceed_rate_vs_dense": exceed_rate
        })

        all_results[cfg["name"]] = sparse_entropies

        del adapter, sparse_model
        torch.cuda.empty_cache()

    # ── Step 3: 汇总报告 ───────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  📋 最终汇总报告（可直接写进论文 Section 3.2 / Table）")
    print(f"{'='*65}")
    print(f"  Dense 模型 τ₉₅ (论文阈值) = {tau_95:.4f}\n")
    print(f"  {'模型':<18} {'平均熵':>10} {'自身τ₉₅':>10} {'超标率(%)':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Dense (0%)':<18} {np.mean(dense_entropies):>10.4f} {tau_95:>10.4f} {'—':>12}")
    for row in tau_summary:
        print(f"  {row['model']:<18} {row['mean_entropy']:>10.4f} "
              f"{row['tau_95_self']:>10.4f} {row['exceed_rate_vs_dense']:>11.2f}%")
    print(f"{'='*65}")

    # ── Step 4: 画分布对比图 ───────────────────────────────────────────────
    plot_distributions(all_results, tau_95, save_path="entropy_distributions.png")

    # ── Step 5: 保存完整结果 JSON ──────────────────────────────────────────
    import json as jsonlib
    result_json = {
        "tau_95_dense": tau_95,
        "dense_mean": float(np.mean(dense_entropies)),
        "sparse_models": tau_summary
    }
    with open("calibration_results.json", "w") as f:
        jsonlib.dump(result_json, f, indent=2, ensure_ascii=False)
    print("\n  💾 完整结果已保存至 calibration_results.json")
    print("\n✅ 标定完成！")