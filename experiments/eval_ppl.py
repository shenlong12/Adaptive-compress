import os
import torch
import math
import gc
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🛡️ 强制开启完全离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils


# ==========================================
# 1. Parquet 数据加载器
# ==========================================
def load_text_data(file_path, max_chars=10000):
    text = ""
    if not os.path.exists(file_path):
        print(f"🔴 找不到数据: {file_path}")
        return text
    print(f"📖 正在通过 Parquet 引擎加载数据: {file_path} ...")
    try:
        df = pd.read_parquet(file_path)
        text_list = df['text'].dropna().astype(str).tolist()
        text = "\n\n".join(text_list)
        return text[:max_chars]
    except Exception as e:
        print(f"🔴 数据加载彻底失败: {e}")
        return ""


# ==========================================
# 2. 纯静态模型 PPL 评测 (当靶子用)
# ==========================================
def calculate_static_perplexity(model, tokenizer, text, max_tokens=2048, device="cuda"):
    model.eval()
    encodings = tokenizer(text, return_tensors="pt").input_ids[:, :max_tokens].to(device)
    with torch.no_grad():
        # 静态模型可以直接用极速的 Teacher Forcing 一次性算出 PPL
        outputs = model(encodings, labels=encodings)
        ppl = torch.exp(outputs.loss).item()
    return ppl


# ==========================================
# 3. SADS 动态一镜到底 PPL 评测 (纯净重算版)
# ==========================================
def calculate_sads_perplexity_continuous(full_model, mid_model, tokenizer, text, tau_threshold=2.0, max_tokens=2048,
                                         device="cuda"):
    mid_model.eval()
    full_model.eval()

    encodings = tokenizer(text, return_tensors="pt").input_ids[:, :max_tokens].to(device)
    seq_len = encodings.size(1)

    nlls = []
    fallback_count = 0
    warmup_len = 128

    # === 将之前的 for 循环和前面的预热状态全部替换为以下逻辑 ===

    # 1. 预热期：小模型正常预热
    prompt_ids = encodings[:, :warmup_len]
    with torch.no_grad():
        outputs = mid_model(prompt_ids, use_cache=True)
        past_key_values_mid = outputs.past_key_values

    current_input_ids = encodings[:, warmup_len - 1].unsqueeze(1)

    # 2. 连续评测期（真正的动态路由！）
    for i in tqdm(range(warmup_len - 1, seq_len - 1), desc="SADS 动态评测中", leave=False):
        target_id = encodings[:, i + 1]

        with torch.no_grad():
            # 🌟 第一步：小模型永远在默默前行，维持自己的 Cache
            outputs_mid = mid_model(current_input_ids, past_key_values=past_key_values_mid, use_cache=True)
            logits_mid = outputs_mid.logits[:, -1, :].float()
            past_key_values_mid = outputs_mid.past_key_values

            # 🌟 第二步：雷达扫描！
            entropy = torch.distributions.Categorical(logits=logits_mid).entropy().item()

            if math.isnan(entropy) or entropy > tau_threshold:
                # 🚨 触发警报！大模型空降，只为“当前这一个 Token”算出最完美的概率
                fallback_count += 1
                current_context_ids = encodings[:, :i + 1]
                outputs_full = full_model(current_context_ids, use_cache=True)
                logits = outputs_full.logits[:, -1, :].float()
            else:
                # ✅ 警报解除，采用小模型的概率
                logits = logits_mid

            # 🌟 第三步：计算这个 Token 的损失
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            true_log_prob = log_probs[0, target_id.item()]
            nlls.append(-true_log_prob.item())

            # 🌟 第四步：步进 (Teacher Forcing, 小模型的大脑吃入正确的下一个词)
            current_input_ids = target_id.unsqueeze(1)

    total_tokens = seq_len - warmup_len
    ppl = math.exp(sum(nlls) / total_tokens) if total_tokens > 0 else float('inf')
    return ppl, fallback_count


# ==========================================
# 4. 主控循环引擎
# ==========================================
def run_all_sparsities():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    wiki_path = "/clzs_test011/qyh/dataset/wikitext"

    wiki_text = load_text_data(wiki_path, max_chars=10000)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

    print("\n[加载 0% 满血模型作为后盾 (仅加载一次) ...]")
    full_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)

    # 先测一下原生的 PPL 当作绝对标杆
    print("\n测算 0% 满血原生 PPL...")
    dense_ppl = calculate_static_perplexity(full_model, tokenizer, wiki_text, device=device)
    print(f"👑 满血基线 PPL: {dense_ppl:.2f}")

    # 🌟 在这里配置你想测试的所有稀疏度
    # 格式: (文件夹后缀, 稀疏度比例)

    test_configs = [
        ("10", 0.10),
        ("15", 0.15),
        ("30", 0.30)  # 👈 换成你拥有的 30% 极限模型
    ]

    results = []

    print("\n" + "=" * 60)
    print(" 🚀 开启多稀疏度横向测评矩阵")
    print("=" * 60)

    for suffix, sparsity in test_configs:
        model_dir = os.path.join(DIR_PATH, f"llama3_8b_{suffix}")
        if not os.path.exists(model_dir):
            print(f"\n⚠️ 跳过 {sparsity * 100}%: 找不到模型文件夹 {model_dir}")
            continue

        print(f"\n[{sparsity * 100}% 稀疏度测试回合]")

        # 加载对应的静态裁剪模型
        mid_adapter, _ = hf_utils.load_sliced_model("meta-llama/Meta-Llama-3-8B-Instruct", model_dir, sparsity=sparsity)
        mid_model = mid_adapter.model.to(torch.bfloat16).to(device)

        # 1. 跑纯静态 PPL
        static_ppl = calculate_static_perplexity(mid_model, tokenizer, wiki_text, device=device)
        print(f"   ├─ 静态 SliceGPT PPL: {static_ppl:.2f}")

        # 2. 跑 SADS 动态 PPL
        sads_ppl, fallbacks = calculate_sads_perplexity_continuous(
            full_model, mid_model, tokenizer, wiki_text, tau_threshold=2.0, device=device
        )
        print(f"   ├─ SADS 动态 PPL: {sads_ppl:.2f} (救场 {fallbacks} 次)")

        # 记录数据
        results.append({
            "Sparsity": f"{int(sparsity * 100)}%",
            "Static PPL": round(static_ppl, 2),
            "SADS PPL": round(sads_ppl, 2),
            "Fallbacks": fallbacks
        })

        # 🧹 极其重要的内存管理：删掉小模型，清空显存，防止下一个循环 OOM
        del mid_adapter
        del mid_model
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================
    # 5. 打印最终的论文表格
    # ==========================================
    print("\n\n" + "🏆 最终实验结果汇总 (Markdown 表格) 🏆".center(60))
    print("| 稀疏度 | Static PPL (不带路由) | SADS PPL (带路由) | 触发救场次数 |")
    print("|:---|:---|:---|:---|")
    print(f"| 0% (Dense) | {dense_ppl:.2f} | - | - |")
    for r in results:
        print(f"| {r['Sparsity']} | {r['Static PPL']} | **{r['SADS PPL']}** | {r['Fallbacks']} |")


if __name__ == "__main__":
    run_all_sparsities()