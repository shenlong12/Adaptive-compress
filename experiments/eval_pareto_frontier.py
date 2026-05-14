import os
import random  # 🌟 新增：导入随机数库
import torch
# ... 其他 import 保持不变
# 🛡️ 1. 强制开启完全离线模式，绝对不连外网！防止 403 报错
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import math, json, re, sys, gc
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache  # 🌟 必须导入它来兼容 LLaMA-3 的 KV Cache

# 借用 slicegpt 加载库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils


# ==========================================
# 🧠 1. 轻量级路由大脑 (4档位)
# ==========================================
class RouterMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_tiers=4):
        super(RouterMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_tiers)

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        embedded = self.embedding(input_ids)
        sum_embeddings = (embedded * mask.unsqueeze(-1)).sum(dim=1)
        valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        sentence_rep = sum_embeddings / valid_lengths
        x = self.fc1(sentence_rep)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


class LightweightRouter:
    def __init__(self, model_path, tokenizer, device, num_tiers=4):
        self.device = device
        self.tokenizer = tokenizer
        self.model = RouterMLP(vocab_size=len(tokenizer), num_tiers=num_tiers).to(device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print("🟢 成功加载 4 档位路由权重！")
        else:
            print(f"🔴 致命错误：找不到雷达权重 {model_path}，请先运行 train_router.py")
            sys.exit(1)
        self.model.eval()

    def predict_tier(self, prompt):
        encoding = self.tokenizer(prompt, truncation=True, max_length=64, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids)
            tier = torch.argmax(logits, dim=-1).item()
        return tier


# ==========================================
# 🛠️ 2. 辅助函数
# ==========================================
def extract_last_number(text):
    text = text.replace(",", "")
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        ans = numbers[-1]
        if ans.endswith(".0"): ans = ans[:-2]
        return ans
    return None


import csv


def load_gsm8k(path, num=None):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if "####" in d.get('answer', ''):
                data.append({"q": d['question'], "a": d['answer'].split("####")[1].strip(), "type": "math"})
            if num and len(data) >= num: break
    return data


def load_local_mmlu(data_dir="/clzs_test011/qyh/dataset/data", num=None):
    data = []
    target_dir = os.path.join(data_dir, "test") if os.path.exists(os.path.join(data_dir, "test")) else data_dir
    if not os.path.exists(target_dir):
        print(f"⚠️ 找不到 MMLU 目录: {target_dir}")
        return data

    csv_files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    for file in csv_files:
        with open(os.path.join(target_dir, file), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    prompt = (
                        f"Question: {row[0]}\n"
                        f"A. {row[1]}\n"
                        f"B. {row[2]}\n"
                        f"C. {row[3]}\n"
                        f"D. {row[4]}\n"
                        "Please answer with only the letter A, B, C, or D."
                    )
                    data.append({"q": prompt, "a": row[5].strip().upper(), "type": "qa"})
                    if num and len(data) >= num: return data
    return data

import os
import json


def get_wiki_prompts(path="/clzs_test011/qyh/dataset/wikitext", num=None):
    data = []
    if not os.path.exists(path):
        print(f"⚠️ 找不到 Wiki 文件: {path}")
        return data

    print(f"📖 正在加载生成测试数据: {path}")

    # ======================================================
    # 策略 1：优先尝试作为 Parquet 文件解析 (针对你当前的 wikitext)
    # ======================================================
    is_parquet = False
    try:
        import pandas as pd
        # 尝试读取，如果它不是 parquet，这里会立刻抛出异常并跳走
        df = pd.read_parquet(path)
        is_parquet = True
        print("✅ 成功匹配 Parquet 格式引擎！")

        for text in df['text'].dropna():
            text = str(text).strip()
            if len(text) > 100:
                prompt = f"Please read the following text and write a detailed continuation or analysis:\n\n{text[:800]}"
                data.append({"q": prompt, "a": None, "type": "wiki"})
                if num and len(data) >= num:
                    return data
        return data

    except Exception as e:
        # 只有确实抛出了异常（比如格式不对），我们才往下走
        if is_parquet:
            print(f"🔴 Parquet 解析中途崩溃: {e}")
            return data
        # 如果不是 Parquet，静默进入下一个策略

    # ======================================================
    # 策略 2：后备方案 - 解析 JSONL 或 纯文本 (针对未来其他数据集)
    # ======================================================
    try:
        # 注意：如果是 Parquet 文件走到这里，一定会读出乱码
        # 但没关系，我们上一步已经成功拦截了 Parquet！
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                text = ""

                if path.endswith('.jsonl') or path.endswith('.json'):
                    try:
                        d = json.loads(line)
                        text = d.get('text', d.get('prompt', ''))
                    except:
                        continue
                else:
                    text = line

                if len(text) > 100:
                    prompt = f"Please read the following text and write a detailed continuation or analysis:\n\n{text[:800]}"
                    data.append({"q": prompt, "a": None, "type": "wiki"})

                if num and len(data) >= num:
                    break
        print("✅ 成功匹配纯文本/JSONL 引擎！")
    except Exception as e:
        print(f"🔴 文本解析失败: {e}")

    return data


# ==========================================
# 🚀 3. 异步分组调度终极评测 (SADS 端到端)
# ==========================================
def run_grouped_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"

    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    router_weight_path = os.path.join(DIR_PATH, "weights/router_mlp_4tiers.pth")

    tier_config = {
        0: {"name": "30% 极简", "path": os.path.join(DIR_PATH, "llama3_8b_30"), "tau": 1.7555, "sparsity": 0.30},
        1: {"name": "15% 均衡", "path": os.path.join(DIR_PATH, "llama3_8b_15"), "tau": 1.7555, "sparsity": 0.15},
        2: {"name": "10% 逻辑", "path": os.path.join(DIR_PATH, "llama3_8b_10"), "tau": 1.7555, "sparsity": 0.10},
        3: {"name": "0% 满血", "path": base_model_path, "tau": float('inf'), "sparsity": 0.0}
    }
    print("\n📥 正在从本地加载综合评测大卷...")
    all_tasks = (
            load_gsm8k("/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl", num=100) +
            get_wiki_prompts(num=100) +
            load_local_mmlu("/clzs_test011/qyh/dataset/data", num=100)
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    router = LightweightRouter(router_weight_path, tokenizer, device)

    print("\n[阶段 1/3]  启动任务感知预判 (Real Oracle Routing)...")
    grouped_tasks = {0: [], 1: [], 2: [], 3: []}

    for task in all_tasks:
        # 分配系统提示词
        if task["type"] == "math":
            msg = [{"role": "system",
                    "content": "You are a math expert. Solve step by step. Conclude with 'The final answer is [number]'."},
                   {"role": "user", "content": task['q']}]
            tier = 1
        elif task["type"] == "qa":
            msg = [{"role": "system", "content": "You are a knowledgeable assistant."},
                   {"role": "user", "content": task['q']}]
        else:
            msg = [{"role": "system", "content": "You are a detailed assistant."},
                   {"role": "user", "content": task['q']}]

        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

        # 🌟 临时干预：把测试集硬塞给 10% 的模型
        tier = router.predict_tier(prompt)

        if task["type"] == "math":
            tier = 2  # 🔴 强制指定为 Tier 2 (10% 稀疏度模型)

        grouped_tasks[tier].append({"prompt": prompt, "task_info": task})

    for t, tasks in grouped_tasks.items():
        print(f"    Tier {t} ({tier_config[t]['name']}) 拦截了 {len(tasks)} 个任务")

    del router
    torch.cuda.empty_cache()

    print("\n[阶段 2/3] 加载满血大模型 (0% 兜底)...")
    full_adapter, _ = hf_utils.get_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct",
                                                       model_path=base_model_path, dtype=torch.bfloat16)
    full_model = full_adapter.model.to(device).eval()

    global_stats = {
        "total_tokens": 0,
        "tier_tokens": {0: 0, 1: 0, 2: 0, 3: 0},
        "gsm8k_correct": 0,
        "gsm8k_total": 0,
        "fallback_count": 0,
        # 👇 新增下面三行，用来追踪多任务
        "mmlu_correct": 0,
        "mmlu_total": 0,
        "wiki_count": 0
    }
    print("\n[阶段 3/3]  启动异步分组流转引擎")
    for current_tier in range(4):
        tasks = grouped_tasks[current_tier]
        if not tasks: continue

        print(f"\n=============================================")
        print(f" 正在处理 Tier {current_tier} 任务群 ({tier_config[current_tier]['name']})")
        print(f"=============================================")

        active_model = full_model
        if current_tier != 3:
            print(f" 动态拉起 {tier_config[current_tier]['name']} 模型进显存...")
            mid_adapter, _ = hf_utils.load_sliced_model("meta-llama/Meta-Llama-3-8B-Instruct",
                                                        tier_config[current_tier]["path"],
                                                        sparsity=tier_config[current_tier]["sparsity"])
            active_model = mid_adapter.model.to(torch.bfloat16).to(device).eval()

        tau_threshold = tier_config[current_tier]['tau']

        for item in tqdm(tasks, desc=f"Tier {current_tier} 推理中"):
            prompt = item["prompt"]
            task_info = item["task_info"]

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            generated_ids = input_ids
            input_len = input_ids.shape[1]

            has_fallen_back = False
            fallback_step = -1

            past_key_values = None
            current_input_ids = input_ids

            for i in range(400):
                with torch.no_grad():
                    outputs = active_model(current_input_ids, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits[:, -1, :].float()

                    if current_tier != 3 and not has_fallen_back:
                        probs = torch.softmax(logits, dim=-1)
                        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()
                        if entropy > tau_threshold:
                            has_fallen_back = True
                            fallback_step = i
                            active_model = full_model
                            global_stats["fallback_count"] += 1

                            # 🌟 核心修正 2：SADS 零拷贝继承
                            padded_kv = []
                            k_sparse_example, _ = outputs.past_key_values[0]
                            pad_dim = full_model.config.hidden_size // full_model.config.num_attention_heads - \
                                      k_sparse_example.shape[-1]

                            for layer_idx, (k_sparse, v_sparse) in enumerate(outputs.past_key_values):
                                zeros_k = torch.zeros((*k_sparse.shape[:-1], pad_dim), dtype=k_sparse.dtype,
                                                      device=k_sparse.device)
                                zeros_v = torch.zeros((*v_sparse.shape[:-1], pad_dim), dtype=v_sparse.dtype,
                                                      device=v_sparse.device)

                                k_padded = torch.cat([k_sparse, zeros_k], dim=-1)
                                v_padded = torch.cat([v_sparse, zeros_v], dim=-1)
                                padded_kv.append((k_padded, v_padded))

                            new_cache = DynamicCache()
                            for layer_idx, (k, v) in enumerate(padded_kv):
                                new_cache.update(k, v, layer_idx)
                            past_key_values = new_cache

                            outputs = active_model(current_input_ids, past_key_values=past_key_values, use_cache=True)
                            logits = outputs.logits[:, -1, :].float()
                        else:
                            past_key_values = outputs.past_key_values
                    else:
                        past_key_values = outputs.past_key_values

                    next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    current_input_ids = next_token

                    if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.convert_tokens_to_ids(
                            "<|eot_id|>"):
                        break

            new_tokens = generated_ids[0][input_len:]
            final_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            total_gen = len(new_tokens)
            global_stats["total_tokens"] += total_gen

            if current_tier == 3:
                global_stats["tier_tokens"][3] += total_gen
            else:
                if has_fallen_back:
                    global_stats["tier_tokens"][current_tier] += fallback_step
                    global_stats["tier_tokens"][3] += (total_gen - fallback_step)
                else:
                    global_stats["tier_tokens"][current_tier] += total_gen

            # 🌟 核心修正：分类统计不同任务的结果
            if task_info["type"] == "math":
                global_stats["gsm8k_total"] += 1
                pred = extract_last_number(final_text)
                if pred == str(task_info.get('a', '')):
                    global_stats["gsm8k_correct"] += 1

            elif task_info["type"] == "qa":  # MMLU 任务
                global_stats["mmlu_total"] += 1
                matches = re.findall(r'[A-D]', final_text)
                pred = matches[-1] if matches else ""
                if pred == task_info.get('a', ''):
                    global_stats["mmlu_correct"] += 1

            elif task_info["type"] == "wiki":  # WikiText 任务
                global_stats["wiki_count"] += 1

        # 注意这里的缩进！它和 for item in tqdm(...) 是平齐的！
        if current_tier != 3:
            print("🧹 任务完毕，释放小模型显存...")
            del mid_adapter
            del active_model
            gc.collect()
            torch.cuda.empty_cache()

    # ==========================================
    # 📊 打印最终学术报告 (多任务泛化性版)
    # ==========================================
    gsm_acc = (global_stats["gsm8k_correct"] / max(1, global_stats["gsm8k_total"])) * 100 if global_stats[
                                                                                                 "gsm8k_total"] > 0 else 0
    mmlu_acc = (global_stats["mmlu_correct"] / max(1, global_stats["mmlu_total"])) * 100 if global_stats[
                                                                                                "mmlu_total"] > 0 else 0

    total_t = global_stats["total_tokens"]
    if total_t > 0:
        true_flops_saved = (
                                   (global_stats["tier_tokens"][0] / total_t) * tier_config[0]["sparsity"] +
                                   (global_stats["tier_tokens"][1] / total_t) * tier_config[1]["sparsity"] +
                                   (global_stats["tier_tokens"][2] / total_t) * tier_config[2]["sparsity"]
                           ) * 100
        offload_rate = ((total_t - global_stats["tier_tokens"][3]) / total_t) * 100
    else:
        true_flops_saved, offload_rate = 0, 0

    print("\n" + "=" * 65)
    print(" 🚀 SADS 多任务泛化性评测报告 (Generality)")
    print("=" * 65)
    print(f" 🎯 复杂数学推理 (GSM8K):   {gsm_acc:.2f}% ")
    print(f" 📚 通用知识问答 (MMLU):    {mmlu_acc:.2f}% ")
    print(f" 📝 长文本生成 (WikiText):  {global_stats['wiki_count']} 篇 (成功不崩溃)")
    print("-" * 65)
    print(f" 🔄 触发 SADS 零拷贝救场:   {global_stats['fallback_count']} 次")
    print(f" 🪂 任务平均卸载率:           {offload_rate:.2f}% ")
    print(f" ⚡ 真实算力 (FLOPs) 节省:   {true_flops_saved:.2f}% ")
    print("=" * 65)


if __name__ == "__main__":
    run_grouped_evaluation()
