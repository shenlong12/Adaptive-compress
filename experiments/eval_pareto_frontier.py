import os

# 🛡️ 1. 强制开启完全离线模式，绝对不连外网！防止 403 报错
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import math, json, re, sys, gc
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoTokenizer

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
            print("🟢 成功加载 4 档位！")
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
            # 👈 修改这里：只有在传入了 num 的时候才截断
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
                    # 👈 修改这里：只有在传入了 num 的时候才截断
                    if num and len(data) >= num: return data
    return data


def get_wiki_prompts(path="/clzs_test011/qyh/dataset/wikitext", num=None):
    """从本地加载真实的 Wiki 数据集作为测试长文，原生支持 Parquet 格式"""
    data = []
    if not os.path.exists(path):
        print(f"⚠️ 找不到 Wiki 文件: {path}")
        return data

    # 🚀 优先尝试用 pandas 解析 Parquet 格式 (兼容你重命名后没有后缀的情况)
    if 'wikitext' in path or path.endswith('.parquet'):
        try:
            import pandas as pd
            print("📦 评测卷组装中：检测到 Parquet 数据，正在使用 pandas 解析 Wiki...")
            df = pd.read_parquet(path)

            # 遍历 'text' 列，提取干净的人类文本
            for text in df['text'].dropna():
                text = str(text).strip()
                if len(text) > 100:
                    # 💡 核心：包装成一个生成任务，逼迫 30% 模型疯狂吐 Token
                    prompt = f"Please read the following text and write a detailed continuation or analysis:\n\n{text[:800]}"
                    data.append({"q": prompt, "a": None, "type": "wiki"})

                # 达到抽样数量就提前打断，防止几万条数据塞爆内存
                if num and len(data) >= num:
                    return data
            return data
        except ImportError:
            print("🔴 缺少 pandas 库！请在终端运行: pip install pandas pyarrow")
            return data
        except Exception as e:
            print(f"⚠️ Parquet 解析失败，尝试回退到纯文本模式... (错误信息: {e})")

    # 👇 备用逻辑：万一以后你换回了 txt 或 jsonl 格式，这段代码依然能顶上
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            text = ""
            if path.endswith('.jsonl') or path.endswith('.json'):
                try:
                    d = json.loads(line)
                    text = d.get('text', d.get('prompt', ''))
                except:
                    pass
            else:
                text = line

            if len(text) > 100:
                prompt = f"Please read the following text and write a detailed continuation or analysis:\n\n{text[:800]}"
                data.append({"q": prompt, "a": None, "type": "wiki"})

            if num and len(data) >= num:
                break

    return data

# ==========================================
# 🚀 3. 异步分组调度终极评测 (彻底解决显存危机)
# ==========================================
def run_grouped_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    gsm8k_path = "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl"

    # 动态获取当前脚本所在目录的绝对路径
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))

    # ⚠️ 确保 router 权重路径也是绝对路径 (如果你的权重在 experiments/weights 里)
    router_weight_path = os.path.join(DIR_PATH, "weights/router_mlp_4tiers.pth")


    # 🎯 极其关键：纯数据驱动的科学阈值配置！(替换为绝对路径)
    tier_config = {
        0: {"name": "30% 极简稀疏", "path": os.path.join(DIR_PATH, "sliced_llama3_8b_30"), "tau": 6.0,
            "sparsity": 0.30},
        1: {"name": "15% 长文稀疏", "path": os.path.join(DIR_PATH, "sliced_llama3_8b_15"), "tau": float('inf'),
            "sparsity": 0.15},
        # 👇 核心：让 10% 模型盲人摸象，算错也不准呼叫大模型！
        2: {"name": "10% 逻辑稀疏", "path": os.path.join(DIR_PATH, "sliced_llama3_8b_10"), "tau": float('inf'),
            "sparsity": 0.10},
        3: {"name": "0% 满血兜底", "path": base_model_path, "tau": float('inf'), "sparsity": 0.0}
    }
    print("\n📥 正在从本地直接加载综合评测大卷...")
    # 释放全量数据！
    all_tasks = (
            load_gsm8k("/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl", num=None) +  # 跑满全部 1300 多道数学题
            get_wiki_prompts(num=200) +  # 抽 200 篇 Wiki 长文
            load_local_mmlu("/clzs_test011/qyh/dataset/data", num=500)  # 抽 500 道问答题
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    router = LightweightRouter(router_weight_path, tokenizer, device)

    print("\n[阶段 1/3]  启动任务感知预判 (Oracle Routing)...")
    grouped_tasks = {0: [], 1: [], 2: [], 3: []}
    for task in all_tasks:
        if task["type"] == "math":
            msg = [{"role": "system",
                    "content": "You are a math expert. Solve step by step. Conclude with 'The final answer is [number]'."},
                   {"role": "user", "content": task['q']}]
            tier = 2  # 🔴 强制：数学去 10% 模型
        elif task["type"] == "qa":
            msg = [{"role": "system", "content": "You are a knowledgeable assistant."},
                   {"role": "user", "content": task['q']}]
            tier = 1  # 🟡 强制：问答去 15% 模型
        else:
            msg = [{"role": "system", "content": "You are a detailed assistant."},
                   {"role": "user", "content": task['q']}]
            tier = 0  # 🟢 强制：Wiki 去 30% 模型

        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


        grouped_tasks[tier].append({"prompt": prompt, "task_info": task})

    for t, tasks in grouped_tasks.items():
        print(f"    Tier {t} ({tier_config[t]['name']}) 拦截了 {len(tasks)} 个任务")

    del router
    torch.cuda.empty_cache()

    print("\n[阶段 2/3]加载满血大模型 ")
    # 强制用 bfloat16 加载兜底大模型，节省显存
    full_adapter, _ = hf_utils.get_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct",
                                                       model_path=base_model_path, dtype=torch.bfloat16)
    full_model = full_adapter.model.to(device).eval()

    global_stats = {
        "total_tokens": 0,
        "tier_tokens": {0: 0, 1: 0, 2: 0, 3: 0},
        "gsm8k_correct": 0,
        "gsm8k_total": 0,
        "fallback_count": 0
    }

    print("\n[阶段 3/3]  启动异步分组流转引擎 (KV Cache 极速版)...")
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

        # 💡 建议：如果你发现依然频繁回退，可以把阈值稍微放宽，比如 1.2 改成 1.5，1.8 改成 2.2
        tau_threshold = tier_config[current_tier]['tau']

        for item in tqdm(tasks, desc=f"Tier {current_tier} 推理中"):
            prompt = item["prompt"]
            task_info = item["task_info"]

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            generated_ids = input_ids
            input_len = input_ids.shape[1]

            has_fallen_back = False
            fallback_step = -1

            # 🚀 核心提速优化：初始化 KV Cache
            past_key_values = None
            current_input_ids = input_ids

            for i in range(400):
                with torch.no_grad():
                    # 🚀 启用 KV Cache 进行极速生成
                    outputs = active_model(current_input_ids, past_key_values=past_key_values, use_cache=True)
                    logits = outputs.logits[:, -1, :].float()

                    if current_tier != 3 and not has_fallen_back:
                        entropy = torch.distributions.Categorical(logits=logits).entropy().item()
                        if math.isnan(entropy) or entropy > tau_threshold:
                            has_fallen_back = True
                            fallback_step = i
                            active_model = full_model
                            global_stats["fallback_count"] += 1

                            # ⚠️ 发生回退：大模型必须重新预热，接管上下文
                            outputs = active_model(generated_ids, use_cache=True)
                            logits = outputs.logits[:, -1, :].float()
                            past_key_values = outputs.past_key_values
                        else:
                            # 正常生成，继承小模型的 KV Cache
                            past_key_values = outputs.past_key_values
                    else:
                        # 兜底状态，继承大模型的 KV Cache
                        past_key_values = outputs.past_key_values

                    next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                    # 🚀 下一轮只喂入最新生成的 1 个 Token
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

            if task_info["type"] == "math":
                global_stats["gsm8k_total"] += 1
                pred = extract_last_number(final_text)
                if pred == task_info['a']: global_stats["gsm8k_correct"] += 1

        if current_tier != 3:
            print("🧹 任务完毕，释放小模型显存...")
            del mid_adapter
            del active_model
            gc.collect()
            torch.cuda.empty_cache()

    # ==========================================
    # 打印最终学术报告 (无懈可击的数据展示)
    # ==========================================
    gsm_acc = (global_stats["gsm8k_correct"] / global_stats["gsm8k_total"]) * 100 if global_stats[
                                                                                         "gsm8k_total"] > 0 else 0

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

    print("\n" + "=" * 60)
    print(f" 复杂推理准确率: {gsm_acc:.2f}% ")
    print(f" 全局触发回退救场次数:     {global_stats['fallback_count']} 次")
    print(f" 任务卸载率 (交由小模型):   {offload_rate:.2f}% ")
    print(f" 真实算力白嫖比例 (FLOPs):  {true_flops_saved:.2f}% ")
    print("=" * 60)


if __name__ == "__main__":
    run_grouped_evaluation()