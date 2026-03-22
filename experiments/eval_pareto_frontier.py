import torch
import math, json, re, os, sys, gc
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


import os
import csv


def load_local_mmlu(data_dir="/clzs_test011/qyh/dataset/data", num=20):
    """从本地加载 MMLU 数据集 (自动解析 CSV 格式，逼出 Fallback 的利器)"""
    data = []
    # 兼容 MMLU 可能的目录结构（直接在 data_dir 下，或者在 data_dir/test 下）
    target_dir = os.path.join(data_dir, "test") if os.path.exists(os.path.join(data_dir, "test")) else data_dir

    if not os.path.exists(target_dir):
        print(f"⚠️ 找不到 MMLU 目录: {target_dir}")
        return data

    csv_files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]

    for file in csv_files:
        with open(os.path.join(target_dir, file), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                # MMLU 标准格式: [题目, 选项A, 选项B, 选项C, 选项D, 正确答案]
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
                    if len(data) >= num:
                        return data
    return data
def load_gsm8k(path, num=30):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if "####" in d.get('answer', ''):
                data.append({"q": d['question'], "a": d['answer'].split("####")[1].strip(), "type": "math"})
            if len(data) >= num: break
    return data


def get_wiki_prompts():
    prompts = [
                  "Write a detailed Wikipedia-style article about the history of the Roman Empire, covering its rise and fall.",
                  "Explain the process of photosynthesis in extreme detail, as if you are writing a biology textbook chapter.",
                  "Write a comprehensive guide on how to build a computer from scratch, step by step.",
                  "Describe the plot and thematic elements of Shakespeare's Hamlet in a long essay format.",
                  "Provide a detailed geographical and cultural overview of the Amazon Rainforest."
              ] * 4
    return [{"q": p, "a": None, "type": "wiki"} for p in prompts]


# ==========================================
# 🚀 3. 异步分组调度终极评测 (彻底解决显存危机)
# ==========================================
# ==========================================
# 🚀 3. 异步分组调度终极评测 (彻底解决显存危机)
# ==========================================
def run_grouped_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    gsm8k_path = "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl"
    router_weight_path = "weights/router_mlp_4tiers.pth"

    # 🎯 极其关键：收紧阈值，逼出 Fallback 救场机制！
    # 🎯 极其关键：纯数据驱动的科学阈值配置！
    tier_config = {
        0: {"name": "30% 极简稀疏", "path": "experiments/sliced_llama3_8b_30", "tau": 6.0, "sparsity": 0.30},
        # 30% 暂时不扫，定个宽松的
        1: {"name": "15% 长文稀疏", "path": "experiments/sliced_llama3_8b_15", "tau": 2.8140, "sparsity": 0.15},
        # 👑 替换为 95th Percentile
        2: {"name": "10% 逻辑稀疏", "path": "experiments/sliced_llama3_8b_10", "tau": 2.3966, "sparsity": 0.10},
        # 👑 替换为 95th Percentile
        3: {"name": "0% 满血兜底", "path": base_model_path, "tau": float('inf'), "sparsity": 0.0}
    }
    print("\n📥 正在从本地直接加载综合评测大卷...")
    # 综合大考：本地数学 + 内部长文 + 本地综合知识 (共 60 题)
    all_tasks = (
            load_gsm8k("/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl", num=20) +
            get_wiki_prompts() +
            load_local_mmlu("/clzs_test011/qyh/dataset/data", num=20)
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    router = LightweightRouter(router_weight_path, tokenizer, device)

    # ⚠️ 之前这里的代码：all_tasks = load_gsm8k(...) + get_wiki_prompts()
    # 把上面的 MMLU 数据给覆盖掉了！我已经帮你删除了那行导致 Bug 的代码。

    print("\n[阶段 1/3]  启动预判,进行题目难度分组...")
    grouped_tasks = {0: [], 1: [], 2: [], 3: []}
    for task in all_tasks:
        if task["type"] == "math":
            msg = [{"role": "system",
                    "content": "You are a math expert. Solve step by step. Conclude with 'The final answer is [number]'."},
                   {"role": "user", "content": task['q']}]
        else:
            msg = [{"role": "system", "content": "You are a detailed assistant."},
                   {"role": "user", "content": task['q']}]

        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        tier = router.predict_tier(prompt)
        grouped_tasks[tier].append({"prompt": prompt, "task_info": task})

    for t, tasks in grouped_tasks.items():
        print(f"    Tier {t} ({tier_config[t]['name']}) 拦截了 {len(tasks)} 个任务")

    del router
    torch.cuda.empty_cache()

    print("\n[阶段 2/3]加载满血大模型 ")
    full_adapter, _ = hf_utils.get_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct",
                                                       model_path=base_model_path)
    full_model = full_adapter.model.to(device).eval()

    # 🎯 严密的全局统计账本
    global_stats = {
        "total_tokens": 0,
        "tier_tokens": {0: 0, 1: 0, 2: 0, 3: 0},  # 记录每个档位实际打工了多少 Token
        "gsm8k_correct": 0,
        "gsm8k_total": 0,
        "fallback_count": 0
    }

    print("\n[阶段 3/3]  启动异步分组流转引擎...")
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
            active_model = mid_adapter.model.to(device).eval()

        tau_threshold = tier_config[current_tier]['tau']

        for item in tqdm(tasks, desc=f"Tier {current_tier} 推理中"):
            prompt = item["prompt"]
            task_info = item["task_info"]

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            generated_ids = input_ids
            input_len = input_ids.shape[1]

            has_fallen_back = False
            fallback_step = -1

            for i in range(400):
                with torch.no_grad():
                    outputs = active_model(generated_ids)
                    logits = outputs.logits[:, -1, :].float()

                    if current_tier != 3 and not has_fallen_back:
                        entropy = torch.distributions.Categorical(logits=logits).entropy().item()
                        if math.isnan(entropy) or entropy > tau_threshold:
                            has_fallen_back = True
                            fallback_step = i
                            active_model = full_model
                            global_stats["fallback_count"] += 1
                            outputs = active_model(generated_ids)
                            logits = outputs.logits[:, -1, :].float()

                    next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                    if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.convert_tokens_to_ids(
                            "<|eot_id|>"):
                        break

            new_tokens = generated_ids[0][input_len:]
            final_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # 🎯 极其严谨的 Token 归属统计
            total_gen = len(new_tokens)
            global_stats["total_tokens"] += total_gen

            if current_tier == 3:
                # 满血档：所有 Token 都是大模型生成的
                global_stats["tier_tokens"][3] += total_gen
            else:
                if has_fallen_back:
                    # 发生回退：回退前的 Token 算给小模型，回退后的算给大模型
                    global_stats["tier_tokens"][current_tier] += fallback_step
                    global_stats["tier_tokens"][3] += (total_gen - fallback_step)
                else:
                    # 没回退：全是小模型的功劳
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
        # 严格套用刚才提供的数学公式
        true_flops_saved = (
                                   (global_stats["tier_tokens"][0] / total_t) * tier_config[0]["sparsity"] +
                                   (global_stats["tier_tokens"][1] / total_t) * tier_config[1]["sparsity"] +
                                   (global_stats["tier_tokens"][2] / total_t) * tier_config[2]["sparsity"]
                           ) * 100
        # 任务卸载率：不是大模型跑的 Token，都叫卸载
        offload_rate = ((total_t - global_stats["tier_tokens"][3]) / total_t) * 100
    else:
        true_flops_saved, offload_rate = 0, 0

    print("\n" + "=" * 60)
    print(f"📊 复杂推理 (GSM8K) 准确率: {gsm_acc:.2f}% ")
    print(f"📊 全局触发回退救场次数:     {global_stats['fallback_count']} 次")
    print(f"🚀 任务卸载率 (交由小模型):   {offload_rate:.2f}% ")
    print(f"🔥 真实算力白嫖比例 (FLOPs):  {true_flops_saved:.2f}% ")
    print("=" * 60)
if __name__ == "__main__":
    run_grouped_evaluation()