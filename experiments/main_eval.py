import os
import json
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM

# ==========================================
# 1. 全局配置 & FLOPs 与 稀疏率监控器
# ==========================================
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_ID = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"


class FLOPsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.flops_ratios = []
        self.sparse_routed_tokens = 0
        self.total_tokens = 0

    def record_step(self, d_active, d_dense, is_routed_to_sparse=False, token_count=1):
        ratio = (d_active / d_dense) ** 2
        self.flops_ratios.extend([ratio] * token_count)
        self.total_tokens += token_count
        if is_routed_to_sparse:
            self.sparse_routed_tokens += token_count

    def get_metrics(self):
        avg_flops = sum(self.flops_ratios) / len(self.flops_ratios) if self.flops_ratios else 1.0
        eff_sparsity = self.sparse_routed_tokens / self.total_tokens if self.total_tokens > 0 else 0.0
        return avg_flops, eff_sparsity


tracker = FLOPsTracker()


# ==========================================
# 2. 方法实现 (DejaVu 简化版 & 模型加载器)
# ==========================================
class DejaVuPredictor(nn.Module):
    """简化的 2-layer MLP predictor for DejaVu"""

    def __init__(self, hidden_size, sparsity=0.25):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 4)
        self.fc2 = nn.Linear(hidden_size // 4, hidden_size)
        self.target_keep_ratio = 1.0 - sparsity

    def forward(self, x):
        logits = self.fc2(F.relu(self.fc1(x)))
        # 实际实现中这里会使用 top-k mask 来选择 active neurons
        # 此处演示动态 tracking 逻辑
        d_dense = x.shape[-1]
        d_active = int(d_dense * self.target_keep_ratio)
        tracker.record_step(d_active, d_dense, token_count=x.shape[0] * x.shape[1])
        return logits


def load_model(method):
    print(f"Loading model for method: {method}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 基础模型加载
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
    )

    # 根据 Method 进行结构修改或 Wrapper 注入
    if method == "Dense":
        pass  # 原始模型

    elif method.startswith("SliceGPT"):
        sparsity = int(method.split("-")[-1].replace("%", "")) / 100.0
        print(f"Applying SliceGPT static structural pruning (Sparsity: {sparsity})...")
        # TODO: 调用官方 SliceGPT apply_mask 逻辑
        # 模拟记录全局静态 FLOPs 减少
        tracker.record_step(d_dense=100, d_active=100 * (1 - sparsity), token_count=1)

    elif method == "DejaVu":
        print("Applying DejaVu dynamic unstructured sparsity (~25%)...")
        hidden_size = model.config.hidden_size
        # 为每层注入 Predictor (伪代码展示注入逻辑)
        for layer in model.model.layers:
            layer.dejavu_predictor = DejaVuPredictor(hidden_size, sparsity=0.25).to(model.device)
            # layer.forward 需要被 hook 以执行 predictor 并 mask attention/mlp

    elif method == "SADS-Macro":
        print("Applying SADS Macro-only (Pre-Router, No Entropy Monitoring)...")
        # TODO: 加载你的 SADS Macro-Router 权重并替换 LLaMA 层

    elif method == "SADS-Full":
        print("Applying SADS Full (Pre-Router + Entropy + Zero-copy)...")
        # TODO: 加载完整的 SADS 动态系统

    model.eval()
    return model, tokenizer


# ==========================================
# 3. 评测函数 (WikiText-2, LM-Harness wrapper)
# ==========================================
def eval_wikitext2(model, tokenizer, method_name):
    """使用 Stride=512 的滑动窗口计算 WikiText-2 PPL"""
    from datasets import load_dataset

    # 直接精确指向这个 Parquet 文件
    data_file = "/clzs_test011/qyh/dataset/wikitext"

    # 告诉 load_dataset 这是一个 parquet 文件，并且把它作为 test split 加载
    dataset = load_dataset("parquet", data_files={"test": data_file}, split="test")

    # Parquet 格式读出来的默认列名通常是 "text"
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    # 手动将最大滑动窗口限制为 2048，防止 Attention 矩阵爆炸
    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    # 手动控制 tqdm 格式以满足终端打印要求
    total_steps = (seq_len - 1) // stride + 1

    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - prev_end_loc  # 预测步长

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Context 部分不计算 Loss

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        # 自定义终端打印进度
        step = i // stride + 1
        current_ppl = torch.exp(torch.stack(nlls).mean()).item()
        print(f"\r[Exp1][{method_name}][WikiText-2] Progress: {step}/{total_steps} | Current PPL: {current_ppl:.2f}",
              end="")

        if end_loc == seq_len:
            break

    print()  # 换行
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return {"PPL": ppl}


def eval_lm_harness(model, tokenizer, dataset_name, task_name, method_name, num_fewshot, local_path=None):

    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)

    print(f"\n[Exp1][{method_name}][{dataset_name}] Starting LM-Harness evaluation...")

    # 如果提供了本地 .jsonl 路径，则动态覆盖 LM-Harness 的数据源
    if local_path and local_path.endswith('.jsonl'):
        tasks_list = [{
            "task": task_name,
            "dataset_path": "json",
            # 增加 "train": local_path，骗过 lm_eval 的 few-shot 初始化检查
            "dataset_kwargs": {"data_files": {"test": local_path, "train": local_path}}
        }]
    else:
        # 如果没有本地路径（比如 MMLU），就保持原样，让它自动下载
        tasks_list = [task_name]

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks_list,
        num_fewshot=num_fewshot,
        batch_size=1,
    )

    metric_map = {
        "gsm8k": "exact_match,strict-match",  # 常见 GSM8K accuracy 键名
        "mmlu": "acc,none",  # MMLU accuracy 键名
        "mbpp": "pass@1,none"  # MBPP pass@1 键名
    }

    # 提取核心指标
    task_res = results['results'].get(task_name, {})

    acc = None
    # 尝试按已知键名匹配
    for key in metric_map[task_name].split(','):
        if key in task_res:
            acc = task_res[key]
            break

    # 如果没匹配到，遍历寻找第一个数字（跳过 alias 等字符串）
    if acc is None:
        for val in task_res.values():
            if isinstance(val, (int, float)):
                acc = val
                break

    # 终极兜底
    if acc is None:
        acc = 0.0

    print(f"[Exp1][{method_name}][{dataset_name}] Finished. Score: {acc}")
    return {"Accuracy": acc}


# ==========================================
# 4. 主控与调度逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="LLM Sparsity Evaluation Script")
    parser.add_argument("--method", type=str, default="ALL",
                        help="Specific method to run (e.g., Dense, SliceGPT-10%, SADS-Full) or ALL")
    args = parser.parse_args()

    all_methods = [
        "Dense",
        "SliceGPT-10%", "SliceGPT-15%", "SliceGPT-30%",
        "DejaVu",
        "SADS-Macro", "SADS-Full"
    ]

    tasks_config = {
        "WikiText2": {
            "type": "custom", "metric_key": "PPL",
            "path": "/clzs_test011/qyh/dataset/wikitext"
        },
        "GSM8K": {
            "type": "harness", "task": "gsm8k", "shots": 0, "metric_key": "Accuracy",
            "path": "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl"
        },
        "MBPP": {
            "type": "harness", "task": "mbpp", "shots": 0, "metric_key": "Accuracy",
            "path": "/clzs_test011/qyh/dataset/mbpp.jsonl"
        },
        "MMLU": {
            "type": "harness", "task": "mmlu", "shots": 5, "metric_key": "Accuracy",
            "path": None  # 图片里没看到 MMLU，先设为 None，它会尝试联网拉取
        },
    }

    methods_to_run = all_methods if args.method == "ALL" else [args.method]

    for method in methods_to_run:
        model, tokenizer = load_model(method)

        for dataset, config in tasks_config.items():
            result_file = os.path.join(RESULTS_DIR, f"exp1_{method}_{dataset}.json")
            if os.path.exists(result_file):
                print(f"Skipping {method} on {dataset}, already exists.")
                continue

            tracker.reset()

            # ---> 这里需要修改：把 config["path"] 传进去 <---
            if config["type"] == "custom":
                metrics = eval_wikitext2(model, tokenizer, method, config["path"])
            else:
                metrics = eval_lm_harness(
                    model, tokenizer, dataset, config["task"], method, config["shots"], config["path"]
                )

            avg_flops, eff_sparsity = tracker.get_metrics()

            # 保存该组合的 JSON
            output_data = {
                "Method": method,
                "Dataset": dataset,
                "MainMetric": metrics[list(metrics.keys())[0]],
                "Avg_FLOPs_ratio": avg_flops,
                "Effective_Sparsity": eff_sparsity if "SADS" in method else "N/A"
            }

            with open(result_file, 'w') as f:
                json.dump(output_data, f, indent=4)

        # 释放内存供下一个 Method 使用
        del model
        torch.cuda.empty_cache()

    generate_main_table()


def generate_main_table():
    """汇总所有 JSON 生成 Table 2 CSV"""
    print("\nGenerating final results/main_table.csv ...")
    csv_file = os.path.join(RESULTS_DIR, "main_table.csv")

    headers = ["Method", "Sparsity", "WikiText2_PPL", "GSM8K_Acc", "MMLU_Acc", "MBPP_Pass1", "Avg_FLOPs_ratio",
               "SADS_Effective_Sparsity"]
    all_methods_seen = set()
    rows = {}

    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith("exp1_") and filename.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, filename), 'r') as f:
                data = json.load(f)

            method = data["Method"]
            dataset = data["Dataset"]
            all_methods_seen.add(method)

            if method not in rows:
                sparsity_label = method.split("-")[-1] if "%" in method else (
                    "25%" if method == "DejaVu" else ("~25%" if "SADS" in method else "0%"))
                rows[method] = {h: "N/A" for h in headers}
                rows[method]["Method"] = method
                rows[method]["Sparsity"] = sparsity_label

            if dataset == "WikiText2": rows[method]["WikiText2_PPL"] = f'{data["MainMetric"]:.2f}'
            if dataset == "GSM8K": rows[method]["GSM8K_Acc"] = f'{data["MainMetric"] * 100:.1f}%'
            if dataset == "MMLU": rows[method]["MMLU_Acc"] = f'{data["MainMetric"] * 100:.1f}%'
            if dataset == "MBPP": rows[method]["MBPP_Pass1"] = f'{data["MainMetric"] * 100:.1f}%'

            # 使用最后一个任务的 FLOPs 统计作为均值（或可改写为累加平均）
            rows[method]["Avg_FLOPs_ratio"] = f'{data["Avg_FLOPs_ratio"]:.2f}x'
            if data["Effective_Sparsity"] != "N/A":
                rows[method]["SADS_Effective_Sparsity"] = f'{data["Effective_Sparsity"] * 100:.1f}%'

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for method in sorted(list(all_methods_seen)):  # 可以自定义排序规则以匹配 Table 2
            writer.writerow(rows[method])

    print(f"Done! Results written to {csv_file}")


if __name__ == "__main__":
    main()