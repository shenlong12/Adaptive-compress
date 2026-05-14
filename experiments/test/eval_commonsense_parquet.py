import os
import torch
import math
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🛡️ 强制开启离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from slicegpt import hf_utils


# ==========================================
# 1. 智能数据适配器 (完美解决不同数据集格式问题)
# ==========================================
def extract_qa_data(row, dataset_name):
    """根据不同的数据集名称，提取题目、选项和正确答案"""
    if dataset_name in ["arc", "obqa"]:
        # ARC 和 OBQA 的结构类似
        question = row.get('question', row.get('question_stem', ''))

        # HuggingFace 的 Parquet 读取后，choices 通常是个 dict: {'text': [...], 'label': [...]}
        # 也有可能是个 numpy array，这里做个兼容
        choices_data = row['choices']
        if isinstance(choices_data, dict):
            choices = choices_data['text']
        else:
            choices = choices_data.tolist()['text'] if hasattr(choices_data, 'tolist') else choices_data

        target_answer = row['answerKey']  # 例如 'A', 'B', '1', '2'

        # ARC 有时答案用 1,2,3,4 表示，需要转成 A,B,C,D
        if target_answer in ['1', '2', '3', '4']:
            target_answer = chr(ord('A') + int(target_answer) - 1)

        return question, choices, target_answer

    elif dataset_name == "hellaswag":
        question = row['ctx']
        choices = row['endings']
        # HellaSwag 的 label 是 '0', '1', '2', '3'
        target_answer = chr(ord('A') + int(row['label']))
        return question, choices, target_answer

    else:
        raise ValueError(f"🔴 不支持的数据集类型: {dataset_name}")


def format_multiple_choice(question, choices):
    prompt = f"Question: {question}\n"
    valid_letters = []
    for i, choice in enumerate(choices):
        letter = chr(ord('A') + i)
        prompt += f"{letter}. {choice}\n"
        valid_letters.append(letter)
    prompt += "Answer: "
    return prompt, valid_letters


# ==========================================
# 2. SADS 核心引擎 (代码极简，速度极快)
# ==========================================
def predict_choice_sads(full_model, mid_model, tokenizer, prompt, valid_letters, tau_threshold, device="cuda"):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False)[-1] for letter in valid_letters]

    with torch.no_grad():
        outputs = mid_model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]

        entropy = torch.distributions.Categorical(logits=next_token_logits.float()).entropy().item()

        fallback_used = False

        # 🚨 SADS 雷达预警
        if math.isnan(entropy) or entropy > tau_threshold:
            fallback_used = True
            outputs_full = full_model(input_ids, use_cache=True)
            next_token_logits = outputs_full.logits[:, -1, :]

        mask = torch.full_like(next_token_logits, -float('Inf'))
        for idx in letter_ids:
            mask[0, idx] = next_token_logits[0, idx]

        predicted_token_id = torch.argmax(mask, dim=-1).item()

        if predicted_token_id in letter_ids:
            predicted_letter = valid_letters[letter_ids.index(predicted_token_id)]
        else:
            predicted_letter = valid_letters[0]

    return predicted_letter, fallback_used


# ==========================================
# 3. 终极主控区
# ==========================================
def run_commonsense_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    DIR_PATH = "/clzs_test011/qyh/slice_new/TransformerCompression-main/TransformerCompression-main/experiments"

    # 🎯 === 任务配置区 === 🎯
    DATASET_NAME = "hellaswag"  # 填入 "arc", "obqa", 或 "hellaswag"
    DATASET_PATH = "/clzs_test011/qyh/dataset/hellaswag.parquet"  # 你的 Parquet 文件路径
    # ==========================

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    print(f"\n[加载 LLaMA-3 满血与稀疏模型...]")
    full_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)
    full_model.eval()

    # 以 40% 的黄金块为例
    sparsity = 0.20
    model_dir = os.path.join(DIR_PATH, f"llama3_8b_20")
    mid_adapter, _ = hf_utils.load_sliced_model("meta-llama/Meta-Llama-3-8B-Instruct", model_dir, sparsity=sparsity)
    mid_model = mid_adapter.model.to(torch.bfloat16).to(device)
    mid_model.eval()

    TAU_THRESHOLD = 4.3

    print("\n" + "=" * 50)
    print(f"🚀 SADS 评测启动 | 数据集: {DATASET_NAME.upper()} | 格式: Parquet")
    print("=" * 50)

    print(f"📦 正在读取 {DATASET_PATH} ...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"✅ 成功加载 {len(df)} 道题目！")

    correct_count = 0
    total_count = 0
    fallback_count = 0

    # 使用 iterrows 遍历 dataframe
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_q, raw_c, target_ans = extract_qa_data(row, DATASET_NAME)
        except Exception as e:
            print(f"\n⚠️ 第 {index} 行数据解析失败跳过, 错误: {e}")
            continue

        prompt, valid_letters = format_multiple_choice(raw_q, raw_c)

        pred, fallback = predict_choice_sads(
            full_model, mid_model, tokenizer, prompt, valid_letters, TAU_THRESHOLD, device
        )

        if pred == target_ans:
            correct_count += 1
        if fallback:
            fallback_count += 1
        total_count += 1

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    fallback_rate = (fallback_count / total_count) * 100 if total_count > 0 else 0

    print(f"\n🏆 {DATASET_NAME.upper()} 战报汇总 🏆")
    print(f"🎯 最终准确率 (Accuracy): {accuracy:.2f}%")
    print(f"🛡️ 最终救场率 (Fallback): {fallback_rate:.2f}%")


if __name__ == "__main__":
    run_commonsense_eval()