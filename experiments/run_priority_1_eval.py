import argparse
import json
import torch
import os
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text):
    """提取 GSM8K 的数字答案"""
    # 优先匹配规范的 #### 输出
    match = re.search(r'####\s*(-?\d+)', text)
    if match:
        return match.group(1)
    # 备用方案：抓取最后出现的数字，处理可能带有千位分隔符的情况
    text = text.replace(',', '')
    numbers = re.findall(r'-?\d+', text)
    return numbers[-1] if numbers else None


def load_local_dataset(file_path):
    """读取本地 JSONL 数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    # 强制离线，防止意外联网卡死
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--model_type", type=str, choices=["dense", "slicegpt", "sads"], required=True)
    parser.add_argument("--dataset_path", type=str, default="/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl")
    args = parser.parse_args()

    print(f"========== 实验启动 ==========")
    print(f"正在加载 {args.model_type.upper()} 模型...")

    # 加载 Tokenizer，并确保 pad_token 设置正确
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    model.eval()
    print("✅ 模型加载成功！")

    dataset = load_local_dataset(args.dataset_path)
    dataset = dataset[:200]
    total_count = len(dataset)
    correct_count = 0

    print(f"✅ 成功读取数据集: {args.dataset_path}，共 {total_count} 条测试例。")
    print(f"🚀 开始推理...")

    for i, item in enumerate(tqdm(dataset, desc="GSM8K 推理进度")):
        question = item.get('question', item.get('instruction', ''))
        gold_num = extract_answer(item.get('answer', item.get('output', '')))

        # 使用 System Prompt 引导模型遵循 GSM8K 的输出格式
        messages = [
            {"role": "system",
             "content": "You are a helpful mathematical reasoning assistant. Please reason step by step and always put your final answer after '####'."},
            {"role": "user", "content": question}
        ]

        # 核心改动 1：套用 LLaMA-3 专属的 Chat Template
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1024,  # 核心改动 2：将最大生成长度放宽到 1024，防止 CoT 被截断
                do_sample=False,  # 贪心解码，保证测试的确定性
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 只解码模型新生成的部分
        input_length = input_ids.shape[1]
        pred_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        pred_num = extract_answer(pred_text)

        if str(pred_num) == str(gold_num):
            correct_count += 1

    accuracy = (correct_count / total_count) * 100
    print("\n" + "=" * 40)
    print(f"[{args.model_type.upper()}] 最终评估结果")
    print(f"总计题数: {total_count}")
    print(f"准确率 (Accuracy): {accuracy:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()