import torch
from transformers import AutoTokenizer
import sys, os, json, re
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils


def load_gsm8k_training_data(file_path, num_samples=400):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q = data.get('question', '')
            a_str = data.get('answer', '')
            if q and "####" in a_str:
                true_ans_str = a_str.split("####")[1].strip().replace(",", "")
                dataset.append({"raw_prompt": q, "true_answer": true_ans_str})
            if len(dataset) >= num_samples: break
    return dataset


def extract_last_number(text):
    text = text.replace(",", "")
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        ans = numbers[-1]
        if ans.endswith(".0"): ans = ans[:-2]
        return ans
    return None


def generate_adaptive_labels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    gsm8k_path = "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl"
    output_file = "experiments/llama3_adaptive_router_data.jsonl"

    # 我们抽取 400 道题来训练这个轻量级接口，足够了
    num_samples = 400
    generate_length = 300

    print("=== 启动【自适应架构】数据打标流水线 ===")

    # 加载 Tokenizer
    _, tokenizer = hf_utils.get_model_and_tokenizer(model_name, model_path=model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("📥 正在加载 50% 稀疏模型 (一档)...")
    low_adapter, _ = hf_utils.load_sliced_model(model_name, "experiments/sliced_llama3_8b_low", sparsity=0.5)
    low_model = low_adapter.model.to(device).eval()

    print("📥 正在加载 25% 稀疏模型 (二档)...")
    mid_adapter, _ = hf_utils.load_sliced_model(model_name, "experiments/sliced_llama3_8b_mid", sparsity=0.25)
    mid_model = mid_adapter.model.to(device).eval()

    train_data = load_gsm8k_training_data(gsm8k_path, num_samples)
    print(f"✅ 准备对 {len(train_data)} 道题目进行大逃杀评估。\n")

    labeled_dataset = []
    counts = {0: 0, 1: 0, 2: 0}

    print("🚀 开始寻找最大利益化点！")
    for item in tqdm(train_data):
        raw_prompt = item["raw_prompt"]
        true_ans = item["true_answer"]

        messages = [
            {"role": "system",
             "content": "You are a math expert. Solve the problem step by step. Conclude your answer by saying 'The final answer is [number]'."},
            {"role": "user", "content": raw_prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
        input_len = input_ids.shape[1]

        # --- 阶段 1：测试一档模型 (50% 稀疏) ---
        with torch.no_grad():
            low_outputs = low_model.generate(
                input_ids, max_new_tokens=generate_length, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        low_text = tokenizer.decode(low_outputs[0][input_len:], skip_special_tokens=True)
        low_pred = extract_last_number(low_text)

        if low_pred == true_ans:
            label = 0  # 一档就能搞定，收益最大化！
        else:
            # --- 阶段 2：如果一档失败，测试二档模型 (25% 稀疏) ---
            with torch.no_grad():
                mid_outputs = mid_model.generate(
                    input_ids, max_new_tokens=generate_length, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            mid_text = tokenizer.decode(mid_outputs[0][input_len:], skip_special_tokens=True)
            mid_pred = extract_last_number(mid_text)

            if mid_pred == true_ans:
                label = 1  # 二档能搞定，平衡算力
            else:
                label = 2  # 太难了，必须留给满血模型兜底

        counts[label] += 1
        labeled_dataset.append({
            "prompt": raw_prompt,  # 路由器只需看原始问题
            "label": label
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        for d in labeled_dataset:
            f.write(json.dumps(d) + "\n")

    print("\n" + "=" * 60)
    print("🎉 自适应训练集生成完毕！")
    print("=" * 60)
    print(f"📁 数据已保存至: {output_file}")
    print("📊 难度分布矩阵 (最大利益化分布):")
    print(f"  - 级别 0 (50% 稀疏可解, 极省算力): {counts[0]} 题")
    print(f"  - 级别 1 (25% 稀疏可解, 平衡算力): {counts[1]} 题")
    print(f"  - 级别 2 (必须唤醒满血大模型):      {counts[2]} 题")
    print("=" * 60)


if __name__ == "__main__":
    generate_adaptive_labels()