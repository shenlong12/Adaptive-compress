import torch
import math, sys, os, json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils


def load_mixed_prompts(wiki_path, gsm8k_path, samples_per_dataset=200):
    prompts = []
    with open(wiki_path, 'r', encoding='utf-8') as f:
        wiki_count = 0
        for line in f:
            words = line.strip().split()
            if len(words) > 15:
                prompts.append(" ".join(words[:15]))
                wiki_count += 1
            if wiki_count >= samples_per_dataset: break

    with open(gsm8k_path, 'r', encoding='utf-8') as f:
        gsm_count = 0
        for line in f:
            data = json.loads(line)
            q = data.get('question', '')
            if q:
                prompts.append(" ".join(q.split()[:20]))
                gsm_count += 1
            if gsm_count >= samples_per_dataset: break
    return prompts


def generate_llama3_training_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # 🎯 替换为你下载好的 LLaMA-3 绝对路径
    model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"

    wiki_path = "/clzs_test011/qyh/dataset/wiki.test"
    gsm8k_path = "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl"

    # 评判标准 (LLaMA-3 的熵值分布可能与 OPT 不同，我们可以稍微收紧一点阈值)
    tau = 3.5
    generate_length = 15
    fallback_threshold = 0.4

    print("=== 正在加载 LLaMA-3-8B (s_low, 50%稀疏) 用于难度探测 ===")
    low_adapter, tokenizer = hf_utils.load_sliced_model(model_name, "experiments/sliced_llama3_8b_low", sparsity=0.5)
    low_model = low_adapter.model.to(device).eval()

    # 修复 LLaMA-3 可能没有默认 pad_token 的问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("正在加载本地文本库...")
    prompts = load_mixed_prompts(wiki_path, gsm8k_path, samples_per_dataset=200)
    print(f"✅ 准备了 {len(prompts)} 条探测数据。")

    output_data = []

    print("\n🚀 开始为 LLaMA-3 自动化打标签...")
    for prompt in tqdm(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated_ids = input_ids

        fallback_count = 0

        for i in range(generate_length):
            with torch.no_grad():
                outputs = low_model(generated_ids)
                logits = outputs.logits[:, -1, :].float()

                for token_id in set(generated_ids[0].tolist()):
                    logits[0, token_id] /= 1.15

                entropy = torch.distributions.Categorical(logits=logits).entropy().item()

                if math.isnan(entropy) or entropy > tau:
                    fallback_count += 1

                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        fallback_rate = fallback_count / generate_length
        label = 1 if fallback_rate > fallback_threshold else 0

        output_data.append({
            "prompt": prompt,
            "fallback_rate": round(fallback_rate, 3),
            "label": label
        })

    # 保存为 LLaMA-3 专属的 JSONL 文件
    output_file = "experiments/llama3_router_training_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"\n🎉 LLaMA-3 专属 Router 训练集已保存至: {output_file}")

    label_1_count = sum(1 for d in output_data if d['label'] == 1)
    label_0_count = len(output_data) - label_1_count
    print(f"📊 数据分布: 复杂任务(Label 1): {label_1_count}条 | 简单任务(Label 0): {label_0_count}条")


if __name__ == "__main__":
    generate_llama3_training_data()