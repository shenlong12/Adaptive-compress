import torch
from transformers import AutoTokenizer
import math, sys, os, json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils


def load_few_gsm8k(file_path, num_samples=5):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q = data.get('question', '')
            if q:
                dataset.append(q)
            if len(dataset) >= num_samples: break
    return dataset


def profile_model_entropy(model, tokenizer, prompts, device, model_name_str):
    print(f"\n🔍 正在探测 [{model_name_str}] 的脑电波 (信息熵)...")
    all_entropies = []

    for prompt in prompts:
        messages = [
            {"role": "system",
             "content": "You are a math expert. Solve the problem step by step. Conclude your answer by saying 'The final answer is [number]'."},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)

        generated_ids = input_ids
        prompt_entropies = []

        # 只探测前 50 个 token 的生成状态，足以看出趋势
        for i in range(50):
            with torch.no_grad():
                outputs = model(generated_ids)
                logits = outputs.logits[:, -1, :].float()

                entropy = torch.distributions.Categorical(logits=logits).entropy().item()
                if not math.isnan(entropy):
                    prompt_entropies.append(entropy)

                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.convert_tokens_to_ids(
                        "<|eot_id|>"):
                    break

        if prompt_entropies:
            all_entropies.extend(prompt_entropies)

    avg_entropy = sum(all_entropies) / len(all_entropies)
    max_entropy = max(all_entropies)
    print(f"📊 [{model_name_str}] -> 平均熵值: {avg_entropy:.2f} | 峰值熵值: {max_entropy:.2f}")
    return avg_entropy, max_entropy


def run_entropy_profiling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    gsm8k_path = "/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl"

    print("=== 启动 LLaMA-3 多档位熵值体检系统 ===")

    full_adapter, tokenizer = hf_utils.get_model_and_tokenizer(model_name, model_path=model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    full_model = full_adapter.model.to(device).eval()

    mid_adapter, _ = hf_utils.load_sliced_model(model_name, "experiments/sliced_llama3_8b_mid", sparsity=0.25)
    mid_model = mid_adapter.model.to(device).eval()

    low_adapter, _ = hf_utils.load_sliced_model(model_name, "experiments/sliced_llama3_8b_low", sparsity=0.5)
    low_model = low_adapter.model.to(device).eval()

    prompts = load_few_gsm8k(gsm8k_path, num_samples=5)

    profile_model_entropy(full_model, tokenizer, prompts, device, "满血模型 (Full)")
    profile_model_entropy(mid_model, tokenizer, prompts, device, "二档模型 (25% 稀疏)")
    profile_model_entropy(low_model, tokenizer, prompts, device, "一档模型 (50% 稀疏)")

    print("\n✅ 体检完成！")


if __name__ == "__main__":
    run_entropy_profiling()