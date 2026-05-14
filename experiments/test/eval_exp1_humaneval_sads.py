import os
import math
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🛡️ 强制开启完全离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from slicegpt import hf_utils


# ==========================================
# 1. 纯本地 HumanEval 加载器
# ==========================================
def load_local_humaneval(file_path):
    """
    离线读取 HumanEval.jsonl 文件
    """
    print(f"📖 正在加载本地 HumanEval 数据集: {file_path}")
    dataset = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"🔴 找不到文件: {file_path}。请去下载并放好！")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    print(f"✅ 成功加载 {len(dataset)} 个编程任务。")
    return dataset


# ==========================================
# 2. SADS 动态代码生成引擎 (白盒自回归)
# ==========================================
def generate_sads_code(full_model, mid_model, tokenizer, prompt, tau_threshold, max_new_tokens=512, device="cuda"):
    mid_model.eval()
    full_model.eval()

    # 按照 LLaMA-3 Coder 习惯，我们可以稍微加一点系统提示，也可以直接续写
    # HumanEval 通常直接续写 prompt 效果最好
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated_ids = input_ids.clone()
    fallback_count = 0

    # 获取停止符
    eos_token_id = tokenizer.eos_token_id
    eot_token_id = tokenizer.convert_tokens_to_ids(
        "<|eot_id|>") if "<|eot_id|>" in tokenizer.get_vocab() else eos_token_id
    stop_tokens = [eos_token_id, eot_token_id]

    # 1. 预热期
    with torch.no_grad():
        outputs = mid_model(input_ids, use_cache=True)
        past_key_values_mid = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    # 2. 连续生成期
    for step in range(max_new_tokens):
        entropy = torch.distributions.Categorical(logits=next_token_logits.float()).entropy().item()

        # 🚨 SADS 雷达：遇到复杂的代码逻辑（如变量声明、缩进控制），大模型空降！
        if math.isnan(entropy) or entropy > tau_threshold:
            fallback_count += 1
            with torch.no_grad():
                outputs_full = full_model(generated_ids, use_cache=True)
                next_token_logits = outputs_full.logits[:, -1, :]

        # HumanEval 通常使用贪婪解码 (Greedy Decoding) 来测 Pass@1 (T=0)
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() in stop_tokens:
            break

        # 小模型步进
        with torch.no_grad():
            outputs = mid_model(next_token, past_key_values=past_key_values_mid, use_cache=True)
            past_key_values_mid = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

    # 仅提取新生成的代码部分
    response_ids = generated_ids[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response_text, fallback_count, len(response_ids)


# 替换原本的 run_exp1_humaneval 主控引擎
def run_exp1_humaneval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    DIR_PATH = "/clzs_test011/qyh/slice_new/TransformerCompression-main/TransformerCompression-main/experiments"

    # 🚨 你的 HumanEval.jsonl 路径
    HUMANEVAL_PATH = "/clzs_test011/qyh/dataset/human-eval-v2-20210705.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    dataset = load_local_humaneval(HUMANEVAL_PATH)

    print("\n[加载 Dense LLaMA-3-8B (全量后盾) ...]")
    full_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)

    # 固定使用 30% 稀疏的底座
    sparsity = 0.30
    suffix = "30"
    model_dir = os.path.join(DIR_PATH, f"llama3_8b_{suffix}")

    mid_adapter, _ = hf_utils.load_sliced_model("meta-llama/Meta-Llama-3-8B-Instruct", model_dir, sparsity=sparsity)
    mid_model = mid_adapter.model.to(torch.bfloat16).to(device)

    # 🌟 核心修改：对代码任务进行低阈值扫雷！
    # 分别测试: 1.5, 1.0, 0.5, 0.1
    tau_candidates = [0.5]

    print("\n" + "=" * 65)
    print(f" 🚀 开启 HumanEval 极限低熵扫雷测试")
    print("=" * 65)

    for tau in tau_candidates:
        output_file = f"sads_humaneval_30_tau{tau}.jsonl"
        print(f"\n[测试阈值 Tau = {tau}] 结果将保存至: {output_file}")

        total_fallbacks = 0
        total_tokens = 0

        with open(output_file, "w", encoding="utf-8") as f_out:
            for item in tqdm(dataset, desc=f"Tau={tau} 生成中"):
                task_id = item["task_id"]
                raw_prompt = item["prompt"]

                # 使用刚才抢救成功的 Chat Template 包裹
                messages = [
                    {"role": "system",
                     "content": "You are an expert Python programmer. Please complete the following Python code block. Only output the python code, do not output any explanations."},
                    {"role": "user", "content": f"Complete this code:\n```python\n{raw_prompt}\n```"}
                ]
                chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                completion, fallbacks, gen_tokens = generate_sads_code(
                    full_model, mid_model, tokenizer, chat_prompt,
                    tau_threshold=tau, device=device
                )

                clean_code = completion
                if "```python" in completion:
                    clean_code = completion.split("```python")[1].split("```")[0]
                elif "```" in completion:
                    clean_code = completion.split("```")[1].split("```")[0]

                final_completion = clean_code.strip()

                f_out.write(json.dumps({
                    "task_id": task_id,
                    "completion": final_completion
                }) + "\n")
                f_out.flush()

        fallback_rate = total_fallbacks / total_tokens if total_tokens > 0 else 0
        real_sparsity = (1.0 - fallback_rate) * sparsity

        print(f"✅ Tau={tau} 生成完成！")
        print(f"📊 救场率: {fallback_rate * 100:.1f}% | 实测稀疏度: {real_sparsity * 100:.1f}%")


if __name__ == "__main__":
    run_exp1_humaneval()