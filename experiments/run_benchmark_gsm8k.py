import torch
import json
import re
import sys, os
from tqdm import tqdm
from transformers import AutoTokenizer

# 借用你之前的 slicegpt 加载库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils

# ==========================================
# 1. 核心配置与阈值 (你跑出来的真实数据)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
SLICED_MODEL_PATH = "experiments/sliced_llama3_8b_15"  # 使用 15% 稀疏模型作为主力生成器
DATASET_PATH = "/clzs_test011/qyh/dataset/data/gsm8k_fixed.jsonl"  # 你的本地数据集

TAU_THRESHOLD = 2.8140  # 刚刚用 WikiText 跑出来的 95th 分位数阈值
MAX_NEW_TOKENS = 256  # 限制最大生成长度，防止死循环


def extract_answer(text):
    """从生成的文本和 GSM8K 答案中提取最终数字进行对比"""
    match = re.search(r'####\s*(-?\d+)', text)
    if match: return match.group(1)
    # 如果没按格式输出，尝试找最后一个数字
    numbers = re.findall(r'-?\d+', text)
    return numbers[-1] if numbers else None


def run_gsm8k_experiment():
    print("=" * 60)
    print("🚀 正在初始化动态路由评测系统 (GSM8K Benchmark)")
    print("=" * 60)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # LLaMA-3 的特殊结束符
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # 2. 加载全量模型 (Dense)
    print("📥 加载兜底模型 (Dense LLaMA-3-8B)...")
    full_adapter, _ = hf_utils.get_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct",
                                                       model_path=BASE_MODEL_PATH)
    full_model = full_adapter.model.to(DEVICE).eval()

    # 3. 加载稀疏模型 (Sparse 15%)
    print("📥 加载主力模型 (15% Sliced LLaMA-3)...")
    sparse_adapter, _ = hf_utils.load_sliced_model("meta-llama/Meta-Llama-3-8B-Instruct", SLICED_MODEL_PATH,
                                                   sparsity=0.15)
    sparse_model = sparse_adapter.model.to(DEVICE).eval()

    # 4. 加载本地数据集
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # 评测统计指标
    correct_count = 0
    total_count = 0
    total_generated_tokens = 0
    total_fallback_tokens = 0

    print("\n🔥 开始端到端测试...")
    # 为了快速验证，先切片跑前 100 题，确认没问题可以把 [:100] 删掉跑全量
    for item in tqdm(dataset[:100], desc="Evaluating GSM8K"):
        question = item['question']
        ground_truth = extract_answer(item['answer'])

        # 构建 LLaMA-3 的 Instruct Prompt
        prompt = f"Solve the following math problem and output the final answer after '####'.\nProblem: {question}\nAnswer: "
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(DEVICE)

        generated_ids = input_ids
        fallback_count_this_turn = 0
        gen_count_this_turn = 0

        # 动态生成循环 (你的核心逻辑套用在这里)
        for _ in range(MAX_NEW_TOKENS):
            with torch.no_grad():
                # 默认走稀疏模型节省算力
                outputs = sparse_model(generated_ids)
                logits = outputs.logits[:, -1, :].float()
                entropy = torch.distributions.Categorical(logits=logits).entropy().item()

                # 动态回退判断
                if entropy > TAU_THRESHOLD:
                    # 算力卸载失败，遇到复杂逻辑，呼叫全量模型兜底
                    outputs = full_model(generated_ids)
                    logits = outputs.logits[:, -1, :].float()
                    fallback_count_this_turn += 1

                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                gen_count_this_turn += 1

                # 提前结束生成
                if next_token.item() == tokenizer.eos_token_id or next_token.item() == eot_id:
                    break

        # 统计全局 Token 算力卸载率
        total_generated_tokens += gen_count_this_turn
        total_fallback_tokens += fallback_count_this_turn

        # 结果校验
        generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        model_answer = extract_answer(generated_text)

        if model_answer == ground_truth:
            correct_count += 1
        total_count += 1

    # 5. 打印最终论文指标
    accuracy = (correct_count / total_count) * 100
    offload_ratio = ((total_generated_tokens - total_fallback_tokens) / total_generated_tokens) * 100

    print("\n" + "=" * 60)
    print(" 🏆 实验结果报告 (Results)")
    print("=" * 60)
    print(f"评测题目总数: {total_count}")
    print(f"精确匹配准确率 (Exact Match Accuracy): {accuracy:.2f}%")
    print(f"Token 算力卸载率 (Offloading Ratio):  {offload_ratio:.2f}% (越高越好，代表多少比例没调动 Dense 模型)")
    print(f"回退触发频次: {total_fallback_tokens} / {total_generated_tokens}")
    print("=" * 60)


if __name__ == "__main__":
    run_gsm8k_experiment()