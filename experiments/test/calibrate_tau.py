import torch
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import sys, os

# 借用你之前的 slicegpt 加载库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils


def collect_entropy_distribution(base_model_path, sliced_model_path, sparsity, calibration_prompts, device):
    print(f"\n🔬 正在拉起 {int(sparsity * 100)}% 稀疏模型")
    # 纯本地加载切片模型
    # 第一个参数恢复成架构名字，第二个参数用你的本地路径，彻底解决不识别的问题
    adapter, _ = hf_utils.load_sliced_model("meta-llama/Meta-Llama-3-8B-Instruct",
                                            sliced_model_path,
                                            sparsity=sparsity)
    model = adapter.model.to(device).eval()

    # 纯本地加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    all_entropies = []

    for prompt in tqdm(calibration_prompts, desc="采集 Token 熵值"):
        msg = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)

        generated_ids = input_ids
        # 截取前 100 个生成的 Token 作为稳定期的心电图特征
        for _ in range(100):
            with torch.no_grad():
                outputs = model(generated_ids)
                logits = outputs.logits[:, -1, :].float()

                # 计算香农熵
                entropy = torch.distributions.Categorical(logits=logits).entropy().item()
                if not math.isnan(entropy):
                    all_entropies.append(entropy)

                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.convert_tokens_to_ids(
                        "<|eot_id|>"):
                    break

    # 释放显存
    del adapter
    del model
    torch.cuda.empty_cache()

    return np.array(all_entropies)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 你的纯本地基座模型路径 (绝对核心，不连外网的保证)
    base_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"

    # 准备校准集 (用模型最擅长的极简百科或常识，测出它的“健康基线”)
    # ==========================================
    # 从本地 WikiText 数据集中随机抽取真实校准样本
    # ==========================================
    import random


    def load_wikitext_prompts(file_path, num_samples=100):
        print(f"📖 正在从 {file_path} 加载校准数据...")
        with open(file_path, 'r', encoding='utf-8') as f:
            # 过滤掉太短的行（比如单独的标题）或者空行
            lines = [line.strip() for line in f.readlines() if len(line.strip()) > 50]

        # 固定随机种子保证每次校准的样本一致，具有可复现性
        random.seed(42)
        # 随机抽取 100 段作为 Prompt
        sampled_lines = random.sample(lines, min(num_samples, len(lines)))
        return sampled_lines


    # 替换成你服务器上 wiki.test 的绝对路径
    wiki_path = "/clzs_test011/qyh/dataset/wiki.test"
    calibration_set = load_wikitext_prompts(wiki_path, num_samples=50)
    print(f"✅ 成功提取 {len(calibration_set)} 条 WikiText 样本用于校准。\n")
    # ==========================================
    # 1. 扫描 Tier 1 (15% 稀疏)
    # ==========================================
    tier1_path = "experiments/sliced_llama3_8b_15"
    entropies_t1 = collect_entropy_distribution(base_path, tier1_path, 0.15, calibration_set, device)

    tau_t1 = np.percentile(entropies_t1, 95)
    mean_t1 = np.mean(entropies_t1)

    print("\n" + "=" * 50)
    print(" 📊 Tier 1 (15% 稀疏) 阈值校准报告")
    print("=" * 50)
    print(f"共收集 Token 样本数: {len(entropies_t1)}")
    print(f"平均熵值 : {mean_t1:.4f}")
    print(f"🎯 科学推荐阈值 (95th Percentile tau): {tau_t1:.4f}")

    # ==========================================
    # 2. 扫描 Tier 2 (10% 稀疏)
    # ==========================================
    tier2_path = "experiments/sliced_llama3_8b_10"
    entropies_t2 = collect_entropy_distribution(base_path, tier2_path, 0.10, calibration_set, device)

    tau_t2 = np.percentile(entropies_t2, 95)
    mean_t2 = np.mean(entropies_t2)

    print("\n" + "=" * 50)
    print(" 📊 Tier 2 (10% 稀疏) 阈值校准报告")
    print("=" * 50)
    print(f"共收集 Token 样本数: {len(entropies_t2)}")
    print(f"平均熵值: {mean_t2:.4f}")
    print(f"🎯 科学推荐阈值 (95th Percentile tau): {tau_t2:.4f}")