import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import math
import sys
import os

# 确保能导入 slicegpt 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils


def run_tau_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    model_name = "facebook/opt-125m"
    model_path = "/clzs_test011/qyh/models/opt-125m"
    sliced_model_path = "experiments/sliced_opt_125m"
    sparsity = 0.5

    # 1. 提前加载模型（只加载一次，节省大量等待时间）
    print("正在加载模型到显存，请稍候...")
    full_adapter, tokenizer = hf_utils.get_model_and_tokenizer(model_name, model_path=model_path)
    full_model = full_adapter.model
    if device.type == "cpu":
        full_model = full_model.float()
    full_model = full_model.to(device).eval()

    sliced_adapter, _ = hf_utils.load_sliced_model(model_name, sliced_model_path, sparsity=sparsity)
    sliced_model = sliced_adapter.model
    if device.type == "cpu":
        sliced_model = sliced_model.float()
    sliced_model = sliced_model.to(device).eval()

    print("✅ 模型加载完毕！\n")

    # 测试参数设置
    prompt = "The future of artificial intelligence is"
    max_new_tokens = 50  # 为了让统计数据更稳定，我们把生成的长度从 20 增加到 50

    # 我们要自动化测试的阈值列表
    tau_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    print(f"测试 Prompt: '{prompt}'")
    print(f"生成长度: {max_new_tokens} tokens\n")
    print("=" * 50)
    print(f"{'阈值(Tau)':<10} | {'触发回退次数':<15} | {'节省计算量(%)':<15}")
    print("-" * 50)

    # 核心测试循环
    for tau in tau_values:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        fallback_count = 0

        for i in range(max_new_tokens):
            with torch.no_grad():
                outputs = sliced_model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]

                logits_fp32 = next_token_logits.float()
                entropy = torch.distributions.Categorical(logits=logits_fp32).entropy().item()

                # 触发回退
                if math.isnan(entropy) or entropy > tau:
                    fallback_count += 1
                    outputs_full = full_model(input_ids)
                    next_token_logits = outputs_full.logits[:, -1, :]

                # 拼接 token
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

        # 计算比例并打印到表格中
        saved_ratio = (max_new_tokens - fallback_count) / max_new_tokens * 100
        print(f"{tau:<12.1f} | {fallback_count:<20} | {saved_ratio:<15.1f}%")

    print("=" * 50)
    print("🎉 批量测试完成！你可以用这组数据去画图表了！")


if __name__ == "__main__":
    run_tau_search()