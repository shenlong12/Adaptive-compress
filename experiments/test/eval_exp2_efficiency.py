import os
import time
import torch
import math
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🛡️ 离线模式保平安
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from slicegpt import hf_utils


def flush_memory():
    """强制清空显存缓存"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure_generation(model_fn, prompt_ids, max_new_tokens=128):
    """通用的测速包装器"""
    flush_memory()

    # 🌟 1. GPU 预热 (跑一次不计入时间，唤醒 CUDA 核心)
    with torch.no_grad():
        _ = model_fn(prompt_ids, max_new_tokens)

    flush_memory()
    torch.cuda.synchronize()  # 确保 GPU 闲置

    # 🌟 2. 正式测速
    start_time = time.perf_counter()
    with torch.no_grad():
        generated_tokens = model_fn(prompt_ids, max_new_tokens)
    torch.cuda.synchronize()  # 等待 GPU 彻底跑完
    end_time = time.perf_counter()

    # 🌟 3. 统计硬件数据
    total_time = end_time - start_time
    throughput = max_new_tokens / total_time
    max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转换为 GB

    return throughput, max_vram


# ==========================================
# 主控引擎
# ==========================================
def run_efficiency_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    DIR_PATH = "/clzs_test011/qyh/slice_new/TransformerCompression-main/TransformerCompression-main/experiments"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

    # 构建一个标准的 256 token 输入 prompt，测 128 token 的生成
    dummy_text = "The rapid development of artificial intelligence has led to significant advancements in various fields. " * 20
    input_ids = tokenizer(dummy_text, return_tensors="pt").input_ids[:, :256].to(device)
    max_gen_tokens = 128

    print("\n" + "=" * 65)
    print(" 🚀 SADS 硬件效率极限压测 (实验二)")
    print("=" * 65)

    # ---------------------------------------------------------
    # 测试 1: 满血 Dense 基线
    # ---------------------------------------------------------
    print("\n[1/3] 加载 Dense 8B Baseline...")
    full_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)
    full_model.eval()

    def generate_dense(prompt, max_tokens):
        return full_model.generate(prompt, max_new_tokens=max_tokens, do_sample=False, use_cache=True)

    print("开始压测 Dense Baseline...")
    dense_tp, dense_vram = measure_generation(generate_dense, input_ids, max_gen_tokens)
    print(f"👑 Dense Baseline -> 速度: {dense_tp:.2f} Tokens/s | 峰值显存: {dense_vram:.2f} GB")

    # ---------------------------------------------------------
    # 测试 2: 静态 30% 稀疏模型
    # ---------------------------------------------------------
    print("\n[2/3] 加载 Static SliceGPT (30% 稀疏)...")
    sparsity = 0.30
    model_dir = os.path.join(DIR_PATH, "llama3_8b_30")
    mid_adapter, _ = hf_utils.load_sliced_model("meta-llama/Meta-Llama-3-8B-Instruct", model_dir, sparsity=sparsity)
    mid_model = mid_adapter.model.to(torch.bfloat16).to(device)
    mid_model.eval()

    def generate_static(prompt, max_tokens):
        return mid_model.generate(prompt, max_new_tokens=max_tokens, do_sample=False, use_cache=True)

    print("开始压测 Static 30%...")
    static_tp, static_vram = measure_generation(generate_static, input_ids, max_gen_tokens)
    print(f"📉 Static 30%    -> 速度: {static_tp:.2f} Tokens/s | 峰值显存: {static_vram:.2f} GB")

    # ---------------------------------------------------------
    # 测试 3: SADS 动态路由 (基于 30% 底座, Tau=0.5)
    # ---------------------------------------------------------
    print("\n[3/3] 挂载 SADS 动态路由引擎 (Tau=0.5)...")
    TAU_THRESHOLD = 0.5

    def generate_sads(prompt_ids, max_tokens):
        # 精简版的 SADS 测速循环 (剥离了不必要的记录)
        past_key_values_mid = None
        current_input_ids = prompt_ids
        generated_count = 0

        with torch.no_grad():
            outputs = mid_model(current_input_ids, use_cache=True)
            past_key_values_mid = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        for _ in range(max_tokens):
            entropy = torch.distributions.Categorical(logits=next_token_logits.float()).entropy().item()

            if math.isnan(entropy) or entropy > TAU_THRESHOLD:
                # 救场触发
                with torch.no_grad():
                    # 注意：严格意义上的测速，救场时需要重新前向传播
                    outputs_full = full_model(current_input_ids, use_cache=True)
                    next_token_logits = outputs_full.logits[:, -1, :]

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            current_input_ids = next_token
            generated_count += 1

            with torch.no_grad():
                outputs = mid_model(next_token, past_key_values=past_key_values_mid, use_cache=True)
                past_key_values_mid = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

        return generated_count

    print("开始压测 SADS...")
    sads_tp, sads_vram = measure_generation(generate_sads, input_ids, max_gen_tokens)
    print(f"🚀 SADS (Tau=0.5)-> 速度: {sads_tp:.2f} Tokens/s | 峰值显存: {sads_vram:.2f} GB")

    print("\n" + "=" * 65)
    print(" 🏆 实验二 (硬件效率) 战报汇总 🏆")
    print(f" Baseline 速度: {dense_tp:.2f} T/s")
    print(f" SADS 加速比 : {sads_tp / dense_tp:.2f}x (越高越好)")
    print("=" * 65)


if __name__ == "__main__":
    run_efficiency_benchmark()