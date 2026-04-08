import torch
import time
from transformers import AutoModelForCausalLM


def run_latency_benchmark():
    # 使用你服务器上的满血模型绝对路径
    model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    device = "cuda"

    print("🚀 [1/3] 正在加载满血 LLaMA-3-8B 进显存...")
    # 只加载一个满血大模型即可，我们通过张量切片来模拟小模型的 KV Cache
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    # 测试不同的上下文长度 (Context Length)
    context_lengths = [128, 512, 1024, 2048, 4096]

    print("\n📊 [2/3] 开始执行端到端系统级延迟基准测试 (Latency Benchmark)\n")
    print(
        f"{'序列长度 (L)':<15} | {'传统重算 Recompute (ms)':<25} | {'显存补零 Zero-Padding (ms)':<25} | {'加速比 Speedup'}")
    print("-" * 90)

    for L in context_lengths:
        # 伪造一个长度为 L 的输入序列 (模拟前面的长文阅读或长推导)
        input_ids = torch.randint(0, 32000, (1, L)).to(device)
        last_token = input_ids[:, -1:]  # 触发回退时的那 1 个 Token

        # =========================================================
        # 方法 A: 传统方法的灾难 (Fallback with Recomputation)
        # =========================================================
        torch.cuda.synchronize()
        start_t = time.time()
        with torch.no_grad():
            # 大模型被迫重新消化前面所有的 L 个 Token，复杂度 O(L^2)
            _ = model(input_ids)
        torch.cuda.synchronize()
        recompute_time = (time.time() - start_t) * 1000

        # =========================================================
        # 准备工作：在显存中模拟 10% 稀疏小模型生成的 KV Cache
        # =========================================================
        with torch.no_grad():
            outputs = model(input_ids[:, :-1], use_cache=True)
            dense_kv = outputs.past_key_values

        # 模拟小模型的特征维度被砍掉了 (比如被 SliceGPT 剪裁了)
        sparse_kv = []
        for k, v in dense_kv:
            # 截断最后的维度，模拟稀疏小模型吐出的、残缺的 KV Cache
            sparse_kv.append((k[:, :, :, :-16], v[:, :, :, :-16]))

        # =========================================================
        # 方法 B: 我们的 SADS 极速救场 (KV Cache Zero-Padding)
        # =========================================================
        torch.cuda.synchronize()
        start_t2 = time.time()
        with torch.no_grad():
            padded_kv = []
            # 1. 直接在显存层面用 0 补齐被砍掉的维度 (纯访存操作，耗时极短)
            for k_sparse, v_sparse in sparse_kv:
                zeros_k = torch.zeros_like(k_sparse[:, :, :, :16])
                zeros_v = torch.zeros_like(v_sparse[:, :, :, :16])
                k_padded = torch.cat([k_sparse, zeros_k], dim=-1)
                v_padded = torch.sparse = torch.cat([v_sparse, zeros_v], dim=-1)
                padded_kv.append((k_padded, v_padded))

            # 2. 大模型直接接收补齐的 KV Cache，只需计算最后这 1 个 Token！复杂度 O(1)
            _ = model(last_token, past_key_values=tuple(padded_kv), use_cache=True)

        torch.cuda.synchronize()
        padding_time = (time.time() - start_t2) * 1000

        # 计算并打印这一轮的加速比
        speedup = recompute_time / padding_time
        print(f"{L:<18} | {recompute_time:<25.2f} | {padding_time:<25.2f} | {speedup:<10.1f}x")


if __name__ == "__main__":
    run_latency_benchmark()