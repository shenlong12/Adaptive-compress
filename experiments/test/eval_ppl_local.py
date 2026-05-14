import os
import torch
import math
import pandas as pd
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM

# 引入项目 src 路径
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from slicegpt import hf_utils


def run_ppl_eval(sparsity, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    DIR_PATH = "/clzs_test011/qyh/slice_new/TransformerCompression-main/TransformerCompression-main/experiments"

    # 🎯 === 路径配置区 === 🎯
    DATASET_PATH = "/clzs_test011/qyh/dataset/validation-00000-of-00001.parquet"
    # ==========================

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

    print(f"\n[加载 LLaMA-3 满血与 {sparsity * 100}% 稀疏模型...]")
    full_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)

    model_dir = os.path.join(DIR_PATH, f"llama3_8b_{int(sparsity * 100)}")
    mid_adapter, _ = hf_utils.load_sliced_model("meta-llama/Meta-Llama-3-8B-Instruct", model_dir, sparsity=sparsity)
    mid_model = mid_adapter.model.to(torch.bfloat16).to(device)

    # 加载本地 Parquet 数据
    print(f"📦 读取本地数据集: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    # 将所有文本拼接在一起 (WikiText 标准处理方式)
    full_text = "\n\n".join(df['text'].tolist())
    encodings = tokenizer(full_text, return_tensors="pt")

    max_length = 2048  # Llama3 的评估窗口
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    fallback_tokens = 0
    total_tokens = 0

    print(f"🚀 PPL 评估启动 | 阈值: {threshold} | 序列总长度: {seq_len}")

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - (begin_loc if begin_loc == 0 else begin_loc + stride)  # 避免重复计算

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            # SADS 动态决策
            mid_outputs = mid_model(input_ids)
            mid_logits = mid_outputs.logits

            # 计算最后窗口的熵 (用最后一个 token 的熵作为启发式判断)
            probs = torch.softmax(mid_logits[0, -1, :].float(), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            logits = mid_logits
            is_fallback = False
            if entropy > threshold:
                is_fallback = True
                fallback_tokens += 1
                logits = full_model(input_ids).logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss * trg_len)
            total_tokens += trg_len

        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    fallback_rate = (fallback_tokens / (seq_len // stride)) * 100

    print(f"\n✅ 评估完成!")
    print(f"📊 PPL: {ppl.item():.4f}")
    print(f"🛡️ Dense Routing (Fallback): {fallback_rate:.2f}%")
    return ppl.item(), fallback_rate


if __name__ == "__main__":
    # 你可以手动改这里，跑 0.2, 0.4 或 0.6
    run_ppl_eval(sparsity=0.40, threshold=4.0)