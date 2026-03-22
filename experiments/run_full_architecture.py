import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import math, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicegpt import hf_utils

# --- 模块 ①：语义路由器 Classifier R(x) ---
class SemanticRouter:
    def __init__(self, threshold_len=15):
        self.threshold_len = threshold_len # 简单演示：根据输入长度判断复杂度

    def decide(self, input_ids):
        # 实际论文中这里可以是一个轻量级分类器，现在我们用序列长度作为复杂度指标
        seq_len = input_ids.shape[1]
        if seq_len < self.threshold_len:
            print(f"Decision: Low Complexity (s_low) -> Start with 50% Sparsity")
            return "low"
        else:
            print(f"Decision: High Complexity (s_mid) -> Start with 25% Sparsity")
            return "mid"

def run_full_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 【修改点 1】统一使用 facebook/opt-125m，确保文件名匹配
    model_name = "facebook/opt-125m"
    model_path = "/clzs_test011/qyh/models/opt-125m"
    tau = 3.5

    # 1. 加载三个级别的模型 (100%, 75%, 50%)
    print("正在加载全量模型 (100%)...")
    full_adapter, tokenizer = hf_utils.get_model_and_tokenizer(model_name, model_path=model_path)
    full_model = full_adapter.model.to(device).eval()

    print("正在加载中等稀疏模型 (25% Sliced)...")
    # 【修改点 2】传入准确的 model_name
    mid_adapter, _ = hf_utils.load_sliced_model(model_name, "experiments/sliced_opt_125m_mid", sparsity=0.25)
    mid_model = mid_adapter.model.to(device).eval()

    print("正在加载高度稀疏模型 (50% Sliced)...")
    # 【修改点 3】传入准确的 model_name
    low_adapter, _ = hf_utils.load_sliced_model(model_name, "experiments/sliced_opt_125m", sparsity=0.5)
    low_model = low_adapter.model.to(device).eval()

    # 初始化路由器
    router = SemanticRouter(threshold_len=10)

    # 测试输入
    prompt = "Artificial intelligence is a branch of computer science that"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # --- 模块 ①：做出初始决策 ---
    decision = router.decide(input_ids)
    active_model = low_model if decision == "low" else mid_model

    print("-" * 60)
    for i in range(20):
        with torch.no_grad():
            # --- 模块 ②：动态执行 ---
            outputs = active_model(input_ids)
            logits = outputs.logits[:, -1, :].float()
            entropy = torch.distributions.Categorical(logits=logits).entropy().item()

            # --- 模块 ③：Fallback 机制 ---
            if entropy > tau:
                print(f"Token {i}: Entropy {entropy:.2f} > {tau} ⚠️ Fallback to 100%")
                outputs = full_model(input_ids)
                logits = outputs.logits[:, -1, :].float()
            else:
                print(f"Token {i}: Entropy {entropy:.2f} ✅ Active path used")

            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    print("-" * 60)
    print(f"最终生成: {tokenizer.decode(input_ids[0])}")

if __name__ == "__main__":
    run_full_experiment()