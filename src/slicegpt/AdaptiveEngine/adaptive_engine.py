import torch
import math

import torch.nn as nn


class RouterMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super(RouterMLP, self).__init__()
        # 极度轻量：只有一个词嵌入层和两个线性层，推理开销近乎为 0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        # 🎯 输出 3 个维度，对应 3 个切片档位
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        sentence_rep = embedded.mean(dim=1)
        x = self.fc1(sentence_rep)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


class LightweightRouter:
    def __init__(self, model_path, tokenizer, device):
        self.device = device
        self.tokenizer = tokenizer
        # 动态适配 LLaMA-3 的词表大小
        self.model = RouterMLP(vocab_size=len(tokenizer)).to(device)

        # 加载你训练好的路由大脑权重
        if torch.cuda.is_available() and model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print("🧠 成功加载自适应轻量级识别接口！")
        else:
            print("⚠️ 尚未找到训练好的权重，当前接口为空白状态。")

        self.model.eval()

    def predict_tier(self, prompt):
        """
        这就是你构想的识别入口！
        一秒看穿数据集类型，返回最优档位 (0=极简, 1=平衡, 2=地狱)
        """
        encoding = self.tokenizer(
            prompt, truncation=True, max_length=64,
            padding='max_length', return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids)
            tier = torch.argmax(logits, dim=-1).item()
        return tier


import torch
import math


class ReactiveAdaptiveEngine:
    def __init__(self, full_model, sliced_model, tokenizer, device, tau_threshold=6.5):
        """
        纯运行时监控引擎 (Cascade Inference)
        :param full_model: 满血大模型 (用于兜底救场)
        :param sliced_model: 默认切片模型 (推荐 25% 稀疏的 s_mid，兼顾能力与算力)
        :param tau_threshold: 熵值报警红线 (25% 稀疏度推荐设为 6.5)
        """
        self.full_model = full_model
        self.sliced_model = sliced_model
        self.tokenizer = tokenizer
        self.device = device
        self.tau_threshold = tau_threshold

    def generate(self, prompt, max_length=150):
        """
        对外暴露的统一生成接口：默认小马拉车，异常瞬间大马接管
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids = input_ids
        input_len = input_ids.shape[1]

        # 初始状态：让切片模型先上
        current_model = self.sliced_model
        has_fallen_back = False
        fallback_step = -1  # 记录在第几个 Token 扛不住的

        for i in range(max_length):
            with torch.no_grad():
                outputs = current_model(generated_ids)
                logits = outputs.logits[:, -1, :].float()

                # 只有在还没回退的时候，才需要算熵值监控小模型
                if not has_fallen_back:
                    entropy = torch.distributions.Categorical(logits=logits).entropy().item()

                    # 🚨 核心触发器：一旦脑电波异常，永久拉起满血模型！
                    if math.isnan(entropy) or entropy > self.tau_threshold:
                        has_fallen_back = True
                        fallback_step = i
                        current_model = self.full_model

                        # 满血模型立刻接管当前的预测
                        outputs = current_model(generated_ids)
                        logits = outputs.logits[:, -1, :].float()

                # 取出预测词并拼接
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # 遇到结束符提前停止
                if next_token.item() == self.tokenizer.eos_token_id or next_token.item() == self.tokenizer.convert_tokens_to_ids(
                        "<|eot_id|>"):
                    break

        new_tokens = generated_ids[0][input_len:]
        final_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "text": final_text.strip(),
            "triggered_fallback": has_fallen_back,
            "fallback_at_token": fallback_step,  # 这个数据写论文分析时极度有用！
            "total_tokens_generated": len(new_tokens)
        }