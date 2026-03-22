import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os


# ==========================================
# 1. 三级智能变速箱 (3-Class Router)
# ==========================================
class AdaptiveRouterMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super(AdaptiveRouterMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        # 🎯 核心改变：输出层变为 3 个神经元，对应 3 个档位
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        sentence_rep = embedded.mean(dim=1)
        x = self.fc1(sentence_rep)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


# ==========================================
# 2. 数据集加载
# ==========================================
class AdaptiveRouterDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=64):
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['prompt']
        label = int(item['label'])

        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(), torch.tensor(label, dtype=torch.long)


# ==========================================
# 3. 核心训练主循环
# ==========================================
def train_adaptive_router():
    model_name = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_file = "experiments/llama3_adaptive_router_data.jsonl"
    dataset = AdaptiveRouterDataset(data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveRouterMLP(vocab_size=len(tokenizer)).to(device)

    # ⚖️ 【极其暴力的惩罚权重】应对 2 : 15 : 383
    # 权重计算逻辑：总数 / (类别数 * 该类样本数)
    # 类别 0 的权重必须极大，迫使模型去关注它
    weights = torch.tensor([400.0 / 2, 400.0 / 15, 400.0 / 383]).float().to(device)
    print(f"⚖️ 注入对抗不平衡的惩罚权重: {weights.tolist()}")

    # 使用交叉熵损失函数，并带入惩罚权重
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print("🚀 开始训练自适应三档变速箱...")
    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for batch_texts, batch_labels in dataloader:
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_texts)

            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch_labels).sum().item()

        acc = correct / len(dataset)
        print(f"Epoch {epoch + 1:02d}/{epochs} | Loss: {total_loss / len(dataloader):.4f} | Accuracy: {acc * 100:.2f}%")

    save_path = "experiments/llama3_adaptive_router.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 训练完成！三档智能大脑已保存至: {save_path}")


if __name__ == "__main__":
    train_adaptive_router()