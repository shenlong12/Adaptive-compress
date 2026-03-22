import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os


# ==========================================
# 1. 路由器网络结构 (保持极轻量)
# ==========================================
class RouterMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super(RouterMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        sentence_rep = embedded.mean(dim=1)
        x = self.fc1(sentence_rep)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()


# ==========================================
# 2. 数据集加载 (适配 LLaMA-3)
# ==========================================
class RouterDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=32):
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
        label = float(item['label'])

        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(), torch.tensor(label)


# ==========================================
# 3. 核心训练主循环
# ==========================================
def train_llama3_router():
    # 🎯 LLaMA-3 的绝对路径
    model_name = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 【极其重要】：修复 LLaMA-3 缺乏 pad_token 的问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_file = "experiments/llama3_router_training_data.jsonl"
    dataset = RouterDataset(data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouterMLP(vocab_size=len(tokenizer)).to(device)  # 使用 len(tokenizer) 防止越界

    # ⚖️ 【硬核点】：应对 376 vs 24 的极端不平衡
    num_ones = 376.0
    num_zeros = 24.0
    criterion = nn.BCELoss(weight=None)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print("🚀 开始训练 LLaMA-3 专属神经路由器...")
    epochs = 15

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for batch_texts, batch_labels in dataloader:
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_texts)

            # 手动应用类别权重：极其严厉地惩罚把 0 猜成 1 的行为
            loss = criterion(predictions, batch_labels)
            weight = torch.where(batch_labels == 1, 1.0, num_ones / num_zeros)
            loss = (loss * weight).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds_binary = (predictions >= 0.5).float()
            correct += (preds_binary == batch_labels).sum().item()

        acc = correct / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader):.4f} | Accuracy: {acc * 100:.2f}%")

    save_path = "experiments/llama3_neural_router.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 训练完成！LLaMA-3 专属大脑已保存至: {save_path}")


if __name__ == "__main__":
    train_llama3_router()