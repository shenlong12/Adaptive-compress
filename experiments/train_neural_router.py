import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os


# ==========================================
# 1. 定义极其轻量的神经路由器模型 (Neural Router)
# ==========================================
class RouterMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super(RouterMLP, self).__init__()
        # 词嵌入层：把文本变成向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        # 核心决策大脑：一个简单的两层全连接网络
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 防止过拟合
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()  # 输出 0~1 的概率

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        # 把一句话所有的词向量平均一下，代表整句话的意思 (Mean Pooling)
        sentence_rep = embedded.mean(dim=1)  # [batch_size, embed_dim]

        x = self.fc1(sentence_rep)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze()  # 输出是个概率值


# ==========================================
# 2. 构建 PyTorch 数据集
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

        # 将文本转化为数字 ID
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return encoding['input_ids'].squeeze(), torch.tensor(label)


# ==========================================
# 3. 核心训练主循环
# ==========================================
def train_router():
    # 我们复用 OPT 的 Tokenizer，这样不用下载新的词表
    model_name = "/clzs_test011/qyh/opt-6.7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 构建数据集和 DataLoader
    data_file = "experiments/router_training_data.jsonl"
    dataset = RouterDataset(data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouterMLP(vocab_size=tokenizer.vocab_size).to(device)

    # 【学术硬核点】：计算类别权重，解决 347:53 的数据不平衡
    num_ones = 347.0
    num_zeros = 53.0
    pos_weight = torch.tensor([num_zeros / num_ones]).to(device)  # 给 0 标签更高的重视

    # 使用带权重的二元交叉熵损失函数 (BCEWithLogitsLoss 的替代方案)
    criterion = nn.BCELoss(weight=None)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    print("🚀 开始训练神经路由器 (Neural Router)...")
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

            # 手动应用类别权重
            loss = criterion(predictions, batch_labels)
            weight = torch.where(batch_labels == 1, 1.0, num_ones / num_zeros)
            loss = (loss * weight).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算准确率 (大于 0.5 判定为 1)
            preds_binary = (predictions >= 0.5).float()
            correct += (preds_binary == batch_labels).sum().item()

        acc = correct / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader):.4f} | Accuracy: {acc * 100:.2f}%")

    # 保存训练好的轻量级模型
    save_path = "experiments/neural_router.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 训练完成！Router 模型已保存至: {save_path}")


if __name__ == "__main__":
    train_router()