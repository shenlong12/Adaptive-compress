import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os


# ==========================================
# 1. 重新定义模型 (保持与推理侧完全一致)
# ==========================================
class RouterMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_tiers=4):
        super(RouterMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_tiers)

    def forward(self, input_ids):
        # 提取 padding 掩码以忽略 pad token 的影响 (更严谨)
        mask = (input_ids != 0).float()
        embedded = self.embedding(input_ids)
        # 求均值时只计算非 pad 的 token
        sum_embeddings = (embedded * mask.unsqueeze(-1)).sum(dim=1)
        valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        sentence_rep = sum_embeddings / valid_lengths

        x = self.fc1(sentence_rep)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


# ==========================================
# 2. 构造训练数据集 (模拟 4 档难度)
# ==========================================
class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length,
            padding='max_length', return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'labels': self.labels[idx]
        }


def get_dummy_training_data():
    """
    这里模拟了四个层级的数据。
    实际写论文时，请按比例从你的 Wiki, C4, Code, GSM8K 数据集中各抽 1000 条填入。
    """
    data = [
               # Tier 0 (类闲聊/极简指令) -> Label 0
               ("Translate the following sentence to French.", 0),
               ("Say hello to my friend John.", 0),
               ("What is the weather like today?", 0),
               ("Write a short greeting email.", 0),

               # Tier 1 (类阅读理解/摘要/长文) -> Label 1
               ("Summarize the main points of this news article.", 1),
               ("Write a detailed Wikipedia-style history of the Roman Empire.", 1),
               ("Explain the process of photosynthesis in detail.", 1),
               ("What are the key benefits of structural pruning?", 1),

               # Tier 2 (类代码/复杂指令/专业推理) -> Label 2
               ("Explain the following Python code line by line.", 2),
               ("Write a script to scrape data from a website.", 2),
               ("Design a system architecture for a high-concurrency app.", 2),
               ("Debug this C++ memory leak issue.", 2),

               # Tier 3 (硬核数学/极度严密逻辑) -> Label 3
               ("Solve the math problem step by step and find x.", 3),
               ("Calculate the probability of drawing two aces from a deck.", 3),
               ("Prove the Pythagorean theorem strictly.", 3),
               ("A train leaves New York at 5 PM travelling at 60 mph...", 3),
           ] * 50  # 复制 50 遍凑够 800 条数据用于测试训练流

    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    return texts, labels


# ==========================================
# 3. 核心训练循环
# ==========================================
def train_router(model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"):
    print("🚀 正在初始化雷达训练引擎...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备数据
    texts, labels = get_dummy_training_data()
    dataset = PromptDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. 初始化极其轻量的 MLP
    model = RouterMLP(vocab_size=len(tokenizer), num_tiers=4).to(device)

    # 4. 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    num_epochs = 15
    print(f"📈 开始训练！总数据量: {len(dataset)}, 批次大小: 32")

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()

        acc = correct / len(dataset)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {total_loss / len(dataloader):.4f} | Accuracy: {acc * 100:.2f}%")

    # 5. 保存巅峰权重
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/router_mlp_4tiers.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ 训练完成！雷达权重已保存至: {save_path}")


if __name__ == "__main__":
    # 🎯 强行指向你本地的绝对路径，彻底绕开 Hugging Face 的网络拦截
    model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    train_router(model_path)