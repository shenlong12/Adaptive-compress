import os
import csv
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 🛡️ 强制离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


# ==========================================
# 1. 模型定义 (保持与推理侧完全一致)
# ==========================================
class RouterMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, num_tiers=4):
        super(RouterMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_tiers)

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        embedded = self.embedding(input_ids)
        sum_embeddings = (embedded * mask.unsqueeze(-1)).sum(dim=1)
        valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        sentence_rep = sum_embeddings / valid_lengths

        x = self.fc1(sentence_rep)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits


# ==========================================
# 2. 数据集加载与处理工具
# ==========================================
class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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


def load_gsm8k(path, max_num=2000):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            data.append(d['question'])
            if len(data) >= max_num: break
    return data


def load_local_mmlu(data_dir, max_num=2000):
    data = []
    target_dir = os.path.join(data_dir, "test") if os.path.exists(os.path.join(data_dir, "test")) else data_dir
    if not os.path.exists(target_dir):
        print(f"⚠️ 找不到 MMLU 目录: {target_dir}")
        return data

    csv_files = [f for f in os.listdir(target_dir) if f.endswith('.csv')]
    for file in csv_files:
        with open(os.path.join(target_dir, file), 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    prompt = (
                        f"Question: {row[0]}\n"
                        f"A. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}\n"
                        "Please answer with only the letter A, B, C, or D."
                    )
                    data.append(prompt)
                    if len(data) >= max_num: return data
    return data


def load_local_wiki(path, max_num=2000):
    """加载本地 Wiki 数据集，完美兼容 Parquet/JSONL/TXT"""
    data = []
    if not os.path.exists(path):
        print(f"⚠️ 找不到 Wiki 文件: {path} (请检查路径)")
        return data

    # 🚀 专门针对你刚下载的 Parquet 二进制文件进行解析
    if path.endswith('.parquet') or 'test-00000' in path:
        try:
            import pandas as pd
            print("📦 检测到 Parquet 格式，正在使用 pandas 解析...")
            df = pd.read_parquet(path)
            # Hugging Face 的 WikiText 通常保存在 'text' 这一列
            for text in df['text'].dropna():
                text = str(text).strip()
                if len(text) > 100:  # 过滤掉太短的无意义标题
                    data.append(text[:600])  # 截取适中长度，避免过长
                if len(data) >= max_num: break
            return data
        except ImportError:
            print("🔴 缺少 pandas 库！请在终端运行: pip install pandas pyarrow")
            return data
        except Exception as e:
            print(f"🔴 读取 Parquet 出错: {e}")
            return data

    # 👇 下面是原本处理 txt/jsonl 纯文本的逻辑
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith('.jsonl') or path.endswith('.json'):
                try:
                    d = json.loads(line)
                    text = d.get('text', d.get('prompt', ''))
                    if text and len(text) > 100:
                        data.append(text[:600])
                except:
                    pass
            else:
                if len(line) > 100:
                    data.append(line[:600])

            if len(data) >= max_num: break
    return data

# ==========================================
# 3. 核心：构建带 Llama-3 模板的真实训练集
# ==========================================
def get_real_training_data(tokenizer, samples_per_class=1000):
    texts = []
    labels = []

    print("📥 正在组装真实训练数据...")

    # 🔴 Tier 2 (Label 2): GSM8K 数学题
    gsm8k_qs = load_gsm8k("/clzs_test011/qyh/dataset/gsm8k_fixed.jsonl", max_num=samples_per_class)
    for q in gsm8k_qs:
        msg = [{"role": "system",
                "content": "You are a math expert. Solve step by step. Conclude with 'The final answer is [number]'."},
               {"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(prompt)
        labels.append(2)

    # 🟡 Tier 1 (Label 1): MMLU 综合问答
    mmlu_qs = load_local_mmlu("/clzs_test011/qyh/dataset/data", max_num=samples_per_class)
    for q in mmlu_qs:
        msg = [{"role": "system", "content": "You are a knowledgeable assistant."},
               {"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(prompt)
        labels.append(1)

    # 🟢 Tier 0 (Label 0): 本地 Wiki 长文
    # ⚠️ 极其关键：请把你本地的 Wiki 数据集真实路径填在下面！
    WIKI_PATH = "/clzs_test011/qyh/dataset/wikitext"
    wiki_qs = load_local_wiki(WIKI_PATH, max_num=samples_per_class)
    for q in wiki_qs:
        msg = [{"role": "system",
                "content": "You are a detailed assistant. Please analyze or continue the following text."},
               {"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        texts.append(prompt)
        labels.append(0)

    # 随机打乱数据，防止模型偏科
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    return list(texts), list(labels)


# ==========================================
# 4. 训练循环
# ==========================================
def train_router(model_name_or_path):
    print("🚀 正在初始化雷达训练引擎...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 准备真实数据
    texts, labels = get_real_training_data(tokenizer, samples_per_class=1000)

    # 扩大 max_length 到 128，确保能罩住加了模板的长 Prompt
    dataset = PromptDataset(texts, labels, tokenizer, max_length=128)
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
    DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(DIR_PATH, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    save_path = os.path.join(weights_dir, "router_mlp_4tiers.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ 训练完成！雷达权重已保存至: {save_path}")


if __name__ == "__main__":
    model_path = "/clzs_test011/qyh/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    train_router(model_path)