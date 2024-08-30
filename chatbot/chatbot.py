import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import pickle
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random

# Hyperparameters and settings
batch_size = 32
block_size = 128
learning_rate = 3e-4
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 1

# Dataset paths and URLs
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

datasets = {
    "alice_in_wonderland.txt": "https://www.gutenberg.org/files/11/11-0.txt",
    "peter_pan.txt": "https://www.gutenberg.org/files/16/16-0.txt",
    "pride_and_prejudice.txt": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "wizards_of_oz.txt": "https://www.gutenberg.org/files/55/55-0.txt"
}

# Download datasets if they don't exist
def download_data(file_name, url):
    file_path = os.path.join(data_dir, file_name)
    if not os.path.isfile(file_path):
        print(f"{file_name} not found. Downloading...")
        response = requests.get(url)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"{file_name} downloaded successfully.")
    return file_path

file_paths = [download_data(file_name, url) for file_name, url in datasets.items()]

# Load datasets and preprocess
def load_data(file_paths):
    text = ""
    chars = set()
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            text += content
            chars.update(set(content))
    
    chars = sorted(chars)
    vocab_size = len(chars)
    string_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_string = {i: ch for i, ch in enumerate(chars)}
    return text, string_to_int, int_to_string, vocab_size

text, string_to_int, int_to_string, vocab_size = load_data(file_paths)
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# Prepare the dataset
class CharDataset(Dataset):
    def __init__(self, text, block_size):
        self.data = encode(text)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

train_dataset = CharDataset(text, block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define model components
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:x.size(1), :x.size(1)] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        return self.ln2(x + self.ffwd(x))

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.ln_f(x)
        logits = self.lm_head(logits)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        return logits, None

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# Training function
def train_model(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs} completed, Avg Loss: {total_loss / len(train_loader):.4f}")

# Load model
model = GPTLanguageModel(vocab_size).to(device)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
model_path = 'model-01.pkl'
if os.path.isfile(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f).to(device)

#train_model(model, train_loader, optimizer, epochs)

# Chatbot interaction
def chatbot_interaction():
    print("Welcome to the baby Chatbot! Type your message below and see the magic unfold. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        input_indices = torch.tensor(encode(user_input), dtype=torch.long, device=device).unsqueeze(0)
        output_indices = model.generate(input_indices, max_new_tokens=150)
        response = decode(output_indices[0].tolist())
        print(f"Bot: {response}")

# Start the chatbot
chatbot_interaction()
