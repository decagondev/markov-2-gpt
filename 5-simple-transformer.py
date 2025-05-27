# Transformer Text Generator from Scratch using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

# Sample corpus
corpus = """
Once upon a time, there was a brave princess who lived in a castle.
One day, the princess met a wise old dragon in the forest.
They became friends and had many adventures together.
""".lower()

# Tokenize
tokens = corpus.replace('\n', ' ').split()
vocab = list(set(tokens))
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Prepare sequences
sequence_length = 4
sequences = []
for i in range(len(tokens) - sequence_length):
    seq = tokens[i:i + sequence_length]
    label = tokens[i + sequence_length]
    sequences.append((seq, label))

# Dataset class
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        seq_idx = torch.tensor([word2idx[word] for word in seq], dtype=torch.long)
        label_idx = torch.tensor(word2idx[label], dtype=torch.long)
        return seq_idx, label_idx

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1])
        return x

# Load data
dataset = TextDataset(sequences)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Instantiate model
model = TransformerModel(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(50):
    for seqs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Text generation
def generate_text_transformer(start_seq, length=10):
    model.eval()
    generated = start_seq[:]
    for _ in range(length):
        seq_idx = torch.tensor([[word2idx[w] for w in generated[-sequence_length:]]], dtype=torch.long)
        with torch.no_grad():
            output = model(seq_idx)
            pred_idx = torch.argmax(output, dim=1).item()
            generated.append(idx2word[pred_idx])
    return ' '.join(generated)

# Example usage
seed = ["the", "princess", "met", "a"]
print("\nGenerated Text:")
print(generate_text_transformer(seed, 15))
