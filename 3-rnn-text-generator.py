# RNN Text Generator using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

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

# Hyperparameters
embedding_dim = 16
hidden_dim = 32
epochs = 100
batch_size = 2

# Model definition
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeds = self.embedding(x)
        out, _ = self.rnn(embeds)
        out = self.fc(out[:, -1, :])
        return out

# Load data
dataset = TextDataset(sequences)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate model
model = RNNModel(len(vocab), embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(epochs):
    for seqs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Text generation
def generate_text_rnn(start_seq, length=10):
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
print(generate_text_rnn(seed, 15))
