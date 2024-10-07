import torch
import torch.nn as nn

class SentimentPredictor(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dim)
        self.linear_layer = nn.Linear(embedding_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        embeddings = self.embedding_layer(x)
        average = torch.mean(embeddings, axis=1)
        projected = self.linear_layer(average)

        return self.tanh(projected)
