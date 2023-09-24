import torch.nn as nn
import torch

class NNLM(nn.Module):
    def __init__(self, Emb, hidden_dim, dropout, device):
        super(NNLM, self).__init__()

        self.Emb = Emb
        self.input_size = Emb.vector_size * 5
        vocab_size = len(Emb.key_to_index)
        self.device = device

        self.model = torch.nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.Emb.vectors[X]
        X = torch.tensor(X).to(self.device)
        X = X.view(-1, self.input_size)
        X = self.dropout(X)
        return self.model(X)
