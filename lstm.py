import torch.nn as nn
import torch

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, func):
        super().__init__()

        self.func = func
        self.X = nn.Linear(input_size, hidden_size)
        self.H = nn.Linear(hidden_size, hidden_size)

    def forward(self, X, H):
        return self.func(self.X(X) + self.H(H))

class LSTM(nn.Module):
    def __init__(self, Emb, hidden_size, dropout, device):
        super().__init__()

        self.Emb = Emb
        self.hidden_size = hidden_size
        self.device = device

        input_size = Emb.vector_size
        vocab_size = len(Emb.key_to_index)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init_hidden()

        self.forget = LSTMCell(input_size, hidden_size, self.sigmoid)          # Forget gate
        self.input_sigmoid = LSTMCell(input_size, hidden_size, self.sigmoid)   # Input gate (sigmoid)
        self.input_tanh = LSTMCell(input_size, hidden_size, self.tanh)         # Input gate (tanh)
        self.output = LSTMCell(input_size, hidden_size, self.sigmoid)          # Output gate
        self.logits = nn.Linear(hidden_size, vocab_size)                       # Output layer

        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        # Forget gate
        X = torch.tensor(self.Emb.vectors[X]).to(self.device)
        X = self.dropout(X)

        self.C = self.forget(X, self.H) * self.C

        # Input gate
        self.C += self.input_sigmoid(X, self.H) * self.input_tanh(X, self.H)

        # Output gate
        self.H = self.output(X, self.H) * self.tanh(self.C)

        return self.logits(self.H)

    def init_hidden(self):
        self.C = torch.zeros(self.hidden_size).to(self.device)
        self.H = torch.zeros(self.hidden_size).to(self.device)
