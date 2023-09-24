from torch.nn import TransformerDecoder, TransformerDecoderLayer
from pos_enc import PositionalEncoding
from torch import Tensor
import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, Emb, nhead: int, dim_feedforward: int, num_layers: int, dropout: float, max_len: int, device) -> None:
        super(Decoder, self).__init__()
        self.Emb = Emb
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        embedding_dim: int = self.Emb.vector_size

        self.pos_encoder = PositionalEncoding(embedding_dim, self.dropout, max_len)

        self.transformer_decoder = TransformerDecoder(
            TransformerDecoderLayer(embedding_dim, self.nhead, self.dim_feedforward, self.dropout, batch_first=True),
            self.num_layers
        )

        self.output_layer = nn.Linear(embedding_dim, len(self.Emb.key_to_index))
        self.tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(max_len, device=self.device)

    def forward(self, tgt: Tensor) -> Tensor:
        tgt = self.Emb.vectors[tgt]
        tgt = torch.tensor(tgt).to(self.device)
        tgt = self.pos_encoder(tgt)

        memory = torch.zeros_like(tgt, device=self.device)

        output = self.transformer_decoder(tgt, memory, tgt_mask=self.tgt_mask)
        output = self.output_layer(output)
        return output