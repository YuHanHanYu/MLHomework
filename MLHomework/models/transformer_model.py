import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len=90, output_dim=90, nhead=4):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out
