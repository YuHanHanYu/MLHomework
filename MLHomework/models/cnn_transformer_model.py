import torch.nn as nn

class CNNTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim=90):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=2, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)          # to (batch, input_dim, seq_len)
        x = self.relu(self.conv1(x))   # (batch, 32, seq_len)
        x = x.permute(0, 2, 1)          # to (batch, seq_len, 32)
        x = self.encoder(x)            # (batch, seq_len, 32)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out
