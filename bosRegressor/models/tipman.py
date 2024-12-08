import torch.nn as nn
import torch

class BoSDistMLP(nn.Module):
    def __init__(self, input_size=72, device='cuda'):
        super(BoSDistMLP, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, 1024).to(device)  # First fully connected layer
        self.fc2 = nn.Linear(1024, 512).to(device)  # Second fully connected layer
        self.fc3 = nn.Linear(512, 256).to(device)  # Third fully connected layer
        self.fc4 = nn.Linear(256, 128).to(device)  # Output layer: single value
        self.fc5 = nn.Linear(128, 1).to(device)  # Output layer: single value

        self.relu = nn.ReLU().to(device)  # Activation function

    def forward(self, x):
        if x.ndim != 2:
            # flatten
            x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class BoSDistTransformer(nn.Module):
    def __init__(self, nhead, num_encoder_layers, d_model, num_intermediate_nodes, dropout=0.5, device='cuda'):
        super(BoSDistTransformer, self).__init__()

        self.embedding = nn.Linear(165, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model * 3, num_intermediate_nodes),  # Multiply by 3 because there are 3 frames.
            nn.ReLU(),
            nn.Linear(num_intermediate_nodes, 1)
        )

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  # (seq_len, batch, embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2).contiguous().view(src.size(1), -1)
        return self.decoder(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Hyperparameters
d_model = 256
nhead = 8
num_encoder_layers = 3
num_intermediate_nodes = 512
dropout = 0.1

