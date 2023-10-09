import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Кодировщик (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),  # Сверточный слой 1
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Сверточный слой 2
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Пулинг слой
        )
        
        # Декодировщик (Decoder)
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),  # Сверточный слой 3
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),  # Сверточный слой 4
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(in_channels=8, out_channels=7, kernel_size=2, stride=2)  # Транспонированный сверточный слой
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MSET(nn.Module):
    def __init__(self):
        super(MSET, self).__init__()
        self.linear = nn.Linear(8, 7)

    def forward(self, x):
        return self.linear(x)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded, _ = self.encoder(x.unsqueeze(0))
        decoded = self.decoder(encoded.squeeze(0))
        return decoded
    
class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x