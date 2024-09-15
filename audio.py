import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import torchaudio
import os

# Dataset para carregar arquivos de áudio
class VctkDataset:
    def __init__(self, dataset_path, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dataset_dir = os.path.join(dataset_path, 'wav48_silence_trimmed')
        self.max_length = 95000
        self.file_list = []
        for speaker in os.listdir(self.dataset_dir):
            speaker_dir = os.path.join(self.dataset_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for audio_file in os.listdir(speaker_dir):
                if audio_file.endswith('.flac'):
                    self.file_list.append(os.path.join(speaker_dir, audio_file))
    
    def pad_or_truncate(self, waveform):
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            pad_amount = self.max_length - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def __iter__(self):
        self.current_batch = []
        for audio_path in self.file_list:
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = self.pad_or_truncate(waveform)
            self.current_batch.append(waveform)

            if len(self.current_batch) == self.batch_size:
                batch = torch.stack(self.current_batch)
                self.current_batch = []
                yield batch

        if len(self.current_batch) > 0:
            batch = torch.stack(self.current_batch)
            yield batch

    def __len__(self):
        return len(self.file_list)


dataset = VctkDataset('data/vctk', 64)

class Encoder(nn.Module):
    def __init__(self, in_channel, z_dim):
        super().__init__()
        self.encoder_stack = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, z_dim)
        self.log_var = nn.Linear(64, z_dim)

    def forward(self, x):
        residual = x
        x = self.encoder_stack(x)
        x = x + residual  # skip connection
        x = x.permute(0, 2, 1)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

example = torch.randn(1, 1, 2000)
dencoder = Encoder(1, 32)
# dout = dencoder(example)
# eps = torch.randn_like(dout[1])
# z = dout[1]*eps + dout[0]
# print(z.shape)

# ex = next(iter(dataset))
# encoded = dencoder(ex)
# eps = torch.randn_like(encoded[1])
# z = encoded[1]*eps + encoded[0]
# print(z.shape)

class Decoder(nn.Module):
    def __init__(self,  z_dim, out_dim = 1):
        super().__init__()
        self.decoder_stack = nn.Sequential(
            nn.ConvTranspose1d(in_channels=z_dim, out_channels=z_dim//2, kernel_size=1),
            nn.InstanceNorm1d(z_dim//2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=z_dim//2, out_channels=out_dim, kernel_size=1),
        )

    def forward(self, x):
        x = self.decoder_stack(x)
        return x 

class VAE(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, z_dim)
        self.decoder = Decoder(z_dim)
        self.in_dim = in_dim 
        self.z_dim =z_dim 

    def forward(self, x):
        mu, log_var = self.encoder(x)
        eps = torch.randn_like(log_var)
        std = torch.exp(0.5 * log_var)
        z = std * eps + mu 
        z = z.permute(0, 2, 1)
        decoded = self.decoder(z)
        return decoded, mu, std


def kl_loss(mu, std):
    kl = -0.5*torch.mean(1 + 2*torch.log(std) - mu.pow(2) - (std.pow(2)))

    return kl

def train(model, dataset = dataset, num_epochs = 1):
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9)
    # resample_transform = torchaudio.transforms.Resample(orig_freq=48800, new_freq=16000)
    mse_loss = nn.MSELoss()

    
    for epoch in range(num_epochs):
        i = 0
        for data in dataset:
            # data = resample_transform(data)
            optim.zero_grad()
            decoded, mu, std = model(data)
            rec_loss = mse_loss(decoded, data)
            reg_loss = kl_loss(mu, std)
            loss = rec_loss + 0.1*reg_loss 
            loss.backward()
            optim.step()
            print(f'loss iter {loss}, kld_loss: {reg_loss}, rec_loss: {rec_loss}')
            data = data.detach()
            plt.plot(data[0, 0, :], label = 'REAL')
            plt.legend()
            plt.show()
            decoded = decoded.detach().cpu()
            plt.plot(decoded[0, 0, :].numpy(), label = 'generated')
            plt.legend()
            plt.show()
            decoded = decoded.squeeze(1)
            print(decoded.shape)
            torchaudio.save(f'output{i}.wav', decoded, 48000)
            i += 1


        

            
vae = VAE(in_dim=1, z_dim=32)
train(vae)


