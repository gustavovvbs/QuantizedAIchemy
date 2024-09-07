import torch 
from torch import nn 
from residual_quantization import ResidualQuantizer
from torchvision import datasets 
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor 
from tqdm import tqdm 
import torchvision
from PIL import Image, ImageDraw, ImageFont


train_dataset = datasets.FashionMNIST(
    root = 'data',
    train = True, 
    download=False,
    transform = ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root = 'data',
    train = False, 
    download = False, 
    transform = ToTensor()
)

train_dataloader = DataLoader(train_dataset, batch_size = 64)
test_dataloader = DataLoader(test_dataset, batch_size = 64)

class Encoder(nn.Module):
    def __init__(self, in_channel, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 16, 4, 2, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, z_dim, 4, 2, 1),
            nn.InstanceNorm2d(z_dim),
            nn.ReLU(),
        )

        self.in_channel = in_channel; self.z_dim = z_dim

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, z_dim, out_channel):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, out_channel, 4, 2, 1),
        )

        self.z_dim = z_dim; self.out_channel = out_channel

    def forward(self, x):
        return self.decoder(x)


class TopLevelVQVAE(nn.Module):
    def __init__(self, in_channel, z_dim, embeddings_dim = 16, num_residual_blocks = 2):
        super().__init__()
        self.encoder = Encoder(in_channel, z_dim)
        self.residual_quantizers = nn.ModuleList([ResidualQuantizer(3, z_dim, embeddings_dim,  0.25) for _ in range(num_residual_blocks)])
        self.decoder = Decoder(z_dim, in_channel)
    def forward(self, x):
        encoded_data = self.encoder(x)
        residual = encoded_data 
        total_loss = 0

        for quantizer in self.residual_quantizers:
            residual, loss = quantizer(residual)
            total_loss += loss
        

        decoded_data = self.decoder(encoded_data - residual)
        return decoded_data, total_loss 

class MidLevelVQVAE(nn.Module):
    def __init__(self, in_channel, z_dim, embeddings_dim = 16, num_residual_blocks = 2):
        super().__init__()
        self.encoder = Encoder(in_channel, z_dim)
        self.residual_quantizers = nn.ModuleList([ResidualQuantizer(3, z_dim, embeddings_dim, 0.25) for _ in range(num_residual_blocks)])
        self.decoder = Decoder(z_dim, in_channel)
    def forward(self, x):
        encoded_data = self.encoder(x)
        residual = encoded_data 
        total_loss = 0

        for quantizer in self.residual_quantizers:
            residual, loss = quantizer(residual)
            total_loss += loss
            
        decoded_data = self.decoder(encoded_data - residual)
        return decoded_data, total_loss

class BottomLevelVQVAE(nn.Module):
    def __init__(self, in_channel, z_dim, embeddings_dim, num_residual_blocks = 2):
        super().__init__()
        self.encoder = Encoder(in_channel, z_dim)
        self.residual_quantizers = nn.ModuleList([ResidualQuantizer(3, z_dim, embeddings_dim, 0.25) for _ in range(num_residual_blocks)])
        self.decoder = Decoder(z_dim, in_channel)
    def forward(self, x):
        encoded_data = self.encoder(x)
        residual = encoded_data 
        total_loss = 0

        for quantizer in self.residual_quantizers:
            residual, loss = quantizer(residual)
            total_loss += loss
            
        decoded_data = self.decoder(encoded_data - residual)
        return decoded_data, total_loss

class HierarchicalVQVAE(nn.Module):
    #top level has the highest degree of absctraction, since it has more dimensions, the model will have the freedom to compress the data more, and the latent space will try to learn more abstract features about the data, like abstract global patterns of the data, while the bottom level will try to learn local patterns that will help to reconstruct the data more efficiently
    #bottom level has the lowest degree of compression, since it has less dimensions, the latent space will try to preserve more information about the raw data to reconstruct it, things like edges, and short-term patterns.
    def __init__(self, in_channel =1, num_residual_blocks = 2, top_dim = 16, mid_dim = 8, bottom_dim = 4):
        super().__init__()
        self.top_level = TopLevelVQVAE(in_channel, 4, top_dim, num_residual_blocks)
        self.mid_level = MidLevelVQVAE(in_channel,4, mid_dim, num_residual_blocks)
        self.bottom_level = BottomLevelVQVAE(in_channel,4, bottom_dim, num_residual_blocks)

    def forward(self, x):
        x_t, loss_t = self.top_level(x)
        x_m, loss_m = self.mid_level(x)
        x_b, loss_b = self.bottom_level(x)

        return (x_t, loss_t), (x_m, loss_m), (x_b, loss_b)


def train_step(model, train_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr = 4e-4)
    loss_fn = torch.nn.MSELoss()

    for im, _ in tqdm(train_dataloader):
        im = im.float().to(device)
        optimizer.zero_grad()
        (out_t, loss_t), (out_m, loss_m), (out_b, loss_b) = model(im)
    
        rec_loss_t = loss_fn(out_t, im)
        rec_loss_m = loss_fn(out_m, im)
        rec_loss_b = loss_fn(out_b, im)
        loss = rec_loss_t + rec_loss_m + rec_loss_b
        loss += loss_t + loss_m + loss_b
        loss.backward()
        optimizer.step()

    return loss

EPOCHS = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HierarchicalVQVAE().to('cuda')

for epoch in range(EPOCHS):
    loss = train_step(model, train_dataloader)
    print(f'EPOCH: {epoch+1}/{EPOCHS}, loss: {loss.item()}')


idxs = torch.randint(0, len(test_dataloader), (100,))
imgs = torch.cat([test_dataloader.dataset[idx][0][None, :] for idx in idxs]).float()
imgs = imgs.to(device)

model.eval()

(out_t, loss_t), (out_m, loss_m), (out_b, loss_b) = model(imgs)

imgs = (imgs + 1) / 2
out_t = 1 - (out_t + 1) / 2
out_m = 1 - (out_m + 1) / 2
out_b = 1 - (out_b + 1) / 2

out = torch.hstack([imgs, out_t, out_m, out_b])
output = torch.reshape(out, (-1, 1, 28, 28))

grid = torchvision.utils.make_grid(output.detach().cpu(), nrow=10)

img = torchvision.transforms.ToPILImage()(grid)

draw = ImageDraw.Draw(img)
font = ImageFont.load_default()

labels = ['Original', 'Top', 'Middle', 'Bottom']

for i in range(10):
    for j, label in enumerate(labels):
        draw.text((i * 28, j * 28), label, font=font, fill=(255, 255, 255))

img.save('reconstruction_with_labels.png')



