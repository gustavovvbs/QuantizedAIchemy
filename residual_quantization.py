import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision
from tqdm import tqdm

train_dataset = datasets.FashionMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

test_dataset = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

class ResidualQuantizer(nn.Module):
    def __init__(self, num_embeddings, z_dim, embedding_dim, commitment_cost):
        super(ResidualQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.quant_conv = nn.Conv2d(z_dim, embedding_dim, 1)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.post_quant_conv = nn.Conv2d(embedding_dim, z_dim, 1)

    def forward(self, x):
        # Quantization
        quant_input = self.quant_conv(x)
        bs, c, h, w = quant_input.shape
        quant_input = quant_input.permute(0, 2, 3, 1).contiguous()
        quant_input = quant_input.view(bs, -1, c)
        
        dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        quantized_space = self.embedding.weight[min_encoding_indices]
        quantized_space = quantized_space.view(bs, h, w, c).permute(0, 3, 1, 2)
        quant_input = quant_input.view(bs, h, w, c).permute(0, 3, 1, 2)
        
        # Compute losses
        commit_loss = torch.mean((quantized_space.detach() - quant_input).pow(2))
        codebook_loss = torch.mean((quantized_space - quant_input.detach()).pow(2))
        total_loss = codebook_loss + self.commitment_cost * commit_loss
        
        # Update residual
        quantized_space = quant_input + (quantized_space - quant_input).detach()
        updated_residual = x - self.post_quant_conv(quantized_space)

        return updated_residual, total_loss

class ResidualVQVAE(nn.Module):
    def __init__(self, num_residual_blocks=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

        self.num_residual_blocks = num_residual_blocks
        self.residual_quantizers = nn.ModuleList([ResidualQuantizer(3, 2, 0.25) for _ in range(num_residual_blocks)])

        self._commitment_cost = 0.25

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded_data = self.encoder(x)
        residual = encoded_data
        total_loss = 0

        for quantizer in self.residual_quantizers:
            residual, loss = quantizer(residual)
            total_loss += loss

        decoded_data = self.decoder(encoded_data - residual)
        return decoded_data, total_loss


if __name__ == '__main__':
    EPOCHS = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_step(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        for im, _ in tqdm(train_dataloader):
            im = im.float().to(device)
            optimizer.zero_grad()
            out, quantize_loss = model(im)
            recon_loss = loss_fn(out, im)
            loss = recon_loss + quantize_loss
            loss.backward()
            optimizer.step()

    model = ResidualVQVAE().to(device)

    for epoch in range(EPOCHS):
        train_step(model)
        print(f'EPOCH: {epoch+1}/{EPOCHS}')

    idxs = torch.randint(0, len(test_dataloader), (100,))
    imgs = torch.cat([test_dataloader.dataset[idx][0][None, :] for idx in idxs]).float()
    imgs = imgs.to(device)

    model.eval()

    reconstructed = model(imgs)[0]
    imgs = (imgs + 1) / 2
    reconstructed = 1 - (reconstructed + 1) / 2
    out = torch.hstack([imgs, reconstructed])
    output = torch.reshape(out, (-1, 1, 28, 28))
    grid = torchvision.utils.make_grid(output.detach().cpu(), nrow=10)
    img = torchvision.transforms.ToPILImage()(grid)
    img.save('reconstruction.png')
