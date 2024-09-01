import torch 
from torch import nn
from torchvision import datasets 
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor, Lambda 
import matplotlib.pyplot as plt 
import torchvision
from tqdm import tqdm



train_dataset = datasets.FashionMNIST(
    root = 'data',
    train = False,
    transform=ToTensor(),
    download = True 
)

test_dataset = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(train_dataset, batch_size = 64)

test_dataloader = DataLoader(test_dataset, batch_size = 64) 

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

        self.pre_quantization_conv = nn.Conv2d(4, 2, 1)
        self.embeddings = nn.Embedding(3, 2)
        self.post_quantization_conv = nn.Conv2d(2, 4, 1)

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

        quant_input = self.pre_quantization_conv(encoded_data)

        bs, c, h, w = quant_input.shape
        quant_input = quant_input.permute(0, 2, 3, 1).contiguous()

        quant_input = quant_input.view(quant_input.size(0), -1, c)

        dist = torch.cdist(quant_input, self.embeddings.weight[None, :].repeat((quant_input.size(0), 1, 1))) # (bs, h*w, n_embed)
        
        min_encoding_indices = torch.argmin(dist, dim=-1)

        quantized_space = self.embeddings.weight[min_encoding_indices]

        quantized_space = quantized_space.reshape(bs, h, w, c).permute(0, 3, 1, 2)
        quant_input = quant_input.view(bs, h, w, c).permute(0, 3, 1, 2)

        #make encoding variables push to the quantized embeddings
        commit_loss = torch.mean((quantized_space.detach() - quant_input).pow(2))

        #push the quantized embeddings to the latent encoded spaces
        codebook_loss = torch.mean((quantized_space - quant_input.detach()).pow(2))

        #add a term to control how the codebook terms are going to be modified in the loss
        loss = codebook_loss + self._commitment_cost*commit_loss

        quantized_space = quant_input + (quantized_space - quant_input).detach()

        decoder_input = self.post_quantization_conv(quantized_space)

        decoded_data = self.decoder(decoder_input)

        return decoded_data, loss 


EPOCHS = 10
device = 'cuda'

def train_step(model):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for im, label in tqdm(train_dataloader):
        im = im.float().to(device)
        optimizer.zero_grad()
        out, quantize_loss = model(im)
        recon_loss = loss_fn(out, im)
        loss = recon_loss + quantize_loss
        loss.backward()
        optimizer.step()

model = VQVAE().to(device)

for epoch in range(EPOCHS):
    train_step(model)
    print(f'EPOCH: {epoch+1}/{EPOCHS}')


idxs = torch.randint(0, len(test_dataloader), (100, ))
imgs = torch.cat([test_dataloader.dataset[idx][0][None, :] for idx in idxs]).float()
imgs = imgs.to(device)

model.eval()

reconstructed = model(imgs)[0]
imgs = (imgs+1)/2
reconstructed = 1 - (reconstructed+1)/2
out = torch.hstack([imgs, reconstructed])
output = torch.reshape(out, (-1, 1, 28, 28))
grid = torchvision.utils.make_grid(output.detach().cpu(), nrow=10)
img = torchvision.transforms.ToPILImage()(grid)
img.save('reconstruction.png')
    

