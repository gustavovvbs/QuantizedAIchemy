import torch 
from torch import nn 


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim, commitment_cost):
        super().__init__()
        self.embeddings_dim = embeddings_dim 
        self.num_embeddings = num_embeddings 
        self._commitment_cost = commitment_cost 
        self._embedding = nn.Embedding(num_embeddings, embeddings_dim)

    def forward(self, x):
        # x: (bs, emdeddings_dim, h, w)
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()
        # x: (bs, h, w, embeddings_dim)

        latents = x_reshaped.view(-1, self.embeddings_dim)
        self.x = latents

        # encoded variables: (N, embeddings_dim)

        # compute euclidian distances between every N encoded vector and n_embeddings nth codebook vector 

        distances = torch.cdist(latents, self._embedding.weight)

        args_min = torch.argmin(distances, dim = 1) #flattened vector with minimun distance of differences between evert nth encoded and codebook

        self.quantized_space = self._embedding.weight[args_min]

        e_latent_loss = nn.functional.mse_loss(self.x.detach(), self.quantized_space)
        q_latent_loss = nn.functional.mse_loss(self.x, self.quantized_space.detach())

        loss = q_latent_loss + self._commitment_cost*e_latent_loss

        #copia os gradientes pro input 
        self.quantized_space = self.x + (self.quantized_space - self.x).detach()

        return self.quantized_space

    


