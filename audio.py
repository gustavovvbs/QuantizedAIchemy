import torch 
from torch import nn
from torchvision import datasets 
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor, Lambda 
import matplotlib.pyplot as plt 
import torchvision
from tqdm import tqdm
import torchaudio  

train_dataset = torchaudio.datasets.VCTK_092(
    root = 'data',
    download = True,
)


train_dataloader = DataLoader(train_dataset, batch_size = 64)


print(train_dataloader.dataset[0])