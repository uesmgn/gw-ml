import torch
import pandas as pd
from torchvision import transforms

from detchar.dataset import Dataset
from detchar.models import vae

df = pd.read_json('dataset.json')
input_size = 512
data_transform = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.Grayscale(),
        transforms.ToTensor()
])
dataset = Dataset(df, data_transform)
train_loader = dataset.get_loader(batch_size=4, shuffle=True)
autoencoder = vae.VAE(input_size, 64, 16)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
autoencoder.init_model(train_loader, optimizer)

for epoch in range(10):
    autoencoder.fit_train(epoch+1)
