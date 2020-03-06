import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms

from detchar.dataset import Dataset
from detchar.models.VAE import VAE


df = pd.read_json('dataset.json')
input_size = 512
data_transform = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.Grayscale(),
        transforms.ToTensor()
])
device = 'cuda:1'
dataset = Dataset(df, data_transform)
old_set, new_set = dataset.split_by_labels(['Helix', 'Scratchy'])
train_set, test_set = old_set.split_dataset(0.7)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
vae = VAE(device, input_size, 64, 16)
optimizer = torch.optim.Adam(vae.net.parameters(), lr=1e-3)
vae.init_model(train_loader, test_loader, optimizer)

for epoch in range(10):
    vae.fit_train(epoch+1)
    vae.fit_test(epoch+1)
