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
loader = dataset.get_loader(batch_size = 4, shuffle = True)
encoder = vae.Encoder()

for idx, (img, label) in enumerate(loader):
    (x, indices) = encoder(img)
    print(x.size())
    break
