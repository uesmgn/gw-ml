'''
'''
import torch
from torch import nn
from torch.utils import data
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        (label, path) = self.df.iloc[idx,:]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return (img, label)

    def get_labels(self, dtype=str):
        labels = self.df.iloc[:,0].unique().astype(dtype)
        return sorted(labels)
