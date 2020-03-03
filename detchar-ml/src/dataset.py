import torch
from torch import nn
from torch.utils import data

'''
json file format:

{
    'id': {
        'label':1,
        'img': abc.png
    }

}
'''
class Dataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def get_labels(self):
        return self.df.columns.to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx, :]
        img_name = f'{self.root_dir}/{item[self._columns[0]]}/{item[self._columns[1]]}'
        img = Image.open(img_name)
        if self.transform:
            img = self.transform(img)
        return (img, item[self._columns[0]])
