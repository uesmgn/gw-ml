from torch.utils import data
from PIL import Image
import pandas as pd
import numpy  as  np


class Dataset(data.Dataset):

    def __init__(self, df, transform=None, columns=('label', 'path')):
        self.df = df
        for column in columns:
            assert column in df.columns
        self.columns = columns
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        (label, path) = self.df.iloc[idx, :]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return (img, label)

    def unique_column(self, column, dtype=str, sort=True):
        arr = self.df[column].unique().astype(dtype)
        arr =  np.array(arr)
        return np.sort(arr) if sort else arr

    def get_by_keys(self, column, keys):
        df = self.df
        assert column in df.columns
        df = df[df[column].isin(keys)]
        return Dataset(df, self.transform)

    def sample(self, column, min_value_count=0,
               n_sample=1, random_state=0, copy=False):
        df = self.df
        assert column in df.columns
        assert min_value_count <= n_sample
        if n_sample > 0:
            value_count = df[column].value_counts()
            idx = value_count[value_count > min_value_count].index
            df = df[df[column].isin(idx)]
            gp = df.groupby(column)
            df = gp.apply(lambda x: x.sample(n=n_sample))
        if copy:
            return Dataset(df, self.transform)
        self.df = df

    def random_split(self, alpha=0.8, random_state=0):
        df = self.df
        n = int(len(df) * alpha)
        # shuffle DataFrame
        df = df.sample(frac=1, random_state=random_state)
        split_df = (df.iloc[:n, :], df.iloc[n:, :])
        return (Dataset(split_df[0], self.transform),
                Dataset(split_df[1], self.transform))
