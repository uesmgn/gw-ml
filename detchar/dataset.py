import torch
from torch import nn
from torch.utils import data
from PIL import Image
import pandas as pd

class Dataset(data.Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        (label, path) = self.df.iloc[idx, :]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return (img, label)

    def get_labels(self, dtype=str):
        labels = self.df.iloc[:, 0].unique().astype(dtype)
        return sorted(labels)

    def split_by_labels(self, new_labels, n_cat=200):
        df = self.df
        columns = self.df.columns

        old_df = df[~df['label'].isin(new_labels)]
        old_labels = old_df['label'].unique()
        old_dict = {c: len(old_df[old_df['label'] == c]) for c in old_labels}
        old_labels = [k for k, v in old_dict.items() if v >= n_cat]

        old_df_ = pd.DataFrame(columns=columns)
        for c in old_labels:
            old_df_ = old_df_.append(
                old_df[old_df['label'] == c].sample(n=n_cat, random_state=123)
            )
        print(f"# of old classes: {len(old_labels)}")
        print(f"{old_labels}")
        print(f"# of old data: {len(old_df_)}")

        new_df = df[df['label'].isin(new_labels)]
        new_df_ = pd.DataFrame(columns=columns)
        for c in new_labels:
            new_df_ = new_df_.append(
                new_df[new_df['label'] == c].sample(n=n_cat, random_state=123)
            )

        print(f"# of new classes: {len(new_labels)}")
        print(f"{new_labels}")
        print(f"# of new data: {len(new_df_)}")

        return Dataset(old_df_, self.transform), Dataset(new_df_, self.transform)

    def split_dataset(self, alpha=0.8):
        N_train = int(self.__len__()*alpha)
        N_test = self.__len__() - N_train
        return data.random_split(self, [N_train, N_test])
