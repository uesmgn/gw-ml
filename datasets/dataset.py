import os
import glob
import PIL
import pandas as pd
import numpy  as  np
import requests
import torchvision.datasets.utils as utils
import torchvision.transforms as transforms
import torch
import codecs
import json
from tqdm import tqdm
import copy

class GravitySpy(torch.utils.data.Dataset):

    resouce = 'https://zenodo.org/record/1476551/files/trainingsetv1d1.tar.gz'

    def __init__(self, root, transform=None, target_transform=None, setup_transform=None, download=False,
                 force_extract=False, force_process=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.setup_transform = setup_transform

        if download:
            self.download(force_extract, force_process)

        if not os.path.exists(self.dataset_file):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data, self.targets = torch.load(self.dataset_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = transforms.ToPILImage()(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def extract_folder(self):
        return os.path.join(self.raw_folder, 'TrainingSet')

    @property
    def dataset_file(self):
        return os.path.join(self.processed_folder, 'dataset.pt')

    @property
    def target_file(self):
        return os.path.join(self.processed_folder, 'target.json')

    def download(self, force_extract=False, force_process=False):
        """Download the GravitySpy data if it doesn't exist."""

        if os.path.exists(self.dataset_file):
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # extract .gz file
        if force_extract or not os.path.exists(self.extract_folder):
            print('Extracting...')
            url = self.resouce
            filename = url.rpartition('/')[-1]
            utils.download_and_extract_archive(url, download_root=self.raw_folder,
                                               filename=filename)
            print('Done!')

        # process and save as torch files
        if force_process or not os.path.exists(self.dataset_file):
            print('Processing...')
            dataset, target_dir = self.read_image_folder(self.extract_folder)
            with open(self.dataset_file, 'wb') as f:
                torch.save(dataset, f)
            with open(self.target_file, 'w') as f:
                json.dump(target_dir, f, indent=4)
            print('Done!')

    def read_image_folder(self, path):
        subdirs = sorted([os.path.basename(p) for p in glob.glob(f'{path}/*')])
        img_tensor_stack = []
        target_stack = []
        target_dir = {}

        for i, subdir in enumerate(tqdm(subdirs)):
            files = glob.glob(os.path.join(path, subdir, '*_1.0.png'))
            target  = torch.tensor(i).long()
            target_dir[i] = subdir

            for f in files:
                img = PIL.Image.open(f)
                img_tensor = self.setup_transform(img)
                img_tensor_stack.append(img_tensor)
                target_stack.append(target)
        img_tensor_stack = torch.stack(img_tensor_stack)
        target_stack = torch.stack(target_stack)
        return (img_tensor_stack, target_stack), target_dir

    def get_by_keys(self, keys):
        data_stack = []
        target_stack = []
        for data, target in zip(self.data, self.targets):
            if int(target) in keys:
                data_stack.append(data)
                target_stack.append(target)
        data_stack = torch.stack(data_stack)
        target_stack = torch.stack(target_stack)
        self.data, self.targets = data_stack, target_stack

    def split_dataset(self, alpha=0.8):
        N_train = int(self.__len__() * alpha)
        train_data, train_target = self.data[:N_train], self.targets[:N_train]
        train_set = copy.deepcopy(self)
        train_set.data, train_set.target = train_data, train_target

        test_data, test_target = self.data[N_train:], self.targets[N_train:]
        test_set = copy.deepcopy(self)
        test_set.data, test_set.target = test_data, test_target

        return train_set, test_set
