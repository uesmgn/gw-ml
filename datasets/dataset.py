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
from tqdm import tqdm


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

    def download(self, force_extract=False, force_process=False):
        """Download the GravitySpy data if it doesn't exist."""

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
            dataset = self.read_image_folder(self.extract_folder)
            with open(self.dataset_file, 'wb') as f:
                torch.save(dataset, f)
            print('Done!')

    def read_image_folder(self, path):
        subdirs = sorted([os.path.basename(p) for p in glob.glob(f'{path}/*')])
        img_tensor_stack = []
        target_stack = []

        for i, subdir in enumerate(tqdm(subdirs)):
            files = glob.glob(os.path.join(path, subdir, '*_1.0.png'))
            target  = torch.tensor(i).long()

            for file in files:
                img = PIL.Image.open(file)
                img_tensor = self.setup_transform(img) 
                img_tensor_stack.append(img_tensor)
                target_stack.append(target)
        img_tensor_stack = torch.stack(img_tensor_stack)
        target_stack = torch.stack(target_stack)
        return (img_tensor_stack, target_stack)
