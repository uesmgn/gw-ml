import os
import glob
import torchvision.transforms as transforms
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

    def __init__(self, root, labels=None, num_per_label=None,
                 transform=None, target_transform=None, setup_transform=None,
                 download=False, force_extract=False, force_process=False):
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

        if labels is not None:
            targets = self.targets.numpy()
            (idx,) = np.hstack([np.where(targets==label) for label in labels])
            np.random.shuffle(idx)
            idx = np.hstack([list(filter(lambda i: targets[i] == label, idx))[:num_per_label] for label in labels])
            self.data, self.targets = self.data[idx], self.targets[idx]

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
    def meta_file(self):
        return os.path.join(self.processed_folder, 'dataset_meta.json')

    def download(self, force_extract=False, force_process=False, filter_str='*'):
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
            images, targets, meta = self.load_image_folder(self.extract_folder, filter_str)
            with open(self.dataset_file, 'w') as f:
                dataset = (images, targets)
                torch.save(dataset, f)
            with open(self.meta_file, 'w') as f:
                json.dump(meta, f, indent=4)
            print('Done!')

    def load_image_folder(self, path, filter_str='*'):
        import uuid

        subdirs = sorted([os.path.basename(p) for p in glob.glob(f'{path}/*')])
        images = []
        targets = []
        meta = {}
        index = 0

        for i, subdir in enumerate(tqdm(subdirs)):
            files = glob.glob(os.path.join(path, subdir, filter_str))
            target  = torch.tensor(i).long()
            print(files)

            for f in files:
                image = PIL.Image.open(f)
                image = trainsforms.functional.pil_to_tensor(image)
                images.append(image)
                targets.append(target)
                id = uuid.uuid4().hex
                meta[id] = {'file': f, 'index': index, 'target': i}

        images = torch.stack(images)
        targets = torch.stack(targets)
        return images, targets, meta

    def split_dataset(self, alpha=0.8, shuffle=True):
        N_train = int(self.__len__() * alpha)
        idx = np.arange(self.__len__())
        if shuffle:
            np.random.shuffle(idx)
        train_idx, test_idx = idx[:N_train], idx[N_train:]

        train_set = copy.deepcopy(self)
        train_set.data, train_set.targets = self.data[train_idx], self.targets[train_idx]

        test_set = copy.deepcopy(self)
        test_set.data, test_set.targets = self.data[test_idx], self.targets[test_idx]

        return train_set, test_set

    def uniform_label_sampler(self, labels, num_per_label=50):
        targets = self.targets.numpy()
        (idx,) = np.hstack([np.where(targets==label) for label in labels])
        np.random.shuffle(idx)
        uni_idx = np.hstack([list(filter(lambda i: targets[i] == label, idx))[:num_per_label] for label in labels])
        rem_idx = np.array(list(set(idx) - set(uni_idx))).astype(np.integer)

        uni_set = copy.deepcopy(self)
        uni_set.data, uni_set.targets = self.data[uni_idx], self.targets[uni_idx]

        rem_set = copy.deepcopy(self)
        rem_set.data, rem_set.targets = self.data[rem_idx], self.targets[rem_idx]

        return uni_set, rem_set
