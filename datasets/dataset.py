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
import sys
import pickle
import io
import uuid

from . import utils as ut

class GravitySpy(torch.utils.data.Dataset):

    resource = 'https://zenodo.org/record/1476551/files/trainingsetv1d1.tar.gz'

    def __init__(self, root, transform=None, target_transform=None,
                 download=False, force_extract=False, force_process=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download_and_extract(force_extract)

        if not os.path.exists(self.extract_folder):
            raise RuntimeError('Extract folder not found.' +
                               ' You can use download(..., force_extract=True) to force extract archive')

        self.targets = sorted([p for p in os.listdir(self.extract_folder)])

        if force_process or not os.path.exists(self.dataset_pickle):

            import  uuid

            _image_transform = transforms.Compose([
                transforms.Lambda(lambda x: transforms.functional.crop(x, top=58, left=100, height=482, width=574)),
                transforms.Grayscale()
            ])

            def _meta_transform():
                latypes = {'0.5': 5, '1.0': 10, '2.0': 20, '4.0': 40}
                def encode(file):
                    la = os.path.splitext(os.path.basename(file))[0][-3:]
                    latype = latypes[la]
                    return {'id': uuid.uuid4().hex, 'type': latype}
                return encode
            _meta_transform = _meta_transform()

            self.proccess_image_folder(x_size=(482, 574),
                                       image_transform=_image_transform,
                                       meta_transform=_meta_transform)

        if not os.path.exists(self.dataset_pickle):
            raise RuntimeError('Dataset not found.' +
                               ' You can use force_process=True to force process it')

        with open(self.dataset_pickle, 'rb') as f:
            output = pickle.load(f)
        image_tensor = torch.Tensor([])
        labels_tensor = torch.Tensor([])
        meta = {}
        for out in output:
            stream = io.BytesIO(out)
            out = np.load(stream, allow_pickle=True)
            images = out['image']
            labels = out['label']
            m = out['meta'].item()


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
    def dataset_pickle(self):
        return os.path.join(self.processed_folder, 'dataset.pkl')

    def download_and_extract(self, force_extract=False):
        """Download and extract the GravitySpy data if it doesn't exist."""

        os.makedirs(self.raw_folder, exist_ok=True)

        # extract .gz file
        if force_extract or not os.path.exists(self.extract_folder):
            print('Extracting...')
            url = self.resource
            filename = url.rpartition('/')[-1]
            utils.download_and_extract_archive(url, download_root=self.raw_folder,
                                               filename=filename)
            for f in os.listdir(self.raw_folder):
                # extracted directory
                if os.path.isdir(os.path.join(self.raw_folder, f)):
                    os.rename(os.path.join(self.raw_folder, f),
                              os.path.join(self.raw_folder, os.path.basename(self.extract_folder)))
            print('Done!')

    def proccess_image_folder(self, x_size=(482, 574), transform=None, meta_transform=None, num_images_per_seq=1000):

        # initialize queue
        images = np.empty((*x_size, 0)).astype(np.uint8)
        labels = np.empty(0).astype(np.uint8)
        meta = {}
        stream = io.BytesIO()
        output = []

        idx = 0

        for label, target in enumerate(self.targets):
            if not os.path.isdir(os.path.join(self.extract_folder, target)):
                continue
            files = os.listdir(os.path.join(self.extract_folder, target))
            print(f'Processing {target}...')
            for file in tqdm(files):
                file = os.path.join(self.extract_folder, target, file)
                try:
                    image = PIL.Image.open(file)
                except:
                    continue
                if transform is not None:
                    image = transform(image)
                image = np.array(image).astype(np.uint8)
                image = np.expand_dims(image, axis=-1)
                images = np.dstack([images, image])
                labels = np.append(labels, label)
                if meta_transform is not None:
                    meta[idx] = meta_transform(file)
                idx += 1

                if images.shape[-1] >= num_images_per_seq:
                    np.savez_compressed(stream,
                                        image=images.transpose(-1, *list(range(images.ndim-1))),
                                        label=labels,
                                        meta=meta)
                    output.append(stream.getvalue())
                    # reset queue
                    images = np.empty((*x_size, 0)).astype(np.uint8)
                    labels = np.empty(0).astype(np.uint8)
                    meta = {}
                    stream.truncate(0)
                    stream.seek(0)

        np.savez_compressed(stream,
                            image=images.transpose(-1, *list(range(images.ndim-1))),
                            target=targets,
                            meta=meta)
        output.append(stream.getvalue())

        with open(self.dataset_pickle, 'wb') as f:
            pickle.dump(output, f)


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
