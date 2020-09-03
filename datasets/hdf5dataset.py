import h5py
import pathlib
from torch.utils import data


class HDF5Dataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        self.data_cache = []
        self.transform = transform
        self.target_transform = target_transform

        root = pathlib.Path(root)
        assert root.is_file()
        self.root = str(root.resolve())
        print('Appending data to cache...')
        with h5py.File(self.root, 'r') as fp:
            self._init_data_cache(fp)
        print(f'Successfully loaded {self.__len__()} of dataset.')

    def __getitem__(self, i):
        ref = self.data_cache[i]
        with h5py.File(self.root, 'r') as fp:
            item = fp[ref]
            target = dict(item.attrs)
            img = item[:]
        if callable(self.target_transform):
            target = self.target_transform(target)
        if callable(self.transform):
            img = self.transform(img)
        return target, img

    def __len__(self):
        return len(self.data_cache)

    def _init_data_cache(self, item):
        if hasattr(item, 'values'):
            # if item is group
            for it in item.values():
                self._init_data_cache(it)
        else:
            # if item is dataset
            self.data_cache.append(item.ref)
