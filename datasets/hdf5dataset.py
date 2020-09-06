import h5py
import pathlib
from torch.utils import data
import copy

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
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

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

    def split_dataset(self, alpha=0.8, shuffle=True):
        N_train = int(self.__len__() * alpha)
        idx = np.arange(self.__len__())
        if shuffle:
            np.random.shuffle(idx)
        train_idx, test_idx = idx[:N_train], idx[N_train:]

        train_set = copy.deepcopy(self)
        train_ref = [self.data_cache[i] for i in train_idx]
        train_set.data_cache = train_ref

        test_set = copy.deepcopy(self)
        test_ref = [self.data_cache[i] for i in test_idx]
        test_set.data_cache = test_ref

        return train_set, test_set
