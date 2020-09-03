import argparse
import os
import re
import warnings
from glob import glob

import h5py
import numpy as np
import pandas as pd
from hashids import Hashids
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray
from tqdm import tqdm

parser = argparse.ArgumentParser()


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is invalid path")


def hdf_file(path):
    if os.path.exists(path):
        while True:
            try:
                p = input(f"{path} is already exists, overwrite? [y/n] ")
                if p is 'y':
                    return path
                elif p is 'n':
                    print('exit...')
                    raise
            except:
                exit(0)
            print(f"enter [y/n]")
    elif os.path.splitext(path)[-1] in ('.h5', '.hdf5'):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} must be .h5 or .hdf5 file")


parser.add_argument('dataset_root', type=dir_path,
                    help='dataset directory')
parser.add_argument('path_to_hdf', type=hdf_file,
                    help='dataset path')
args = parser.parse_args()

dataset_root = os.path.abspath(args.dataset_root)

# target names
targets = sorted(os.path.basename(p)
                 for p in glob(os.path.join(dataset_root, '*')))

meta = []
idx = 0
for target in tqdm(targets):
    target_root = os.path.join(dataset_root, target)
    if not os.path.isdir(target_root):
        continue
    files = glob(os.path.join(target_root, '*'))
    df = pd.DataFrame(files, columns=['file'])
    df = df.assign(bundle=df.file.apply(
        lambda x: re.sub(r'([H,L]1)_([a-zA-Z0-9]{10})_.*.png', r'\1_\2', os.path.basename(x))))
    for bundle, bf in df.groupby('bundle'):
        bundle_id = Hashids(min_length=6).encode(idx)
        idx += 1
        for file in bf.file:
            span = re.sub(r'.*([0-9]\.[0-9]).png', r'\1', os.path.basename(file)).replace('.', '')
            uid = bundle_id + '_' + span
            assert uid not in meta
            meta.append({'unique_id': uid,
                         'file_path': file,
                         'target_name': target,
                         'bundle_id': bundle_id,
                         'span': span})

df = pd.DataFrame(meta)

def file2img(file):
    # imread and remove alpha channel
    img_rgb = io.imread(file)[..., :3]
    # grayscale
    img_gray = rgb2gray(img_rgb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_gray = img_as_ubyte(img_gray)
    return img_gray


dataset_hdf = os.path.abspath(args.path_to_hdf)


def sample_for_bundle(ddf, n_bundle=4):
    bundles = np.random.choice(ddf.bundle.unique(), n_bundle, replace=False)
    return ddf[ddf.bundle.isin(bundles)]

# df = df.groupby('target_name', group_keys=False).apply(lambda d: sample_for_each_bundle(d))
df = df.groupby(['target_name', 'bundle_id'], group_keys=False).apply(lambda d: d.sort_values('span'))
df = df.reset_index(drop=True)
df = df.assign(target_index=pd.Categorical(df['target_name']).codes)
df = df.reindex(columns=['unique_id', 'file_path', 'target_name', 'target_index', 'bundle_id', 'span'])

with h5py.File(dataset_hdf, mode='w') as fp:
    for (target_name, target_index), tf in df.groupby(['target_name', 'target_index']):
        tp = fp.create_group(target_name)
        print(f'Stroing {target}...')
        for bundle, bf in tqdm(tf.groupby('bundle_id')):
            images = []
            bf = bf.sort_values('span')
            for index, row in bf.iterrows():
                images.append(file2img(row['file_path']))
            images = np.stack(images)
            ds = tp.create_dataset(bundle,
                                   data=images,
                                   shape=images.shape,
                                   dtype='uint8',
                                   compression="gzip")
            ds.attrs['target_name'] = target_name
            ds.attrs['target_index'] = target_index
        fp.flush()
