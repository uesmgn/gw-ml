import os
import argparse
import time

parser = argparse.ArgumentParser()

def hdf_file(path):
    if os.path.exists(path):
        if os.path.splitext(path)[-1] in ('.h5', '.hdf5'):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} must be .h5 or .hdf5 file")
    else:
        raise argparse.ArgumentTypeError(f"{path} is not exist")


parser.add_argument('path_to_hdf', type=hdf_file,
                    help='dataset path')
args = parser.parse_args()

from tqdm import tqdm
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

dataset_hdf = args.path_to_hdf
# start  = time.time()
# df = pd.read_hdf(dataset_hdf, '/meta')
# print(f'Time to read metadata: {time.time() - start:.3f} sec')
# index = df[df['label'].eq(3)].index.values
# start  = time.time()
# with h5py.File(dataset_hdf, mode='r') as f:
#     images = f['data'][()]
# print(f'Time to read images: {time.time() - start:.3f} sec')
# for i, img in enumerate(images[index]):
#     plt.imshow(img, cmap='gray')
#     plt.show()
#     if i > 10:
#         break
with h5py.File(dataset_hdf, mode='r') as f:
    print(f)
