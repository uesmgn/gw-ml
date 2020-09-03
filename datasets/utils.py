import _pickle as pickle
import bz2
import torch

def bz2_load(fname):
    with bz2.BZ2File(fname, 'rb') as f:
        pkl = torch.load(f)
    return pickle.loads(pkl)

def save(obj, fname, level=9):
    torch.save(pkl, fname, pickle_protocol=4)
