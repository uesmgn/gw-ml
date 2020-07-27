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


class SampleGenerator(object):


  def __init__(self, data, sample_count):
    self._data = data
    self._sample_count = sample_count
    self._count = 0

  def __iter__(self):
    return SampleGenerator(self._data, self._sample_count)

  def __len__(self):
    return self._sample_count

  def __next__(self):
    return self.next()

  def next(self):
    if self._count >= self._sample_count:
      raise StopIteration
    self._count += 1
    return self._data
