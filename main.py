import os
import json
import configparser
from glob import glob
from collections import defaultdict
import re
import pandas as pd


DATASET_DIR = '../H1L1'

if __name__ == '__main__':

    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'utf-8')

    dataset_dir = ini.get('dataset','dataset_dir')
    filter_str = ini.get('dataset','filter_str') or '*_1.0.png'
    subsets = json.loads(ini.get('dataset', 'subsets')) or ['test', 'train', 'validation']

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = f'{base_dir}/{DATASET_DIR}'
    assert os.path.exists(dataset_path)

    data = defaultdict()

    for subset in subsets:
        labels = [os.path.basename(p) for p in glob(f'{dataset_path}/{subset}/*')]
        for label in labels:
            paths = glob(f'{dataset_path}/{subset}/{label}/{filter_str}')
            for path in paths:
                idx = re.match(r'.+(L1|H1)_([a-zA-Z0-9]+)_.+', path).group(2)
                data[idx] = {'label': label, 'path': path}

    df = pd.DataFrame.from_dict(data, orient='index')
    df.to_json(f'{dataset_dir}.json')
    print(f'Success to make .json file have columns (label, path) , shape {df.shape}.')
