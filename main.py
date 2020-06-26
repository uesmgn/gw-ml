import os
from glob import glob
from collections import defaultdict
import re
import pandas as pd

DATASET_DIR = '../TrainingSet'

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = f'{base_dir}/{DATASET_DIR}'
    labels = [os.path.basename(p) for p in glob(f'{dataset_path}/*')]

    data = defaultdict()

    for label in labels:
        # paths = glob(f'{dataset_path}/{label}/*_1.0.png')
        paths = glob(f'{dataset_path}/{label}/*_1.0.png')
        for path in paths:
            idx = re.match(r'.+(L1|H1)_([a-zA-Z0-9]+)_.+', path).group(2)
            data[idx] = {'label': label, 'path': path}
    df = pd.DataFrame.from_dict(data, orient='index')
    df.to_json('dataset.json')
    print(df.shape)
