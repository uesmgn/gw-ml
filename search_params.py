import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import optuna
import time
from gmvae.utils import parameters
from gmvae.dataset import Dataset
from gmvae.network import GMVAE
from gmvae import loss_function


n_epoch = 20
batch_size = 32
num_workers = 4
x_size = 486
dataset_json = 'dataset.json'

df = pd.read_json(dataset_json)
data_transform = transforms.Compose([
    transforms.CenterCrop((x_size, x_size)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
dataset = Dataset(df, data_transform)
dataset.sample('label', min_value_count=200, n_sample=200)
xlabels = dataset.unique_column('label')
loader = DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=True)

def objective(trial):
    device_ids = range(torch.cuda.device_count())
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'

    nargs = dict()

    nargs['x_shape'] = (1, x_size, x_size)
    nargs['y_dim'] = xlabels.size
    nargs['pool_kernels'] = [3, 3, 3, 3]
    nargs['middle_size'] = 6

    nargs['bottle_channel'] = 16
    nargs['z_dim'] = parameters.get_dim(trial, 'z', 32, 512, 32)
    nargs['w_dim'] = parameters.get_dim(trial, 'w', 2, 32, 2)
    nargs['conv_channels'] = parameters.get_channels(trial, 32, 128, 32, 4)
    nargs['kernels'] = parameters.get_kernels(trial, 3, 7, 2, 4)
    nargs['dense_dim'] = parameters.get_dim(trial, 'dense', 64, 128, 32)
    nargs['activation'] = parameters.get_activation(trial)
    nargs['pooling'] = parameters.get_pooling(trial)

    largs = dict()
    largs['rec_wei'] = 1.
    largs['cond_wei'] = 1.
    largs['w_wei'] = 1.
    largs['y_wei'] = 1.
    largs['y_thres'] = 0.

    model = GMVAE(nargs)

    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    model.to(device)

    optimizer = parameters.get_optimizer(trial, model)
    criterion = loss_function.Criterion()

    try:
        time_start = time.time()
        for i in range(n_epoch):
            model.train()
            n_samples = 0
            total_loss = 0
            for _, (x, l) in enumerate(loader):
                x = x.to(device)
                optimizer.zero_grad()
                params = model(x, return_params=True)
                gmvae_loss = criterion.gmvae_loss(params, largs, reduction='sum')
                gmvae_loss.backward()
                optimizer.step()
                total_loss += gmvae_loss.item()
                n_samples += x.shape[0]
            total_loss /= n_samples
        time_elapse = time.time() - time_start
        print(f'calc time: {time_elapse:.3f} sec')
        return total_loss
    except:
        return 1e10

if __name__ == '__main__':
    trial_size = 100
    study = optuna.create_study()
    study.optimize(objective, n_trials=trial_size)
    print(study.best_params)