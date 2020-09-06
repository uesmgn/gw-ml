import os
import argparse
import torch
from attrdict import AttrDict
from tqdm import tqdm

import datasets


FLAGS = AttrDict(
    batch_size=128,
    num_workers=1,
    dataset_name='HDF5Dataset',
    x_shape=(160, 160)
)

cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if cuda else torch.device('cpu')
if cuda:
    torch.backends.cudnn.benchmark = True


def is_file(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"{path} is not a file.")
    else:
        raise argparse.ArgumentTypeError(f"{path} must is not exists.")


parser = argparse.ArgumentParser()

parser.add_argument('path_to_hdf', type=is_file,
                    help='path to hdf file')
args = parser.parse_args()

root = args.path_to_hdf

dataset = getattr(datasets, FLAGS.dataset_name)(root, data_transform,
                                                target_transform)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size,
                                          num_workers=FLAGS.num_workers,
                                          shuffle=True, drop_last=True)

for step, (x, target) in tqdm(enumerate(data_loader)):
    pass


# def _data_loader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers, shuffle=True, drop_last=True):
#     return
# def _to_acronym(arr, dict):
#     return np.array([f'{l}-{ut.acronym(dict[l])}' for l in arr])
#
# def _to_values(arr, dic):
#     ret = [dic[x] for x in arr]
#     return ret
#
# cuda = torch.cuda.is_available()
# assert cuda, 'This program only support running on GPU.'
# device = torch.device('cuda:0')
# torch.cuda.set_device(FLAGS.local_rank)
# torch.backends.cudnn.benchmark = True
# assert torch.backends.cudnn.enabled, "amp requires cudnn backend to be enabled."
#
# data_transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize(FLAGS.x_size),
#                 transforms.ToTensor()
#             ])
#
# target_transform = transforms.Compose([
#                 transforms.Lambda(lambda x: x['target_index'])
#             ])
#
# dataset = getattr(datasets, FLAGS.dataset)(root=ROOT,
#                                            transform=data_transform,
#                                            target_transform=target_transform)
#
# targets = torch.Tensor(FLAGS.targets).to(torch.long)
# dataset = dataset.get_by_keys(target_labels)
# targets_dict = {k: ut.acronym(v) for k, v in dataset.targets_dict.items()}
#
# train_set, test_set = dataset.split_dataset()
# labeled_set, unlabeled_set = train_set.uniform_label_sampler(target_labels, num_per_class=10)
# grid_set, _ = dataset.uniform_label_sampler(target_labels, num_per_class=1)
# n_steps = len(unlabeled_set) // FLAGS.batch_size
# labeled_batch = len(labeled_set) // n_steps
#
# print(f'dataset length: {len(dataset)}')
# print(f'train_set length: {len(train_set)}')
# print(f'test_set length: {len(test_set)}')
# print(f'unlabeled_set length: {len(unlabeled_set)}')
# print(f'labeled_set length: {len(labeled_set)}')
# print(f'unlabeled_batch: {FLAGS.batch_size}')
# print(f'labeled_batch: {labeled_batch}')
# print(f'target: {targets_dict}')
#
# alpha = len(dataset) * 0.1
#
# data_loader = _data_loader(dataset)
# unlabeled_loader = _data_loader(unlabeled_set)
# # labeled_loader = _data_loader(labeled_set, batch_size=labeled_batch)
# labeled_loader = _data_loader(labeled_set)
# test_loader = _data_loader(test_set)
# grid_loader = _data_loader(grid_set, batch_size=len(grid_set), shuffle=False)
#
# model = models.resvae.ResVAE_M2(x_dim=FLAGS.x_dim, z_dim=FLAGS.z_dim, y_dim=len(target_labels),
#                                 num_blocks=FLAGS.num_blocks, planes=FLAGS.planes,
#                                 filter_size=FLAGS.filter_size, verbose=True)
# # model = apex.parallel.convert_syncbn_model(model)
# model = model.cuda()
# # optim = optimizers.FusedAdam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
# optim = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
# # optim = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
# model, optim = amp.initialize(model, optim, opt_level=FLAGS.opt_level)
#
# if FLAGS.distributed:
#     model = DDP(model, delay_allreduce=True)
# model = nn.DataParallel(model)
#
# print(model.state_dict())
#
# stats = defaultdict(lambda: list())
# outdir = f'result_{int(time.time())}'
#
# for epoch in range(1, FLAGS.num_epochs):
#     loss = defaultdict(lambda: 0)
#     model.train()
#
#     for step, ((ux, _), (lx, target)) in enumerate(zip(unlabeled_loader, cycle(labeled_loader))):
#         target = F.one_hot(target, num_classes=len(target_labels))
#         ux, lx, target = ux.to(device), lx.to(device), target.to(device)
#         labeled_loss, unlabeled_loss = model(ux, lx=lx, target=target, alpha=alpha)
#         step_loss = labeled_loss.mean() + unlabeled_loss.mean()
#         optim.zero_grad()
#         with amp.scale_loss(step_loss, optim) as scaled_loss:
#             scaled_loss.backward()
#         optim.step()
#         loss['m2_loss'] += step_loss.item()
#     print(f'epoch: {epoch} -', ', '.join([f'{k}: {v:.3f}' for k, v in loss.items()]))
#
#     for k, v in loss.items():
#         stats[k].append([epoch, v])
#
#     if epoch % FLAGS.eval_step == 0:
#
#         if not os.path.exists(outdir):
#             os.mkdir(outdir)
#
#         features = torch.Tensor()
#         target_stack = np.array([]).astype(np.integer)
#         pred_stack = np.array([]).astype(np.integer)
#         acc = 0
#         N = 0
#
#         model.eval()
#         with torch.no_grad():
#             for step, (x, target) in enumerate(test_loader):
#                 x = x.to(device)
#                 out_params = model(x, return_params=True)
#                 features = torch.cat((features, out_params['z'].cpu()), 0)
#                 target_stack = np.append(target_stack, list(target.numpy()))
#                 y_pred = out_params['y_pred'].cpu()
#                 acc += torch.nonzero(target==y_pred).size(0)
#                 N += y_pred.size(0)
#                 pred_stack = np.append(pred_stack, list(y_pred.numpy()))
#         features = features.numpy()
#         acc = acc / N * 100
#         print(f'acc = {acc:.3f} at epoch {epoch}')
#         stats['test_acc'].append([epoch, acc])
#
#         if features.shape[1] > 2:
#             features = decomposition.TSNE().fit_transform(features)
#         if features.shape[1] != 2:
#             raise ValueError('features can not visualize')
#
#         trues = _to_values(target_stack, targets_dict)
#         preds = _to_values(pred_stack, targets_dict)
#
#         plot.scatter(features[:, 0], features[:, 1], trues, targets_dict.values(),
#                      out=f'{outdir}/latent_true_{epoch}.png',
#                      title=f'latent features at epoch {epoch} - true label')
#         plot.scatter(features[:, 0], features[:, 1], preds, targets_dict.values(),
#                      out=f'{outdir}/latent_pred_{epoch}.png',
#                      title=f'latent features at epoch {epoch} - pred label')
#         cm, cm_xlabels, cm_ylabels = metrics.confusion_matrix(
#                     pred_stack, target_stack, targets_dict.keys(), targets_dict.keys(), return_labels=True)
#         cm_figsize = (len(cm_xlabels) / 1.2, len(cm_ylabels) / 1.5)
#         cm_ylabels = _to_values(cm_ylabels, targets_dict)
#         plot.plot_confusion_matrix(cm, cm_xlabels, cm_ylabels,
#                                    out=f'{outdir}/cm_{epoch}.png',
#                                    xlabel='predicted', ylabel='true',
#                                    figsize=cm_figsize)
#         for k, v in stats.items():
#             xx = np.array(v)[:, 0]
#             plot.plot(v, out=f'{outdir}/{k}_{epoch}.png', title=f'{k}', xmin=min(xx), xmax=max(xx), xlabel='epoch', ylabel=k)
#
#         weights = model.get_layer("conv1").get_weights()[0]
