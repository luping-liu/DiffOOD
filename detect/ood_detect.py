import numpy as np
import torch as th
import argparse
import sys

from einops import rearrange
from scipy.stats import kstest, ks_2samp
from sklearn.metrics import roc_auc_score
from tabulate import tabulate


def auroc(id_score, ood_score, reverse=False):
    y = [0] * len(id_score) + [1] * len(ood_score)
    if reverse:
        y = list(reversed(y))

    scores = th.cat([id_score, ood_score], dim=0)
    result = roc_auc_score(y, scores)

    return result


normalizer = lambda x: x / (th.norm(x, dim=-1, keepdim=True) + 1e-10)
# normalizer = lambda x: x

parser = argparse.ArgumentParser()
parser.add_argument('--repeat_size', '-r', type=int, default=4)
parser.add_argument('--space', '-s', type=str, default='logit')
parser.add_argument('--id_name', '-n', type=str, default='cifar10')
args = parser.parse_args()
id_name = args.id_name
repeat_size = args.repeat_size
detect_space = args.space

in_data = np.load(f'temp/sample/{id_name}_{id_name}_r{repeat_size}.npz')
# ood_list = ['SVHN', 'CELEBA', 'CIFAR100', 'CIFAR10T']
ood_list = ['cifar10', 'cifar100', 'tin', 'svhn', 'texture', 'places365']
# ood_list = ['imagenet', 'inaturalist', 'openimage_o', 'imagenet_o', 'species']
ood_data = {ood_name: np.load(f'temp/sample/{id_name}_{ood_name}_r{repeat_size}.npz')
            for ood_name in ood_list if ood_name != id_name}

in_fea = th.from_numpy(in_data[detect_space])
in_fea = normalizer(in_fea.view(*in_fea.shape[:2], -1))
in_fea = rearrange(in_fea, 'm (b r) ... ->m b r ...', r=repeat_size)
print(in_fea.shape)

# ind = in_fea[0].max(dim=1)[1].view(-1, 1)
# in_fea = in_fea.gather(2, ind)
width_num = len(in_fea) - 1
# print('CIFAR10', in_imgr.shape)
in_imgr_diff = [(in_fea[i] - in_fea[i+1]).abs().sum(dim=(-1)).mean(dim=1) for i in range(len(in_fea) - 1)]
in_imgr_diff += [(in_fea[0] - in_fea[i+1]).abs().sum(dim=(-1)).mean(dim=1) for i in range(len(in_fea) - 1)]
# in_imgr_diff = [(in_fea[i].gather(1, ind) - in_fea[i+1].gather(1, ind)).view(-1).abs() for i in range(len(in_fea) - 1)]
# in_imgr_diff += [(in_fea[0].gather(1, ind) - in_fea[i+1].gather(1, ind)).view(-1).abs() for i in range(len(in_fea) - 1)]
output_tab = []
avg_auroc = [[] for _ in range(min(8, width_num))]
for name in ood_list:
    if name == id_name:
        continue
    ood_fea = th.from_numpy(ood_data[name][detect_space])
    ood_fea = normalizer(ood_fea.view(*ood_fea.shape[:2], -1))
    ood_fea = rearrange(ood_fea, 'm (b r) ... ->m b r ...', r=repeat_size)
    # ind = ood_fea[0].max(dim=1)[1].view(-1, 1)
    # print(name, ood_imgr.shape)
    output_tab.append([name])
    ood_imgr_diff = [(ood_fea[i] - ood_fea[i + 1]).abs().sum(dim=(-1)).mean(dim=1) for i in range(len(ood_fea) - 1)]
    ood_imgr_diff += [(ood_fea[0] - ood_fea[i + 1]).abs().sum(dim=(-1)).mean(dim=1) for i in range(len(ood_fea) - 1)]
    # ood_imgr_diff = [(ood_fea[i].gather(1, ind) - ood_fea[i + 1].gather(1, ind)).view(-1).abs() for i in range(len(ood_fea) - 1)]
    # ood_imgr_diff += [(ood_fea[0].gather(1, ind) - ood_fea[i + 1].gather(1, ind)).view(-1).abs() for i in range(len(ood_fea) - 1)]

    # print(in_imgr_diff[-1], ood_imgr_diff[-1])
    output_tab[-1].append('%.2f' % auroc(in_imgr_diff[-1], ood_imgr_diff[-1]))

    for i in range(min(8, width_num)):
        in_score = in_imgr_diff[i]
        ood_score = ood_imgr_diff[i]
        auroc_l = auroc(in_score, ood_score)

        in_score = in_imgr_diff[i + width_num]
        ood_score = ood_imgr_diff[i + width_num]
        auroc_g = auroc(in_score, ood_score)

        output_tab[-1].append('%.2f/%.2f' % (auroc_l, auroc_g * 100))
        avg_auroc[i].append(auroc_g * 100)
        # print(i, auroc(in_score, ood_score), end=' '

output_tab.append(['avg', ''])
for item in avg_auroc:
    output_tab[-1].append('%.2f' % (np.mean(item)))

print(tabulate(output_tab, ['name', 'base'] + list(range(1, 9)), tablefmt='github'))

# in_loss = th.from_numpy(in_data['loss'])[:, 1:]
# print(in_loss.shape)
# for name in ood_list:
#     ood_loss = th.from_numpy(ood_data[name]['loss'])
#     # print(name, auroc(in_loss[-1], ood_loss[-1]))
#     for i in range(in_loss.shape[1]):
#         print(i, '%.2f' % auroc(in_loss[:, i], ood_loss[:, i]), end=' ')
#     print()

