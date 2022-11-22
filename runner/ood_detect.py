# Copyright 2022 Luping Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import math
import time
import copy
import faiss
import random

import torch as th
import numpy as np
from einops import rearrange
import torchvision.utils as tvu
import torch.utils.data as data
import torch.distributed as dist
import torchvision.transforms.functional as F
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from dataset import get_dataset
from runner.runner import Runner
from network.ood.resnet18 import ResNet18_32x32
from network.ood.resnet50 import ResNet50
from detect.dataset.imglist_dataset import ImglistDataset


@th.no_grad()
class KNN(object):
    def __init__(self, dim, net=None):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.net = net
        self.y = None

    def encoder(self, y: th.Tensor):
        normalizer = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)

        if self.net is not None:
            y = th.clamp(y * 0.5 + 0.5, 0, 1)  # 设置标准输入！
            _, index = self.net(y, return_feature=True)
            index = index.cpu().numpy()
            # print(index.shape)
        else:
            index = y.mean(dim=1).cpu().numpy().reshape(len(y), -1)

        index = normalizer(index)
        assert index.shape[1] == self.dim

        return index

    def add(self, y: th.Tensor):
        index = self.encoder(y)

        self.index.add(index)
        self.y = th.cat([self.y, y], dim=0) if self.y is not None else y
        print('the shape of y in KNN', y.shape)

    def search(self, y: th.Tensor, k=1, return_y=False):
        index = self.encoder(y)

        if return_y:
            loss, ind = self.index.search(index, k)
            y = rearrange(self.y[ind.reshape(-1)], '(b1 b2) ... -> b1 b2 ...', b2=k)  # todo check
            return loss, ind, y
        else:
            return self.index.search(index, k)


class OodDetection(Runner):
    def __init__(self, args, config, schedule, model):
        super(OodDetection, self).__init__(args, config, schedule, model)
        # self-train version
        self.discriminator = ResNet18_32x32(num_classes=10).to(self.device)
        state_dict = th.load('temp/model/ood_cifar10_res18.ckpt', map_location=self.device)
        try:
            self.discriminator.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            self.discriminator.load_state_dict(new_state_dict, strict=True)
        self.discriminator.eval()  # 兄弟，要eval啊！

    @th.no_grad()
    def noise_encoder(self):
        """
        generate image representation
        """
        model = self.model
        schedule = self.schedule
        device = self.device
        continuous = False

        seq, skip, train_loader = self.before_sample()

        image_list, noise_list = [], []

        def gather(obj):
            if self.world_size >= 2:
                obj = obj.cuda()
                obj_gather = [th.zeros_like(obj) for _ in range(self.world_size)]
                dist.all_gather(obj_gather, obj)
                obj = th.cat(obj_gather, dim=0).cpu()
            return obj

        for (img, y) in tqdm(train_loader, disable=self.rank == self.world_size - 1):
            img = img.to(device) * 2 - 1
            noise_repr = schedule.multi_iteration(img, -1, 980 - 1, model,
                                                  last=True, fresh=True, continuous=continuous)

            image_list.append(img.cpu())
            noise_list.append(noise_repr.cpu())

        image_list = th.cat(image_list, dim=0)
        noise_list = th.cat(noise_list, dim=0)
        if self.world_size >= 2:
            image_list = gather(image_list).numpy()
            noise_list = gather(noise_list).numpy()

        if self.rank == 0:
            print(image_list.shape, noise_list.shape)
            # np.save(f'{self.args.image_path}/{self.args.category}_{self.args.category_value}.npy', noise)
            np.save(f'temp/noise/{self.args.model_name}_img_1.npy', image_list)
            np.save(f'temp/noise/{self.args.model_name}_noise_1.npy', noise_list)

        # for k in tqdm(range(4, 50, 5), desc='gen_edit'):
        #     t = seq[k] * th.tensor([1] * repeat_size).to(device)
        #     # print(img.shape, t.shape)
        #     img_n, _, _ = schedule.diffusion(img, t, noise=noise)
        #
        #     noise_r = schedule.multi_iteration(img_n, k * skip - 1, 49 * skip - 1, model,
        #                                        last=True, fresh=True, continuous=continuous)
        #     img_r = schedule.multi_iteration(img_n, k * skip - 1, 0 * skip - 1, model,
        #                                      last=True, fresh=True, continuous=continuous)
        #
        #     img_r = th.clamp(img_r * 0.5 + 0.5, 0, 1)
        #     noise_r = th.clamp(noise_r * 0.5 + 0.5, 0, 1)
        #     for i in range(repeat_size):
        #         tvu.save_image(img_r[i], os.path.join(self.args.image_path,
        #                                               f"img-{i + 1}-{k}.png"))
        #         tvu.save_image(noise_r[i], os.path.join(self.args.image_path,
        #                                                 f"noise-{i + 1}-{k}.png"))

    @th.no_grad()
    def enhancement(self):
        """
        test noise enhancement
        """
        test_size = 2

        model = self.model
        schedule = self.schedule
        device = self.device
        continuous = False

        seq, skip, train_loader = self.before_sample()

        cat_np = []
        for (img, y) in train_loader:
            for i in range(len(y)):
                if y[i] == 3:
                    cat_np.append(img[i].numpy())
            break

        np.save('temp/cat_np.npy', cat_np)
        # img = img[:test_size].to(device) * 2 - 1
        #
        # noise = schedule.multi_iteration(img, - 1, 49 * skip - 1, model,
        #                                  last=True, fresh=True, continuous=continuous)
        #
        # noise = transforms.RandomHorizontalFlip(p=1)(noise)
        # img_r = schedule.multi_iteration(noise, 49 * skip - 1, - 1, model,
        #                                  last=True, fresh=True, continuous=continuous)
        #
        # img = th.clamp(img * 0.5 + 0.5, 0, 1)
        # img_r = th.clamp(img_r * 0.5 + 0.5, 0, 1)
        # for i in range(test_size):
        #     tvu.save_image(img[i], os.path.join(self.args.image_path,
        #                                         f"img-{i + 1}.png"))
        #     tvu.save_image(img_r[i], os.path.join(self.args.image_path,
        #                                           f"img_r-{i + 1}.png"))
        #
        # break

    @th.no_grad()
    def interp_detect(self):
        batch_size = 250
        iter_size = 4
        iter_size = iter_size // self.world_size if self.world_size >= 2 else iter_size
        repeat_size = self.args.repeat_size
        knn_num = self.args.debug_value
        id_name = self.args.model_name

        model = self.model
        schedule = self.schedule
        device = self.device
        continuous = False

        # load model
        self.before_sample()

        def gather(obj):
            if self.world_size >= 2:
                obj = obj.cuda()
                obj_gather = [th.zeros_like(obj) for _ in range(self.world_size)]
                dist.all_gather(obj_gather, obj)
                obj = th.cat(obj_gather, dim=0).cpu()
            return obj

        # load & process dataset
        ood_list, ood_dict = ['svhn', 'texture', 'places365', 'cifar100', 'cifar10', 'tin'], {}
        # ood_list, ood_dict = ['imagenet', 'inaturalist', 'openimage_o', 'imagenet_o', 'species'], {}

        for ood_name in ood_list:
            dataset = ImglistDataset(id_name, 'test', 32,
                                     f'./data/benchmark_imglist/{id_name}/test_{ood_name}.txt',
                                     f'./data/images_classic/')
            if self.world_size >= 2:
                sampler = data.distributed.DistributedSampler(dataset)
                ood_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                             num_workers=4)
            else:
                ood_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=4)
            # print(len(dataset))

            norm_dict = {'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
                         'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
                         'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]], }
            mean, std = norm_dict[id_name]
            norm_fn = transforms.Normalize(mean=mean, std=std)

            imgr_np, fea_np, out_np = [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)]
            for i, output in enumerate(tqdm(ood_loader, total=iter_size, disable=self.rank + 1 - self.world_size,
                                            desc=f'process {ood_name} data')):
                if i == iter_size:
                    for j in range(len(imgr_np)):
                        imgr_np[j] = th.cat(imgr_np[j], dim=0)
                        fea_np[j] = th.cat(fea_np[j], dim=0)
                        out_np[j] = th.cat(out_np[j], dim=0)
                        imgr_np[j] = gather(imgr_np[j]).numpy()
                        fea_np[j] = gather(fea_np[j]).numpy()
                        out_np[j] = gather(out_np[j]).numpy()

                    np.savez(f'temp/sample_ood/{id_name}_{ood_name}_knn{knn_num}_{repeat_size}.npz', imgr=imgr_np,
                             fea=fea_np, out=out_np)
                    break

                img = output['data']
                ood_img = img.to(device) * 2 - 1

                # loss, ind = index.search(ood_img, repeat_size+1)
                # noise = index.y[ind[:, -1].reshape(-1)]
                # noise = th.randn_like(ood_img)
                noise_list = [th.randn_like(ood_img) for i in range(repeat_size)]
                out_ = self.discriminator(norm_fn(img.cuda()), return_feature=False)
                score = th.softmax(out_, dim=1)
                _, y_pred = th.max(score, dim=1)

                tq = tqdm(total=8 * repeat_size, leave=False, desc='subprocess',
                          disable=self.rank + 1 - self.world_size)
                for j, t in enumerate([0, 240]):  # 1000
                    imgr_rp, fea_rp, out_rp = [], [], []
                    for k in range(repeat_size):
                        tq.update()

                        img_n, _, _ = schedule.diffusion(ood_img, th.ones(batch_size, device=device, dtype=th.long) * t,
                                                         noise=noise_list[k].cuda())
                        # img_n = slerp(noise1, noise, t / 1000.0)
                        img_r = schedule.multi_iteration(img_n, t - 1, -1, model, y=y_pred,
                                                         class_num=self.config['Train']['num_classes'],
                                                         last=True, fresh=True, continuous=continuous)
                        img_r = th.clamp(img_r * 0.5 + 0.5, 0, 1)
                        img_r = norm_fn(img_r)
                        logit, feature = self.discriminator(img_r.cuda(), return_feature=True)

                        imgr_rp.append(img_r.cpu())
                        out_rp.append(logit.cpu())
                        fea_rp.append(feature.cpu())

                    imgr = rearrange(th.stack(imgr_rp, dim=1), 'b r ... -> (b r) ...')
                    out = rearrange(th.stack(out_rp, dim=1), 'b r ... -> (b r) ...')
                    fea = rearrange(th.stack(fea_rp, dim=1), 'b r ... -> (b r) ...')
                    imgr_np[j].append(imgr)
                    fea_np[j].append(fea)  # 这种写反的错误都能犯？？？
                    out_np[j].append(out)
                tq.close()
