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
    def __init__(self, args, config, schedule, model, discriminator):
        super(OodDetection, self).__init__(args, config, schedule, model)
        # self-train version
        self.discriminator = discriminator
        state_dict = th.load(self.args.disc_path, map_location=self.device)
        self.discriminator.load_state_dict(state_dict, strict=True)
        self.discriminator.eval()  # 兄弟，要eval啊！

    @th.no_grad()
    def diff_detect(self):
        batch_size = 250
        iter_size = 4
        iter_size = iter_size // self.world_size if self.world_size >= 2 else iter_size
        repeat_size = self.args.repeat_size

        id_name = self.config['Dataset']['name'].lower()
        timestep_list = [0, 120, 240, 360]
        ood_list = ['cifar10', 'cifar100', 'tin', 'svhn', 'texture', 'places365']
        # ood_list = ['imagenet', 'inaturalist', 'openimage_o', 'imagenet_o', 'species']

        model = self.model
        schedule = self.schedule
        device = self.device

        def gather(obj):
            if self.world_size >= 2:
                obj_ = obj.cuda()
                if self.rank == 0:
                    obj_gather = [th.zeros_like(obj_) for _ in range(self.world_size)]
                    dist.gather(obj_, obj_gather)
                    obj = th.cat(obj_gather, dim=0).cpu()
                else:
                    dist.gather(obj_)
            return obj

        # load model
        self.before_sample()

        # load dataset
        for ood_name in ood_list:
            dataset = ImglistDataset(id_name, 'test', 32,
                                     f'./benchmark_imglist/{id_name}/test_{ood_name}.txt',
                                     f'./images_classic/')
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

            # main detection process
            tn = len(timestep_list)
            num_classes = self.config['Model']['num_classes']
            imgr_np, feature_np, logit_np = [[] for _ in range(tn)], [[] for _ in range(tn)], [[] for _ in range(tn)]
            for i, output in enumerate(tqdm(ood_loader, total=iter_size, disable=self.rank + 1 - self.world_size,
                                            desc=f'process {ood_name} data')):

                if i == iter_size:
                    for j in range(len(imgr_np)):
                        imgr_np[j] = th.cat(imgr_np[j], dim=0)
                        feature_np[j] = th.cat(feature_np[j], dim=0)
                        logit_np[j] = th.cat(logit_np[j], dim=0)
                        imgr_np[j] = gather(imgr_np[j]).numpy()
                        feature_np[j] = gather(feature_np[j]).numpy()
                        logit_np[j] = gather(logit_np[j]).numpy()

                    if self.rank == 0:
                        np.savez(f'temp/sample/{id_name}_{ood_name}_r{repeat_size}.npz', imgr=imgr_np,
                                 feature=feature_np, logit=logit_np)
                    break

                img = output['data']
                ood_img = img.to(device) * 2 - 1

                # # generate random noise
                # loss, ind = index.search(ood_img, repeat_size+1)
                # noise = index.y[ind[:, -1].reshape(-1)]
                # noise = th.randn_like(ood_img)
                noise_list = [th.randn_like(ood_img) for _ in range(repeat_size)]
                if num_classes > 0:
                    out_ = self.discriminator(norm_fn(img.cuda()), return_feature=False)
                    score = th.softmax(out_, dim=1)
                    _, y_pred = th.max(score, dim=1)
                else:
                    y_pred = None

                tq = tqdm(total=tn * repeat_size, leave=False, desc='subprocess',
                          disable=self.rank + 1 - self.world_size)
                for j, t in enumerate(timestep_list):
                    imgr_list, feature_list, logit_list = [], [], []
                    for k in range(repeat_size):
                        img_n, _, _ = schedule.diffusion(ood_img, th.ones(batch_size, device=device, dtype=th.long) * t,
                                                         noise=noise_list[k].cuda())
                        # img_n = slerp(noise1, noise, t / 1000.0)
                        if num_classes > 0:
                            img_r = schedule.multi_iteration(img_n, t - 1, -1, model,
                                                             y=y_pred, num_classes=num_classes, beta=2,
                                                             last=True, fresh=True)
                        else:
                            img_r = schedule.multi_iteration(img_n, t - 1, -1, model,
                                                             last=True, fresh=True)

                        img_r = th.clamp(img_r * 0.5 + 0.5, 0, 1)
                        img_r_ = norm_fn(img_r)
                        logit, feature = self.discriminator(img_r_.cuda(), return_feature=True)

                        imgr_list.append(img_r.cpu())
                        feature_list.append(feature.cpu())
                        logit_list.append(logit.cpu())

                        tq.update()

                    imgr_ = rearrange(th.stack(imgr_list, dim=1), 'b r ... -> (b r) ...')
                    feature_ = rearrange(th.stack(feature_list, dim=1), 'b r ... -> (b r) ...')
                    logit_ = rearrange(th.stack(logit_list, dim=1), 'b r ... -> (b r) ...')
                    imgr_np[j].append(imgr_)
                    feature_np[j].append(feature_)  # 这种写反的错误都能犯？？？
                    logit_np[j].append(logit_)

                tq.close()

    @th.no_grad()
    def interpolation(self):
        """
        test image interpolation
        """
        batch_size = 16

        model = self.model
        schedule = self.schedule
        device = self.device

        seq, skip, train_loader = self.before_sample()

        def slerp(z1, z2, alpha):
            theta = th.acos(th.sum(z1 * z2) / (th.norm(z1) * th.norm(z2)))
            return (th.sin((1 - alpha) * theta) / th.sin(theta) * z1
                    + th.sin(alpha * theta) / th.sin(theta) * z2)

        img1, img2 = None, None
        for img, y in train_loader:
            img1 = img[:batch_size].to(device) * 2 - 1
            img2 = img[batch_size:batch_size * 2].to(device) * 2 - 1
            break

        img1_ = th.clamp(img1 * 0.5 + 0.5, 0, 1)
        img2_ = th.clamp(img2 * 0.5 + 0.5, 0, 1)
        for i in range(batch_size):
            tvu.save_image(img1_[i], os.path.join(self.args.image_path,
                                                  f"img100-{i + 1}.png"))
            tvu.save_image(img2_[i], os.path.join(self.args.image_path,
                                                  f"img200-{i + 1}.png"))

        noise1 = schedule.multi_iteration(img1, - 1, 49 * skip - 1, model,
                                          last=True, fresh=True).to(device)
        noise2 = schedule.multi_iteration(img2, - 1, 49 * skip - 1, model,
                                          last=True, fresh=True).to(device)

        timestep_list = list(range(5, 50, 5))

        for k in tqdm(timestep_list, desc='gen_edit1'):
            t = seq[k] * th.tensor([1] * batch_size).to(device)
            img_n, _, _ = schedule.diffusion(img1, t, noise=noise2)

            # noise_r = schedule.multi_iteration(img_n, k * skip - 1, 49 * skip - 1, model,
            #                                    last=True, fresh=True, continuous=continuous)
            img_r = schedule.multi_iteration(img_n, k * skip - 1, 0 * skip - 1, model,
                                             last=True, fresh=True)

            img_r = th.clamp(img_r * 0.5 + 0.5, 0, 1)
            # noise_r = th.clamp(noise_r * 0.5 + 0.5, 0, 1)
            for i in range(batch_size):
                tvu.save_image(img_r[i], os.path.join(self.args.image_path,
                                                      f"img1-{i + 1}-{k}.png"))
                # tvu.save_image(noise_r[i], os.path.join(self.args.image_path,
                #                                         f"noise-{i + 1}-{k}.png"))

        for k in tqdm(timestep_list, desc='gen_edit2'):
            t = seq[k] * th.tensor([1] * batch_size).to(device)
            img_n, _, _ = schedule.diffusion(img2, t, noise=noise1)

            # noise_r = schedule.multi_iteration(img_n, k * skip - 1, 49 * skip - 1, model,
            #                                    last=True, fresh=True, continuous=continuous)
            img_r = schedule.multi_iteration(img_n, k * skip - 1, 0 * skip - 1, model,
                                             last=True, fresh=True)

            img_r = th.clamp(img_r * 0.5 + 0.5, 0, 1)
            # noise_r = th.clamp(noise_r * 0.5 + 0.5, 0, 1)
            for i in range(batch_size):
                tvu.save_image(img_r[i], os.path.join(self.args.image_path,
                                                      f"img2-{i + 1}-{k}.png"))
                # tvu.save_image(noise_r[i], os.path.join(self.args.image_path,
                #                                         f"noise-{i + 1}-{k}.png"))

        for k in tqdm(timestep_list, desc='gen_edit3'):
            noise = slerp(noise1, noise2, k / 50.0)

            # noise_r = schedule.multi_iteration(img_n, k * skip - 1, 49 * skip - 1, model,
            #                                    last=True, fresh=True, continuous=continuous)
            img_r = schedule.multi_iteration(noise, 49 * skip - 1, - 1, model,
                                             last=True, fresh=True)

            img_r = th.clamp(img_r * 0.5 + 0.5, 0, 1)
            # noise_r = th.clamp(noise_r * 0.5 + 0.5, 0, 1)
            for i in range(batch_size):
                tvu.save_image(img_r[i], os.path.join(self.args.image_path,
                                                      f"img3-{i + 1}-{k}.png"))
                # tvu.save_image(noise_r[i], os.path.join(self.args.image_path,
                #                                         f"noise-{i + 1}-{k}.png"))
