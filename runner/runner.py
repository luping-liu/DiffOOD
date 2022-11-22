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
import random

import torch as th
import numpy as np
import torch.optim as optimi
import torch.utils.data as data
import torchvision.utils as tvu
import torch.utils.tensorboard as tb
import torch.distributed as dist
import torchvision.transforms.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
# from scipy import integrate
# from torchdiffeq import odeint
from tqdm.auto import tqdm

from dataset import get_dataset
from network.ema import EMAHelper


class Runner(object):
    def __init__(self, args, config, schedule, model, model_1=None):
        self.args = args
        self.config = config
        self.diffusion_step = config['Schedule']['diffusion_step']
        self.sample_speed = args.sample_speed
        self.device = th.device(args.device)
        rank = os.environ.get("LOCAL_RANK")
        self.rank = 0 if rank is None else int(rank)
        world_size = os.environ.get("WORLD_SIZE")
        self.world_size = 1 if world_size is None else int(world_size)

        self.schedule = schedule
        self.model = model
        self.model_1 = model_1
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.tb_logger = None

    def get_optim(self, params, config):
        if config['optimizer'] == 'adam':
            optimizer = optimi.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'],
                                    betas=(config['beta1'], 0.999), amsgrad=config['amsgrad'],
                                    eps=config['eps'])
        elif config['optimizer'] == 'adamw':
            optimizer = optimi.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'],
                                     betas=(config['beta1'], 0.999), amsgrad=config['amsgrad'],
                                     eps=config['eps'])
        elif config['optimizer'] == 'sgd':
            optimizer = optimi.SGD(params, lr=config['lr'], momentum=0.9)
        else:
            optimizer = None

        def scheduler_(step):
            if step < config['warmup']:
                return step / config['warmup']
            else:
                if config['lr_decay']:
                    return config['lr_gamma'] ** ((step - config['warmup']) // config['lr_step'])
                else:
                    return 1

        scheduler = optimi.lr_scheduler.LambdaLR(optimizer, scheduler_)

        return optimizer, scheduler

    def get_data(self, args, config):
        dataset, _ = get_dataset(self.args, config)
        world_size = self.world_size

        if world_size >= 2:
            sampler = data.distributed.DistributedSampler(dataset)
            train_loader = data.DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler,
                                           num_workers=config['num_workers'])
        else:
            sampler = None
            train_loader = data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                           num_workers=config['num_workers'])

        return sampler, train_loader

    def get_loss(self, config):
        if config == 'linear':
            return lambda a, b: (a - b).abs().sum(dim=(1, 2, 3)).mean(dim=0)
            # def loss_fn(a, b):
            #     return (a - b).abs().sum(dim=(1, 2, 3)).mean(dim=0)
        elif config == 'square':
            return lambda a, b: (a - b).square().sum(dim=(1, 2, 3)).mean(dim=0)
        else:
            return None

    # def gen_t(self, config, size):
    #     n = size
    #     # diffusion_span = self.diffusion_step // self.sample_speed
    #
    #     if config['t_type'] == 'symmetry':
    #         t = th.randint(low=0, high=self.diffusion_step, size=[n // 2 + 1])
    #         t = th.cat([t, self.diffusion_step - t - 1], dim=0)[:n].to(self.device)
    #     elif config['t_type'] == 'split':
    #         t = th.randint(low=0, high=self.diffusion_step // 2, size=[n // 2])
    #         t = th.cat([t, self.diffusion_step - t - 1], dim=0).to(self.device)
    #     # elif config['t_type'] == 'triangle':
    #     #     t0_ = list(range(self.diffusion_step))
    #     #     t0 = random.choices(list(reversed(t0_)), weights=t0_, k=n // 2)
    #     #     t = t0 = th.tensor(t0, device=self.device)
    #     # elif config['t_type'] == 'ladder':
    #     #     t = th.randint(low=0, high=self.diffusion_step - diffusion_span, size=[n], device=self.device)
    #     else:
    #         t = None
    #
    #     return t

    def before_train(self):
        pass

    def before_train_(self):
        self.before_train()

        rank = self.rank
        world_size = self.world_size

        if world_size >= 2:
            th.cuda.set_device(rank)
            dist.init_process_group(backend='nccl')
            # self.model.load_state_dict(th.load(self.args.model_path, map_location=self.device), strict=True)
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        # if self.args.teacher:
        # self.teacher.load_state_dict(th.load(self.args.model_path, map_location=self.device), strict=True)
        self.model.train()

        if self.optimizer is None:
            self.optimizer, self.scheduler = self.get_optim(self.model.parameters(), self.config['Optim'])

        config = self.config['Train']
        if config['ema'] and rank == 0:
            self.ema = EMAHelper(mu=config['ema_rate'])
            model_ = self.model.module if world_size >= 2 else self.model
            self.ema.register(model_)

        if self.args.restart:
            train_state = th.load(os.path.join(self.args.train_path, 'train.ckpt'), map_location=self.device)
            self.model.load_state_dict(train_state[0])
            self.optimizer.load_state_dict(train_state[1])
            self.scheduler.load_state_dict(train_state[2])
            epoch, step, time_start = train_state[3:6]
            if self.ema is not None:
                ema_state = th.load(os.path.join(self.args.train_path, 'ema.ckpt'), map_location=self.device)
                self.ema.load_state_dict(ema_state)
        elif self.args.reinitialize is not None:
            train_state = th.load(os.path.join(self.args.train_path, 'train_init.ckpt'), map_location=self.device)
            self.model.load_state_dict(train_state[0])
            if self.ema is not None:
                ema_state = th.load(os.path.join(self.args.train_path, 'ema_init.ckpt'), map_location=self.device)
                self.ema.load_state_dict(ema_state)
            epoch, step = 0, 0
            time_start = time.strftime('%m%d-%H%M')
        else:
            epoch, step = 0, 0
            time_start = time.strftime('%m%d-%H%M')

        if rank == 0:
            # print(f"{time_start}:{self.args.config.split('.')[0]}")
            os.makedirs(self.args.train_path, exist_ok=True)
            with open(f'{self.args.train_path}/info.txt', 'a') as f:
                f.write(f"{time.strftime('%m%d-%H%M')}, {step}:\n")
                f.write(f"{time_start}:{self.args.config.split('.')[0]}\n")

        sampler, train_loader = self.get_data(self.args, self.config['Dataset'])

        if rank == 0:
            # self.tb_logger = tb.SummaryWriter(f'temp/tensorboard/dev')
            self.tb_logger = tb.SummaryWriter(f"temp/tensorboard/{time_start}:{self.args.config.split('.')[0]}")
            print(f"tensorboard: {time_start}:{self.args.config.split('.')[0]}")

        return (epoch, step, time_start), (sampler, train_loader)

    def train(self, train_loader, model, loss_fn, config):
        model.train()

        n = None
        for i, (img, y) in enumerate(train_loader):
            n = img.shape[0] if n is None else n
            if img.shape[0] != n:
                # print("Error, img.shape[0] != config['batch_size']")
                break

            img = img.to(self.device) * 2 - 1
            y = y.to(self.device)
            ind, replace = th.rand(y.shape, device=y.device), th.ones_like(y) * config['num_classes']
            y = th.where(ind < 0.9, y, replace)

            if config['t_type'] == 'symmetry':
                t = th.randint(low=0, high=config['train_step'], size=(math.ceil(n * 0.5),))
                t = th.cat([t, config['train_step'] - t - 1], dim=0)[:n].to(self.device)
            else:
                t = None

            img_n, t, noise = self.schedule.diffusion(img, t)

            noise_p = model(img_n, t, y=y)
            loss = loss_fn(noise_p, noise)

            yield {'loss': loss, 't': t}

    @th.no_grad()
    def valid(self, img_valid, config):
        img_valid, y_valid = img_valid
        if config['iter_type'] == 'diffusion':
            img_input = img_valid[:8].to(self.device) * 2 - 1
            t = th.ones(8, device=self.device, dtype=th.long) * config['train_step'] - 20
            img_input, _, _ = self.schedule.diffusion(img_input, t)
        else:
            img_input = None

        img_p = self.schedule.multi_iteration(img_input, config['train_step'] - 21, -1, self.model, y=y_valid[:8],
                                              class_num=config['num_classes'], last=True, fresh=True, continuous=False)
        # noise_r = schedule.multi_iteration(img_p, 18, 998, self.model, last=True, fresh=True, continuous=False)
        #
        # dis1 = th.abs(noise_r - noise0.to('cpu')).sum(dim=(1, 2, 3)).mean(dim=0)

        # noise = schedule.multi_iteration(img_c, -1, 980 - 1, self.model, last=True, fresh=True, continuous=False)
        # img_r = schedule.multi_iteration(noise, 980 - 1, -1, self.model, last=True, fresh=True, continuous=False)
        #
        # dis_i1 = th.abs(img_r - img_c.to('cpu')).sum(dim=(1, 2, 3)).mean(dim=0)

        # return {"dis_n": dis1, "dis_i": dis_i1, }, \
        #        {"img_c": img_c, "noise": noise, "img_r": img_r, "img_p": img_p, }
        return {}, {"img_p": img_p}

    def logger(self, tb_logger, step, scalars, images=None):
        if step % 10 == 0:
            for name in scalars:
                tb_logger.add_scalar(name, scalars[name], global_step=step)
            if images is not None:
                for name in images:
                    images[name] = th.clamp(images[name] * 0.5 + 0.5, 0.0, 1.0)
                    tb_logger.add_images(name, images[name], global_step=step)
        if step % 50 == 0:
            print(f'{step}:', end=' ')
            for name in scalars:
                print(name, '%.5g' % scalars[name], end=' ')
            print()

    def train_loop(self):  # todo 改名
        start, loader = self.before_train_()
        epoch_, step, time_start = start
        sampler, train_loader = loader
        valid_loader_iter = iter(train_loader)

        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        ema = self.ema
        tb_logger = self.tb_logger
        rank, world_size = self.rank, self.world_size

        config = self.config['Train']
        loss_fn = self.get_loss(config['loss_type'])

        for epoch in range(epoch_, config['epoch']):
            if sampler is not None:
                sampler.set_epoch(epoch)

            train_iter = self.train(train_loader, model, loss_fn, config)

            accum_num = config['accum_num']
            start = time.time()
            for k, results in enumerate(train_iter):

                loss = results['loss'] / accum_num
                loss.backward()

                th.nn.utils.clip_grad_norm_(model.parameters(), self.config['Optim']['grad_clip'])
                # try:
                #     th.nn.utils.clip_grad_norm_(model.parameters(), self.config['Optim']['grad_clip'])
                # except Exception:
                #     pass
                if (k + 1) % accum_num == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    time_cost = time.time() - start
                    start = time.time()
                else:
                    continue

                if rank != 0:
                    continue

                if ema is not None:
                    model_ = model.module if world_size >= 2 else model
                    ema.update(model_)

                scalars = {'loss/loss': results['loss'].item(),
                           'train/t': results['t'].float().mean().item(), 'train/lr': scheduler.get_last_lr()[0],
                           'train/cost': time_cost}
                images = {}

                if step % 200 == 0:
                    try:  # todo 改为真验证集
                        img_valid = next(valid_loader_iter)
                    except StopIteration:
                        valid_loader_iter = iter(train_loader)
                        img_valid = next(valid_loader_iter)

                    model.eval()
                    scalars_, images_ = self.valid(img_valid, config)
                    model.train()
                    for name in scalars_:
                        scalars['valid/' + name] = scalars_[name]
                    for name in images_:
                        images['valid/' + name] = images_[name]

                if step % 2500 == 0:
                    train_state = [model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                                   epoch, step, time_start]
                    th.save(train_state, os.path.join(self.args.train_path, 'train.ckpt'))
                    # th.save(model.state_dict(), os.path.join(self.args.train_path, 'model.ckpt'))
                    if ema is not None:
                        th.save(ema.state_dict(), os.path.join(self.args.train_path, 'ema.ckpt'))

                if step % 20000 == 0:
                    if world_size >= 2:
                        th.save(model.module.state_dict(), os.path.join(self.args.train_path, f'model-{step}.ckpt'))
                    else:
                        th.save(model.state_dict(), os.path.join(self.args.train_path, f'model-{step}.ckpt'))
                    if ema is not None:
                        th.save(ema.state_dict(), os.path.join(self.args.train_path, f'ema-{step}.ckpt'))

                self.logger(tb_logger, step, scalars, images)

    def before_sample(self):
        config = self.config['Dataset']

        model = self.model
        device = self.device
        rank = self.rank
        world_size = self.world_size

        if world_size >= 2:
            th.cuda.set_device(rank)
            dist.init_process_group(backend='nccl')

        state_dict = th.load(self.args.model_path, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict, strict=True)
        model.eval()

        skip = self.diffusion_step // self.sample_speed
        seq = list(range(0, self.diffusion_step + 1, skip))

        dataset, test_dataset = get_dataset(self.args, config)
        # print(len(dataset))
        batch_size = self.config['Sample']['batch_size']
        if world_size >= 2:
            sampler = data.distributed.DistributedSampler(dataset)
            train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                           num_workers=config['num_workers'])
        else:
            train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=config['num_workers'])

        return seq, skip, train_loader

    @th.no_grad()
    def sample_fid(self):  # todo 通过before_sample初始化
        config = self.config['Sample']
        rank = self.rank
        world_size = self.world_size

        if world_size >= 2:
            th.cuda.set_device(rank)
            dist.init_process_group(backend='nccl')

        model = self.model
        device = self.device
        continuous = True if self.args.method == 'PF' else False

        state_dict = th.load(self.args.model_path, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():  # module.xxx
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict, strict=True)
        model.eval()

        n = config['batch_size']
        total_num = config['total_num']
        # total_num = 25000

        skip = self.diffusion_step // self.sample_speed
        image_num = 0

        config = self.config['Dataset']

        if self.args.method in ('DDIM', 'PNDM2', 'PNDM4'):
            interval = (self.diffusion_step - skip, -1)
        elif self.args.method in ('PF', 'NDM1', 'NDM4'):
            interval = (self.diffusion_step, 0)
        else:
            interval = None

        for _ in tqdm(range(total_num // n + 1), desc="gen_image", disable=rank + 1 - world_size):
            noise = th.randn(n, config['channels'], config['image_size'],
                             config['image_size'], device=self.device)

            img = self.schedule.multi_iteration(noise, *interval, self.model,
                                                last=True, fresh=True, continuous=continuous)

            img = th.clamp(img / 2 + 0.5, 0, 1)
            for i in range(img.shape[0]):
                if image_num + i > total_num:
                    break
                tvu.save_image(img[i], os.path.join(self.args.image_path,
                                                    f"{rank}-{image_num + i}.png"))

            image_num += n

    @th.no_grad()
    def test(self):
        """
        test noise enhancement
        """
        test_size = 8

        model = self.model
        schedule = self.schedule
        device = self.device
        continuous = self.args.method == 'PF'
        # self.config['Dataset']['split_block'] = 0

        seq, skip, train_loader = self.before_sample()
        alpha = self.schedule.alphas_cump
        choice_list = list(range(960, -20, -20))

        for (img, y) in train_loader:
            img = img[:test_size].to(device) * 2 - 1

            # # 切出步测试
            # noise = th.randn_like(img)
            # noise = F.pad(noise, [4]*4)
            # img_r = self.schedule.multi_iteration(noise, 980 - 1, -1, model,
            #                                       last=False, fresh=True, continuous=continuous)
            #
            # noise_list = self.schedule.e_accum
            # img0_r = [(img_r[i+1] - noise_list[i] * alpha[choice_list[i]]) / (1 - alpha[choice_list[i]])
            #           for i in range(len(choice_list))]
            #
            # img_middle = F.resize(F.resize(img_r[-1], [16, 16]), [32, 32])
            # choice_list = list(range(980, 0, -20))
            # loss_list = [(item - img_middle).square().sum(dim=(1, 2, 3)).mean() for item in img0_r]
            # for out in zip(choice_list, loss_list):
            #     print(out)

            # # 数据编辑测试
            # _, _, h, w = img.shape
            # config = self.config['Train']
            #
            # img_input = self.schedule.splitter(img, 'center', 4)
            img_split, img_condition = img.split([3, 3], dim=1)
            # img_col = self.schedule.collection(img_split, 4, last=True)

            # img_split, img_condition = img_input.split((3, 3), dim=1)
            # b, c, h, w = img.shape
            # img_r = img.view(-1, c, 2, h // 2, 2, w // 2)
            # img_r = img_r.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, h // 2, w // 2)
            # img_r = th.index_select(img_r, 0, th.arange(0, b, 1) * 4 + th.randint(0, 4, (b,)))
            #
            # img = transforms.Resize((32, 32))(img)
            # img = img.unsqueeze(1).repeat(1, 4, 1, 1, 1).reshape(-1, c, h // 2, w // 2)
            start = time.time()
            img_split = th.clamp(img_split * 0.5 + 0.5, 0, 1)
            img_condition = th.clamp(img_condition * 0.5 + 0.5, 0, 1)
            for i in range(test_size):
                tvu.save_image(img_split[i], os.path.join(self.args.image_path,
                                                          f"img_split-{i + 1}.png"))
                # tvu.save_image(img_split[i], os.path.join(self.args.image_path,
                #                                       f"img_split-{i + 1}.png"))
                tvu.save_image(img_condition[i], os.path.join(self.args.image_path,
                                                              f"img_condition-{i + 1}.png"))

            print(time.time() - start)
            sys.exit()


