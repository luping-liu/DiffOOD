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


import math
import torch as th
import torchvision.transforms.functional as F
import numpy as np
from scipy import integrate
from einops import rearrange
from collections import deque

from method.pndm import Pseudo_Numerical_Method, gen_pflow


def get_schedule(args, config):
    def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    if config['type'] == "quad":
        betas = (np.linspace(config['beta_start'] ** 0.5, config['beta_end'] ** 0.5, config['diffusion_step'],
                             dtype=np.float64) ** 2)
    elif config['type'] == "linear":
        betas = np.linspace(config['beta_start'], config['beta_end'], config['diffusion_step'], dtype=np.float64)
    elif config['type'] == 'cosine':
        betas = betas_for_alpha_bar(config['diffusion_step'],
                                    lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else:
        betas = None

    betas = th.from_numpy(betas).float()
    alphas = 1.0 - betas
    alphas_cump = alphas.cumprod(dim=0)

    return betas, alphas_cump


def choose_item(alpha_list, t):
    alpha = alpha_list.index_select(0, t).view(-1, 1, 1, 1)
    return alpha


class Schedule(object):
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.device = device = th.device(args.device)
        betas, alphas_cump = get_schedule(args, config)

        self.betas, alphas_cump = betas.to(device), alphas_cump.to(device)
        self.alphas_cump = alphas_cump
        self.alphas_cump_sqrt = th.sqrt(alphas_cump)
        self.alphas_cump_1m_sqrt = th.sqrt(1 - alphas_cump)
        self.alphas_cump_recip_sqrt = th.sqrt(1.0 / alphas_cump)
        self.alphas_cump_recip_m1_sqrt = th.sqrt(1.0 / alphas_cump - 1.0)
        self.diffusion_step = config['diffusion_step']
        self.sample_speed = args.sample_speed

        mtd_dict = {'NDM1': ('Linear', 'FE1'), 'NDM4': ('Linear', 'LMS4'),
                    'DDIM': ('DDIM', 'FE1'), 'PNDM2': ('DDIM', 'LMS2'), 'PNDM4': ('DDIM', 'LMS4')}

        if args.method == 'PF':
            self.method = gen_pflow
        else:
            self.method = Pseudo_Numerical_Method(*mtd_dict[args.method])
            if args.method in ('DDIM', 'PNDM2', 'PNDM4'):
                self.method.transfer.alphas = self.alphas_cump

        self.e_simple = deque(maxlen=4)
        self.e_accum = []

    def diffusion(self, img, t_end, noise=None, teacher=None):
        if noise is None:
            noise = th.randn_like(img)
        else:
            noise = noise.to(img.device)

        alpha_sqrt = self.alphas_cump_sqrt.index_select(0, t_end).view(-1, 1, 1, 1)
        alpha_1m_sqrt = self.alphas_cump_1m_sqrt.index_select(0, t_end).view(-1, 1, 1, 1)

        img_n = img * alpha_sqrt + noise * alpha_1m_sqrt

        if teacher is not None:
            with th.no_grad():
                noise = teacher(img_n, t_end).detach()

        return img_n, t_end, noise

    def denoising(self, img_n, t_start, t_end, model, first_step=False, continuous=False):
        if continuous:
            drift = self.method(img_n, t_start, t_end, model, self.betas, self.diffusion_step)
            # img_next = self.method(img_n, t_start, t_end, model, self.betas, self.ets, self.diffusion_step)

            return drift
        else:
            if first_step:
                self.e_simple = deque(maxlen=4)
                self.e_accum = []
            img_next, noise_1, noise = self.method(img_n, t_start, t_end, model, self.e_simple)
            self.e_simple.append(noise_1)
            self.e_accum.append(noise)

        return img_next, noise

    def splitter(self, img, split_type, split_block=4):
        """
        切块模块，目前还是输入数据集数据，后续要改为输入生成的低分辨率样本
        """
        if split_type == 'corner':
            pass
        elif split_type == 'center':
            _, _, h, w = img.shape
            assert h == w
            split_ratio = int(math.sqrt(split_block))
            h_new, w_new = h // split_ratio, w // split_ratio

            img = F.pad(img, [h_new // 2] * 4)
            img_condition_list = [F.crop(img, x * h_new, y * h_new, 2 * h_new, 2 * h_new)
                                  for x in range(split_ratio) for y in range(split_ratio)]
            img_condition = th.stack(img_condition_list, dim=1)
            img_condition = rearrange(img_condition, 'b r ... -> (b r) ...')

            img_split = F.crop(img_condition, h_new // 2, h_new // 2, h_new, h_new)
            img_condition = F.resize(img_condition, [h_new, h_new])
            img_input = th.cat([img_split, img_condition], dim=1)

            return img_input
        elif split_type == 'onestep':
            _, _, h, w = img.shape
            assert h == w
            split_ratio = int(math.sqrt(split_block))
            h_new, w_new = h * 2 // split_ratio, w * 2 // split_ratio

            img = F.pad(img, [h_new // 4] * 4)
            img_condition_list = [F.crop(img, x * h_new // 2, y * h_new // 2, h_new, h_new)
                                  for x in range(split_ratio) for y in range(split_ratio)]
            img_condition = th.stack(img_condition_list, dim=1)
            img_condition = rearrange(img_condition, 'b r ... -> (b r) ...')

            img_input = th.cat([th.randn_like(img_condition), img_condition], dim=1)

            return img_input
        elif split_type == 'resize':
            _, _, h, w = img.shape
            assert h == w
            split_ratio = int(math.sqrt(split_block))
            h_new, w_new = h * 2 // split_ratio, w * 2 // split_ratio

            img_condition_list = [F.crop(img, x * h_new // 2, y * h_new // 2, h_new // 2, h_new // 2)
                                  for x in range(split_ratio) for y in range(split_ratio)]
            img_condition = th.stack(img_condition_list, dim=1)
            img_condition = rearrange(img_condition, 'b r ... -> (b r) ...')

            img_condition = F.resize(img_condition, [h_new] * 2)
            img_input = th.cat([th.randn_like(img_condition), img_condition], dim=1)

            return img_input
        elif split_type == 'split_pad':
            _, _, h, w = img.shape
            assert h == w
            split_ratio = int(math.sqrt(split_block))
            h_new, w_new = h // split_ratio, w // split_ratio

            # img = F.resize(img, [h_new * 2] * 2)
            img = F.pad(img, [4]*4)
            img = th.cat([th.randn_like(img), img], dim=1)
            img_condition_list = [F.crop(img, x * h_new, y * h_new, h_new + 8, h_new + 8)
                                  for x in range(split_ratio) for y in range(split_ratio)]
            img_condition = th.stack(img_condition_list, dim=1)
            img_condition = rearrange(img_condition, 'b r ... -> (b r) ...')

            img_input = img_condition

            return img_input

    def collection(self, img, split_block, pad=0, last=False):
        split_ratio = int(math.sqrt(split_block))
        if last:
            img = img[..., pad:-pad, pad:-pad] if pad > 0 else img
            img_col = rearrange(img, '(b r1 r2) c h w -> b c (r1 h) (r2 w)', r1=split_ratio, r2=split_ratio)

            return img_col
        else:
            average_1 = th.arange(1, 9, device=self.device).view(1, 1, 1, 1, 8, 1) / 9
            average_2 = average_1.view(1, 1, 1, 1, 1, 8)

            img_col = rearrange(img, '(b r1 r2) ... -> b r1 r2 ...', r1=split_ratio, r2=split_ratio)
            for i in range(split_ratio-1):
                correction = img_col[:, i, ..., -8:, :] * (1 - average_1) + img_col[:, i+1, ..., :8, :] * average_1
                img_col[:, i, ..., -8:, :] = img_col[:, i+1, ..., :8, :] = correction

            for j in range(split_ratio-1):
                correction = img_col[:, :, j, ..., -8:] * (1 - average_2) + img_col[:, :, j+1,  ..., :8] * average_2
                img_col[:, :, j, ..., -8:] = img_col[:, :, j+1,  ..., :8] = correction

            img_col = rearrange(img_col, 'b r1 r2 ... -> (b r1 r2) ...')

            return img_col

    @th.no_grad()
    def multi_iteration(self, img_n, t_start, t_end, model, seq=None, y=None, class_num=10, split_block=0,
                        last=True, fresh=True, continuous=False):
        """
        Multi_iteration is designed for both multistep diffusion and multistep denoising.
        The only difference between them is t_start < t_end or vice versa.
        """
        if t_end == t_start:
            return img_n.to('cpu')

        if continuous:
            shape = img_n.shape
            device = self.device
            speed = self.sample_speed
            tol = 7e-4 if speed > 1 else speed
            interval = (t_start / self.diffusion_step if t_start != 0 else 1e-3,
                        t_end / self.diffusion_step if t_end != 0 else 1e-3)

            # self.count = []
            def drift_func(t, x):
                # self.count.append(t)
                x = th.from_numpy(x.reshape(shape)).to(device).type(th.float32)
                drift = self.denoising(x, t, None, model, continuous=continuous)
                drift = drift.cpu().numpy().reshape((-1,))
                return drift

            solution = integrate.solve_ivp(drift_func, interval, img_n.cpu().numpy().reshape((-1,)),
                                           rtol=tol, atol=tol, method='RK45')
            # print("len:", len(self.count), end=" ")
            imgs = th.tensor(solution.y).reshape(*shape, -1).type(th.float32)
            imgs = imgs.permute(4, 0, 1, 2, 3)
            # print(len(imgs), len(self.count))
            # sys.exit()
        else:
            imgs = [img_n.to(self.device)]
            n = img_n.shape[0]
            speed = np.sign(t_end - t_start) * (self.diffusion_step // self.sample_speed)

            if seq is None:
                seq = range(t_start, t_end, speed)
            seq_next = list(seq)[1:] + [t_end]

            if fresh:
                self.e_simple = deque(maxlen=4)
                self.e_accum = []

            if self.args.method in ('NDM1', 'NDM4'):
                seq, seq_next = np.array(seq) / self.diffusion_step, np.array(seq_next) / self.diffusion_step

                def grad(x, t):
                    total_step = self.diffusion_step
                    beta_0, beta_1 = self.betas[0], self.betas[-1]

                    t_start = th.ones(n, device=x.device) * t
                    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
                    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
                    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

                    # drift, diffusion -> f(x,t), g(t)
                    drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * x, th.sqrt(beta_t)
                    score = - model(x, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
                    drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

                    if split_block > 0:
                        drift = self.collection(drift, split_block)
                    return drift
            else:
                def grad(x, t):  # careful!
                    uncond, beta = th.ones_like(y) * class_num, 2
                    out = (1 + beta) * model(x, t, y=y) - beta * model(x, t, y=uncond)
                    # out = model(x, t)
                    if split_block > 0:
                        out = self.collection(out, split_block)
                    return out

            for i, j in zip(seq, seq_next):
                t = (th.ones(n, device=self.device) * i)
                t_next = (th.ones(n, device=self.device) * j)

                img_t = imgs[-1]
                img_next, _ = self.denoising(img_t, t, t_next, grad, continuous=False)

                imgs.append(img_next.detach())

        if last:
            return imgs[-1].to('cpu')
        else:
            return imgs

    @th.no_grad()
    def mask_iteration(self, img, noise, mask, t_start, t_end, model,
                       seq=None, last=True, fresh=True):
        if t_end == t_start:
            return img.to('cpu')

        img_n, _, _ = self.diffusion(img, t_start, noise)

        imgs = [img_n.to(self.device)]
        n = img_n.shape[0]
        speed = np.sign(t_end - t_start) * (self.diffusion_step // self.sample_speed)

        if seq is None:
            seq = range(t_start, t_end, speed)
        seq_next = list(seq)[1:] + [t_end]

        if fresh:
            self.ets = []
            self.noise = []

        if self.args.method in ('NDM1', 'NDM4'):
            seq, seq_next = np.array(seq) / self.diffusion_step, np.array(seq_next) / self.diffusion_step

            def grad(x, t):
                total_step = self.diffusion_step
                beta_0, beta_1 = self.betas[0], self.betas[-1]

                t_start = th.ones(n, device=x.device) * t
                beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step
                log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
                std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

                # drift, diffusion -> f(x,t), g(t)
                drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * x, th.sqrt(beta_t)
                score = - model(x, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
                drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

                return drift
        else:
            def grad(x, t):
                return model(x, t)

        for i, j in zip(seq, seq_next):
            t = (th.ones(n, device=self.device) * i)
            t_next = (th.ones(n, device=self.device) * j)

            img_t = imgs[-1]
            img_next = self.denoising(img_t, t, t_next, grad, continuous=False)

            img_next_ = self.diffusion(img, t_next, noise)
            img_next[:, (mask == 1.)] = img_next_[:, (mask == 1.)]
            # noise是不是也要改？
            # 调控空间是不是可以更灵活？

            imgs.append(img_next.detach())

        if last:
            return imgs[-1].to('cpu')
        else:
            return imgs

    def correction(self, img_n, t_start, t_end, model, first_step=False):
        method = Pseudo_Numerical_Method('DDIM', 'FE1')
        device = self.device
        n, img_n = img_n.shape[0], img_n.to(device)
        t_start = (th.ones(n, device=device) * t_start)
        t_end = (th.ones(n, device=device) * t_end)
        if first_step:
            self.ets = []
        img_next = method(img_n, t_start, t_end, model, self.alphas_cump, self.ets)
        return img_next.to('cpu')

    def diffusion_resample(self, img, t_end, span, noise, teacher=None):
        t_add = th.randint_like(t_end, 0, 2 * span)

        alpha = self.alphas_cump.index_select(0, t_end).view(-1, 1, 1, 1)
        alpha_pre = self.alphas_cump.index_select(0, t_end + t_add).view(-1, 1, 1, 1)
        alpha_pre_sqrt = self.alphas_cump_sqrt.index_select(0, t_end + t_add).view(-1, 1, 1, 1)
        alpha_1m_sqrt = self.alphas_cump_1m_sqrt.index_select(0, t_end).view(-1, 1, 1, 1)
        alpha_pre_1m_sqrt = self.alphas_cump_1m_sqrt.index_select(0, t_end + t_add).view(-1, 1, 1, 1)
        alpha_recip_sqrt = self.alphas_cump_recip_sqrt.index_select(0, t_end).view(-1, 1, 1, 1)
        alpha_recip_m1_sqrt = self.alphas_cump_recip_m1_sqrt.index_select(0, t_end).view(-1, 1, 1, 1)

        img_c = alpha_recip_sqrt * img - alpha_recip_m1_sqrt * noise
        noise_ = (noise * alpha_1m_sqrt + th.randn_like(noise) * th.sqrt(alpha - alpha_pre)) / alpha_pre_1m_sqrt
        img_n = img_c * alpha_pre_sqrt + noise_ * alpha_pre_1m_sqrt

        return img_n, t_end + t_add, noise_
