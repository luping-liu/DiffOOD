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
import sys

import torch as th
import torchvision.transforms.functional as F
import numpy as np
from scipy import integrate
from einops import rearrange
from collections import deque

from runner.method import Pseudo_Numerical_Method, gen_pflow


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
        self.sample_step = args.sample_step

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

    @th.no_grad()
    def multi_iteration(self, img_n, t_start, t_end, model, seq=None, y=None, num_classes=0, beta=2, split_block=0,
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
            # PNDM is here.
            imgs = [img_n.to(self.device)]
            n = img_n.shape[0]
            speed = np.sign(t_end - t_start) * (self.diffusion_step // self.sample_step)

            if seq is None:
                seq = range(t_start, t_end, speed)
            seq_next = list(seq)[1:] + [t_end]

            if fresh:
                self.e_simple = deque(maxlen=4)
                self.e_accum = []

            if self.args.method in ('DDIM', 'PNDM2', 'PNDM4'):
                def grad(x, t):
                    if num_classes > 0:
                        uncond = th.ones_like(y) * num_classes
                        out = (1 + beta) * model(x, t, y=y) - beta * model(x, t, y=uncond)
                    else:
                        out = model(x, t)
                    return out
            else:
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
