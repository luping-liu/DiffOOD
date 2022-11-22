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


import sys
import copy
import types
import torch as th

transfer, gradient = {}, {}


def register(tg_type, tg_name=None):
    def decorator(target):
        name = target.__name__ if tg_name is None else tg_name
        target = target if isinstance(target, types.FunctionType) else target()

        if tg_type == 'transfer':
            transfer[name] = target
        elif tg_type == 'gradient':
            gradient[name] = target

        return target

    return decorator


class Pseudo_Numerical_Method(object):
    def __init__(self, tran_name, grad_name):
        self.transfer = transfer[tran_name]
        self.gradient = gradient[grad_name]

    def __call__(self, *args, **kwargs):
        return self.gradient(self.transfer, *args, **kwargs)


@register('gradient', tg_name='FE1')
def forward_euler(tran, x, t, t_next, func, e_list=None):
    e = func(x, t)
    x_next = tran(x, e, t, t_next)

    return x_next, e, e


@register('gradient', tg_name='IE2')
def improved_euler(tran, x, t, t_next, func, e_list=None):
    e_1 = func(x, t)
    x_1 = tran(x, e_1, t, t_next)

    e_2 = func(x_1, t_next)
    e = 0.5 * (e_1 + e_2)
    x_next = tran(x, e, t, t_next)

    return x_next, e_1, e


@register('gradient', tg_name='LMS2')
def linear_multi_step_2(tran, x, t, t_next, func, e_list):
    if len(e_list) == 0:
        return improved_euler(tran, x, t, t_next, func)

    e_1 = func(x, t)
    e = 0.5 * (3 * e_1 - e_list[-1])
    x_next = tran(x, e, t, t_next)

    return x_next, e_1, e


@register('gradient', tg_name='RK4')
def runge_kutta(tran, x, t, t_next, func, e_list=None):
    t_middle = 0.5 * (t + t_next)

    e_1 = func(x, t)
    x_1 = tran(x, e_1, t, t_middle)

    e_2 = func(x_1, t_middle)
    x_2 = tran(x, e_2, t, t_middle)

    e_3 = func(x_2, t_middle)
    x_3 = tran(x, e_3, t, t_next)

    e_4 = func(x_3, t_next)
    e = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)
    x_next = tran(x, e, t, t_next)

    return x_next, e_1, e


@register('gradient', tg_name='LMS4')
def linear_multi_step_4(tran, x, t, t_next, func, e_list):
    if len(e_list) < 3:
        return runge_kutta(tran, x, t, t_next, func)

    e_1 = func(x, t)
    e = (1 / 24) * (55 * e_1 - 59 * e_list[-1] + 37 * e_list[-2] - 9 * e_list[-3])
    x_next = tran(x, e, t, t_next)

    return x_next, e_1, e


@register('transfer', tg_name='Linear')
def tran_linear(x, e, t, t_next):
    c = x.shape[1]
    x, x_condition = x.split([3, c - 3], dim=1)

    x = x + (t_next - t).view(-1, 1, 1, 1) * e
    x = th.cat([x, x_condition], dim=1)

    return x


@register('transfer', tg_name='DDIM')
class tran_ddim(object):
    def __init__(self):
        self.alphas = None

    def __call__(self, x, e, t, t_next):
        c = x.shape[1]
        x, x_condition = x.split([3, c-3], dim=1)

        a_t = self.alphas[t.long() + 1].view(-1, 1, 1, 1)
        a_next = self.alphas[t_next.long() + 1].view(-1, 1, 1, 1)
        a_t_sq, a_next_sq = a_t.sqrt(), a_next.sqrt()

        x_delta = (a_next - a_t) * ((1 / (a_t_sq * (a_t_sq + a_next_sq))) * x -
                                    1 / (a_t_sq * (((1 - a_next) * a_t).sqrt() + ((1 - a_t) * a_next).sqrt())) * e)

        x = th.cat([x + x_delta, x_condition], dim=1)

        return x


def gen_pflow(img, t, t_next, model, betas, total_step):
    n = img.shape[0]
    beta_0, beta_1 = betas[0], betas[-1]

    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # drift, diffusion -> f(x,t), g(t)
    drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    score = - model(img, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
    drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

    return drift


if __name__ == '__main__':
    print(gradient, transfer)

    def func0(x, t):
        return x

    method = Pseudo_Numerical_Method('DDIM', 'RK4')
    output = method(0.1, th.tensor(0), th.tensor(4), func0)
    print(output)
