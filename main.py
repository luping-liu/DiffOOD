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


import argparse
import yaml
import sys
import os
import numpy as np
import torch as th

from runner.schedule import Schedule
from runner.runner import Runner


def args_and_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runner", type=str, default='sample',
                        help="Choose the mode of runner")
    parser.add_argument("--config", type=str, default='32_cifar10.yml',
                        help="Choose the config file")
    parser.add_argument("--model", type=str, default='DDIM',
                        help="Choose the model's structure (DDIM, iDDPM, PF)")
    parser.add_argument("--method", type=str, default='PNDM4',
                        help="Choose the numerical methods (DDIM, PNDM2, PNDM4, NDM1, NDM4, PF)")
    parser.add_argument("--sample_speed", type=int, default=50,
                        help="Control the total generation step")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Choose the device to use")
    parser.add_argument("--image_path", type=str, default='temp/sample',
                        help="Choose the path to save images")
    parser.add_argument("--category", type=str, default='None',
                        help="Choose the category of images")
    parser.add_argument("--category_value", type=int, default=1,
                        help="Choose the value of the category of images")
    parser.add_argument("--model_path", type=str, default='temp/model/ddim_cifar10.ckpt',
                        help="Choose the path of model")
    parser.add_argument("--model_name", type=str, default='cifar10',
                        help="Set the model's name")
    parser.add_argument("--repeat_size", type=int, default=1,
                        help="Set the model's name")
    parser.add_argument("--restart", action="store_true",
                        help="Restart a previous training process")
    parser.add_argument("--train_path", type=str, default='temp/train',
                        help="Choose the path to save training status")
    parser.add_argument("--reinitialize", type=str, default=None,
                        help="Choose certain part of the model to fine tune")
    parser.add_argument("--debug_value", type=int, default=1,
                        help="Set the debug value")

    args = parser.parse_args()

    rank = os.environ.get("LOCAL_RANK")
    rank = 0 if rank is None else int(rank)
    world_size = os.environ.get("WORLD_SIZE")
    world_size = 1 if world_size is None else int(world_size)
    if world_size >= 2:
        parser.set_defaults(device=th.device(args.device, rank))
        args = parser.parse_args()

    work_dir = os.getcwd()
    with open(f'{work_dir}/config/{args.config}', 'r') as f:
        config = yaml.safe_load(f)

    return args, config


def check_config(config):
    # assert config['Dataset']['batch_size'] == config['Train']['batch_size'] // 2
    pass


if __name__ == "__main__":
    args, config = args_and_config()
    check_config(config)

    device = th.device(args.device)
    schedule = Schedule(args, config['Schedule'])

    # Load model
    if config['Model']['struc'] == 'DDIM':
        from model.ddim import Model
        model = Model(args, config['Model']).to(device)
    else:
        model = None

    # Load runner
    if args.runner == 'base_train':
        runner = Runner(args, config, schedule, model)
        runner.train_loop()
    elif args.runner == 'base_fid':
        runner = Runner(args, config, schedule, model)
        runner.sample_fid()
    elif args.runner == 'ood_repr':
        from runner.ood_detect import OodDetection
        runner = OodDetection(args, config, schedule, model)
        runner.noise_encoder()
    elif args.runner == 'ood_ehc':
        from runner.ood_detect import OodDetection
        runner = OodDetection(args, config, schedule, model)
        runner.enhancement()
    elif args.runner == 'ood_itp':
        from runner.ood_detect import OodDetection
        runner = OodDetection(args, config, schedule, model)
        runner.interp_detect()
    else:
        print(f'Do not find the runner {args.runner}.')
