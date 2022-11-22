

import os
import math
import torch
import numbers
from einops import rearrange
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from dataset.celeba import CelebA
# from dataset.ffhq import FFHQ
from dataset.lsun import LSUN
from dataset.diffae import CelebAttrDataset, CelebHQAttrDataset
from torch.utils.data import Subset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img


class Split(object):  # todo 将图像处理并入这里
    def __init__(self, split_type, split_block, split_smooth=False):
        self.type = split_type
        self.smooth = split_smooth
        # self.block = split_block
        self.ratio = int(math.log2(split_block)) if split_type == 'corner' else int(math.sqrt(split_block))

    @ torch.no_grad()
    def __call__(self, img):
        split_tpye = self.type
        if split_tpye == 'corner':
            split_ratio = self.ratio

            def split_fn(img):
                c, h, w = img.shape

                img_split = img.view(c, 2, h // 2, 2, w // 2)  # todo 改为rearrange版本
                img_split = img_split.permute(1, 3, 0, 2, 4).reshape(-1, c, h // 2, w // 2)
                img_split = torch.index_select(img_split, 0, torch.randint(0, 4, (1,)))

                # img_small = transforms.Resize((h // 2, w // 2))(img)
                img_small = F.resize(img, [h // 2, w // 2])
                return img_split, img_small

            img_split, img_small = None, None
            for _ in range(split_ratio // 2):
                img_split, img_small = split_fn(img)
                img = img_split
        elif split_tpye == 'center':
            split_ratio = self.ratio
            c, h, w = img.shape
            assert h == w
            h_new, w_new = h // split_ratio, w // split_ratio

            x, y = torch.randint(0, split_ratio, (2,))
            img = F.pad(img, [h_new // 2] * 4)
            # img = img[..., x * h_new:(x + 2) * h_new, y * h_new:(y + 2) * h_new]
            img = F.crop(img, x * h_new, y * h_new, 2 * h_new, 2 * h_new)

            # img_split = img[..., h_new // 2:-h_new // 2, h_new // 2:-h_new // 2]
            img_split = F.crop(img, h_new // 2, h_new // 2, h_new, h_new)
            # img_small = transforms.Resize((h_new, h_new))(img)
            img_small = F.resize(img, [h_new, h_new])
        elif split_tpye == 'resize':
            split_ratio = self.ratio
            c, h, w = img.shape
            assert h == w
            h_new, w_new = h // split_ratio, w // split_ratio

            if self.smooth:
                x, y = torch.randint(0, h - h_new, (2, ))
                img_split = F.crop(img, x, y, h_new, h_new)
            else:
                x, y = torch.randint(0, split_ratio, (2,))
                # img = F.pad(img, [4] * 4, fill=0)
                img_split = F.crop(img, x * h_new, y * h_new, h_new, h_new)
            # img_small = F.resize(F.resize(img_split, [(h_new + 8) // 2]*2), [h_new + 8]*2)
            return img_split
        else:
            img_split, img_small = None, None

        img_output = torch.cat([img_split, img_small], dim=0)

        return img_output


def get_dataset(args, config):
    if config['random_flip'] is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config['image_size']), transforms.ToTensor(),
             Split(config['split_type'], config['split_block'], config['split_smooth'])
             if config['split_block'] > 0 else Identity(), ]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config['image_size']),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                Split(config['split_type'], config['split_block'], config['split_smooth'])
                if config['split_block'] > 0 else Identity(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config['image_size']), transforms.ToTensor(),
             Split(config['split_type'], config['split_block'], config['split_smooth'])
             if config['split_block'] > 0 else Identity(), ]
        )

    if config['dataset'] == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(os.getcwd(), "temp/dataset", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(os.getcwd(), "temp/dataset", "cifar10"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config['dataset'] == "CIFAR100":
        dataset = CIFAR100(
            os.path.join(os.getcwd(), "temp/dataset", "cifar100"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR100(
            os.path.join(os.getcwd(), "temp/dataset", "cifar100"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config['dataset'] == "SVHN":
        dataset = SVHN(
            os.path.join(os.getcwd(), "temp/dataset", "svhn"),
            split="train",
            download=True,
            transform=tran_transform,
        )
        test_dataset = SVHN(
            os.path.join(os.getcwd(), "temp/dataset", "svhn"),
            split="test",
            download=True,
            transform=test_transform,
        )

    elif config['dataset'] == "CELEBA":
        cx, cy = 89, 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        dataset = CelebA(
            root=os.path.join(os.getcwd(), "temp/dataset", "celeba"),
            split="train",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config['image_size']) if config['image_size'] != 128 else Identity(),
                    transforms.RandomHorizontalFlip() if config['random_flip'] else Identity(),
                    transforms.ToTensor(),
                    Split(config['split_type'], config['split_block'], config['split_smooth'])
                    if config['split_block'] > 0 else Identity(),
                ]
            ),
            download=True,
        )

        test_dataset = CelebA(
            root=os.path.join(os.getcwd(), "temp/dataset", "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config['image_size']) if config['image_size'] != 128 else Identity(),
                    transforms.ToTensor(),
                    Split(config['split_type'], config['split_block'], config['split_smooth'])
                    if config['split_block'] > 0 else Identity(),
                ]
            ),
            download=True,
        )

    elif config['dataset'] == "LSUN":
        train_folder = "{}_train".format(config['category'])
        val_folder = "{}_val".format(config['category'])
        dataset = LSUN(
            root=os.path.join(os.getcwd(), "temp/dataset", "lsun"),
            classes=[train_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config['image_size']),
                    transforms.CenterCrop(config['image_size']),
                    transforms.RandomHorizontalFlip() if config['random_flip'] else Identity(),
                    transforms.ToTensor(),
                    Split(config['split_type'], config['split_block'], config['split_smooth'])
                    if config['split_block'] > 0 else Identity(),
                ]
            ),
        )

        test_dataset = LSUN(
            root=os.path.join(os.getcwd(), "temp/dataset", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config['image_size']),
                    transforms.CenterCrop(config['image_size']),
                    transforms.ToTensor(),
                    Split(config['split_type'], config['split_block'], config['split_smooth'])
                    if config['split_block'] > 0 else Identity(),
                ]
            ),
        )
    elif config['dataset'] == "CELEBA(attr)":
        root = os.path.join(os.getcwd(), "temp/dataset", "celeba")
        attr_path = os.path.join(root, "celeba/list_attr_celeba.txt")

        dataset = CelebAttrDataset(root, image_size=config['image_size'], attr_path=attr_path,
                                   only_cls_name=args.category, only_cls_value=args.category_value,
                                   do_augment=True, do_transform=True, do_normalize=False, d2c=True)

        test_dataset = None
    elif config['dataset'] == "CELEBAHQ(attr)":
        root = os.path.join(os.getcwd(), "temp/dataset", "celebahq")
        attr_path = os.path.join(root, "CelebAMask-HQ-attribute.txt")

        split_fn = Split(config['split_type'], config['split_block'], config['split_smooth']) \
            if config['split_block'] > 0 else None
        dataset = CelebHQAttrDataset(root, image_size=config['image_size'], attr_path=attr_path,
                                     only_cls_name=args.category, only_cls_value=args.category_value,
                                     do_augment=True, do_transform=True, do_normalize=False, split_fn=split_fn)

        test_dataset = None

    # elif config.data.dataset == "FFHQ":
    #     if config.data.random_flip:
    #         dataset = FFHQ(
    #             path=os.path.join(args.exp, "datasets", "FFHQ"),
    #             transform=transforms.Compose(
    #                 [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
    #             ),
    #             resolution=config.data.image_size,
    #         )
    #     else:
    #         dataset = FFHQ(
    #             path=os.path.join(args.exp, "datasets", "FFHQ"),
    #             transform=transforms.ToTensor(),
    #             resolution=config.data.image_size,
    #         )
    #
    #     num_items = len(dataset)
    #     indices = list(range(num_items))
    #     random_state = np.random.get_state()
    #     np.random.seed(2019)
    #     np.random.shuffle(indices)
    #     np.random.set_state(random_state)
    #     train_indices, test_indices = (
    #         indices[: int(num_items * 0.9)],
    #         indices[int(num_items * 0.9) :],
    #     )
    #     test_dataset = Subset(dataset, test_indices)
    #     dataset = Subset(dataset, train_indices)
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config['uniform_dequantization']:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config['gaussian_dequantization']:
        X = X + torch.randn_like(X) * 0.01

    if config['rescaled']:
        X = 2 * X - 1.0
    elif config['logit_transform']:
        X = logit_transform(X)

    # if hasattr(config, "image_mean"):
    #     return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    # if hasattr(config, "image_mean"):
    #     X = X + config.image_mean.to(X.device)[None, ...]

    if config['logit_transform']:
        X = torch.sigmoid(X)
    elif config['rescaled']:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
