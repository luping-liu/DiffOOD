
import ast
import io
import sys
import logging
import os

import torch
import torchvision.transforms as tvs_trans
from PIL import Image, ImageFile

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


interpolation = 'bilinear'
normalization_dict = {'cifar10': [[0.4914, 0.4822, 0.4465], [0.247, 0.2435, 0.2616]],
                      'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
                      'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                      'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]]}


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ImglistDataset(BaseDataset):
    def __init__(self,
                 name,
                 split,
                 image_size,
                 imglist_pth,
                 data_dir,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(ImglistDataset, self).__init__(**kwargs)

        self.name = name
        self.image_size = image_size
        imglist_pth = os.path.join('/home/liuluping/llp/Workspace/OpenOOD', imglist_pth[2:])
        data_dir = os.path.join('/home/liuluping/llp/Workspace/OpenOOD', data_dir[2:])
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        # self.num_classes = num_classes

        mean, std = normalization_dict[self.name]
        self.preprocessor = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(image_size),  # interpolation=interpolation
            tvs_trans.CenterCrop(image_size),
            tvs_trans.ToTensor(),
            # tvs_trans.Normalize(mean=mean, std=std),
        ])

        self.transform_image = self.preprocessor
        # self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name

        # kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        # self.preprocessor.setup(**kwargs)

        try:
            if not self.dummy_read:
                with open(path, 'rb') as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
                # sample['data_aux'] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            # # Generate Soft Label
            # soft_label = torch.Tensor(self.num_classes)
            # if sample['label'] < 0:
            #     soft_label.fill_(1.0 / self.num_classes)
            # else:
            #     soft_label.fill_(0)
            #     soft_label[sample['label']] = 1
            # sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample
