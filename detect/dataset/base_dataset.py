# MIT License
#
# Copyright (c) 2021 Jingkang Yang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
import random
import traceback

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, pseudo_index=-1, skip_broken=False, new_index='next'):
        super(BaseDataset, self).__init__()
        self.pseudo_index = -1
        self.skip_broken = skip_broken
        self.new_index = new_index
        if new_index not in ('next', 'rand'):
            raise ValueError('new_index not one of ("next", "rand")')

    def __getitem__(self, index):
        # in some pytorch versions, input index will be torch.Tensor
        index = int(index)

        # if sampler produce pseudo_index,
        # randomly sample an index, and mark it as pseudo
        # if index == self.pseudo_index:
        #     index = random.randrange(len(self))
        #     pseudo = 1
        # else:
        #     pseudo = 0

        while True:
            try:
                sample = self.getitem(index)
                break
            except Exception as e:
                if self.skip_broken and not isinstance(e, NotImplementedError):
                    if self.new_index == 'next':
                        new_index = (index + 1) % len(self)
                    else:
                        new_index = random.randrange(len(self))
                    logging.warn(
                        'skip broken index [{}], use next index [{}]'.format(
                            index, new_index))
                    index = new_index
                else:
                    logging.error('index [{}] broken'.format(index))
                    traceback.print_exc()
                    logging.error(e)
                    raise e

        sample['index'] = index
        # sample['pseudo'] = pseudo
        return sample

    def getitem(self, index):
        raise NotImplementedError
