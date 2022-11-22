import time
import torch as th
import numpy as np
import torchvision.utils as tvu
from tqdm.auto import tqdm

name = 'cifar10'
total_size = 5000

start = time.time()
data = np.load(f'temp/itp_fid/{name}_inter.npz')
img1, img2 = data['img1'], data['img2']
sym, inter1, inter2 = data['sym'], data['inter1'], data['inter2']

shape = img1.shape
img1 = th.from_numpy(img1)
img2 = th.from_numpy(img2)
sym, inter1, inter2 = th.tensor(np.array(sym)), th.tensor(np.array(inter1)), th.tensor(np.array(inter2))

for i in tqdm(range(len(sym)), leave=False):
    for j in range(len(sym[i])):
        tvu.save_image(sym[i][j], f'temp/itp_fid/sym/{i + 1}-{j}.png')
        tvu.save_image(inter1[i][j], f'temp/itp_fid/inter1/{i + 1}-{j}.png')
        tvu.save_image(inter2[i][j], f'temp/itp_fid/inter2/{i + 1}-{j}.png')

print(name, time.time() - start)
