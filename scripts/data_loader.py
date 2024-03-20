from pathlib import Path
from glob import glob
import numpy as np
import torch
from pathlib import Path
from random import shuffle


def data_loader(batch_size, train_set = True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if train_set:
        folder = 'Train'
    else:
        folder = 'Test'
    
    data_dir = f'/home/jin/Data/{folder}'
    files = [ Path(x) for x in glob(f'{data_dir}/*.npy')]
    files = [x.stem for x in files if '_y.npy' not in str(x)]
    shuffle(files)
   
    for i in files:
        x = np.float32(np.load(f'{data_dir}/{i}.npy'))
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x).to(device)

        y = np.float32(np.load(f'{data_dir}/{i}_y.npy'))
        y = torch.from_numpy(y).to(device)

        yield x, y
