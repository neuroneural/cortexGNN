import numpy as np
from tqdm import tqdm
import config
import torch
from torch.utils.data import DataLoader

from config import load_config
from data.dataload import load_data, BrainDataset



from csv import writer

if __name__ == '__main__':
    test_data = load_data(data_path = config.data_path,
                          hemisphere = config.hemisphere)
    n_data = len(test_data)
    L,W,H = test_data[0].volume[0].shape  # shape of MRI
    LWHmax = max([L,W,H])

    test_set = BrainDataset(test_data)
    testloader = DataLoader(test_set, batch_size=1, shuffle=True)
    have = []
    for idx, data in tqdm(enumerate(testloader)):
        have.append(idx,data)
        print(idx, data)
