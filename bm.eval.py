import numpy as np
from tqdm import tqdm
import os
import csv



import torch
from torch.utils.data import DataLoader

from config import load_config
from data.dataload import load_data, BrainDataset
from model.pialnn import PialNN
from model.cortexGNN import CortexGNN

from utils import compute_normal, save_mesh_obj, compute_distance, compute_hausdorff
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    
    """set device"""
    device_name = None
    if torch.cuda.is_available():
        device_name = "cuda:0"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    print('device',device_name)

    """load configuration"""
    config = load_config()

    """load dataset"""
    print("----------------------------")
    print("Start loading dataset ...")
    allocated = []

    test_data = load_data(data_path = config.data_path,
                          hemisphere = config.hemisphere)
    allocated.append(torch.cuda.memory_allocated())
    n_data = len(test_data)
    L,W,H = test_data[0].volume[0].shape  # shape of MRI
    LWHmax = max([L,W,H])
    allocated.append(torch.cuda.memory_allocated())
    test_set = BrainDataset(test_data)
    testloader = DataLoader(test_set, batch_size=1, shuffle=True)
    print("Finish loading dataset. There are total {} subjects.".format(n_data))
    print("----------------------------")
    allocated.append(torch.cuda.memory_allocated())
    """load model"""
    print("Start loading model ...")
    
    model = None
    num_blocks = None
    sf = .1
    model_name = 'your_model_name'
    if config.cortexGNN and config.gnn_layers>1:
        num_blocks = 1
        model = CortexGNN(config.nc, config.K, config.n_scale,num_blocks,sf,config.gnn_layers,config.gnnVersion).to(device)#todo:revise num_blocks
        allocated.append(torch.cuda.memory_allocated())
    
        if config.gnnVersion==0:
            model_name = "PialGCN"
        elif config.gnnVersion==1:
            model_name = "PialGAT"
        
    else:
        num_blocks = 1    
        model_name ='PialNN'
        model = PialNN(config.nc, config.K, config.n_scale).to(device)#todo:revise 7
        allocated.append(torch.cuda.memory_allocated())
    
    allocated.append(torch.cuda.memory_allocated())
    print("Model is ", model_name)
    print('config.model_location',config.model_location)
    # model.load_state_dict(torch.load(f"{config.model_location}",
    #                                  map_location=device))
    allocated.append(torch.cuda.memory_allocated())

    model.initialize(L, W, H, device)
    model.eval()
    print("Finish loading model")
    print("----------------------------")
    
    
    """evaluation"""
    print("Start evaluation ...")
    n = 1
    CD = []
    AD = []
    HD = []
    for idx, data in tqdm(enumerate(testloader)):
        torch.cuda.empty_cache()

        for i in range(n):
            torch.cuda.empty_cache()
            with torch.no_grad():
                volume_in = None
                v_gt = None
                f_gt = None
                v_in = None
                f_in = None
                _subj= None
                volume_in, v_gt, f_gt, v_in, f_in,_subj = data
                allocated.append(torch.cuda.memory_allocated())
                # Calculate the size of each segment
                print(v_in.shape)
                segment_size = v_in.shape[1] // n
                segment_start = i * segment_size
                segment_end = segment_start + segment_size

                volume_in = volume_in.to(device)
                allocated.append(torch.cuda.memory_allocated())
                v_gt = v_gt[:,segment_start:segment_end,:].to(device)
                
                v_in = v_in.to(device)
                f_in = f_in.to(device)

                allocated.append(torch.cuda.memory_allocated())

                v_out = None
                if config.cortexGNN:
                    v_out = model(v=v_in, f=f_in, volume=volume_in,
                                n_smooth=config.n_smooth, lambd=config.lambd,
                                start = segment_start,end = segment_end)
                else:
                    v_out = model(v=v_in, f=f_in, volume=volume_in,
                                n_smooth=config.n_smooth, lambd=config.lambd)
                
                allocated.append(torch.cuda.memory_allocated())

                # Slicing the segment of interest from v_out
                v_out_segment = v_out[:, segment_start:segment_end, :]

                allocated.append(torch.cuda.memory_allocated())

                
    allocated.append(torch.cuda.memory_allocated())
    max_memory_usage = max(allocated)/ (1024 ** 3)#GiB
    print("max memory usage",max_memory_usage)
    data = [model_name, config.gnn_layers, max_memory_usage]
    
    # File path for the CSV
    csv_file_path = '/pialnn/memory_stats.csv'

    # Check if file exists, if not create, if yes append
    if not os.path.isfile(csv_file_path):
        # Writing headers and data to CSV
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'Layers', 'max_memory_usage_GiB'])
            writer.writerow(data)
    else:
        # Appending data to CSV without header
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    print("Finish evaluation.")
    print("----------------------------")
