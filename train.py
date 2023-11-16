import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import load_config
from data.dataload import load_data, BrainDataset
from model.pialnn import PialNN
from utils import compute_normal, save_mesh_obj


if __name__ == '__main__':
    
    """set device"""
    if torch.cuda.is_available():
        device_name = "cuda:0"
    else:
        device_name = "cpu"
    device = torch.device(device_name)


    """load configuration"""
    config = load_config()
    
    
    """load data"""
    print("----------------------------")
    print("Start loading dataset ...")
    all_data = load_data(data_path=config.data_path,
                         hemisphere=config.hemisphere)

    L,W,H = all_data[0].volume[0].shape    # shape of MRI
    LWHmax = max([L,W,H])
    n_data = len(all_data)
    
    # split training / validation
    n_train = int(n_data * config.train_data_ratio)
    n_valid = n_data - n_train
    train_data = all_data[:n_train]
    valid_data = all_data[n_train:] 
    train_set = BrainDataset(train_data)
    valid_set = BrainDataset(valid_data)

    # batch size can only be 1
    trainloader = DataLoader(train_set, batch_size=1, shuffle=True)
    validloader = DataLoader(valid_set, batch_size=1, shuffle=False)
    
    print("Finish loading dataset. There are total {} subjects.".format(n_data))
    print("----------------------------")
    
    
    """load model"""
    print("Start loading model ...")
    model = PialNN(config.nc, config.K, config.n_scale).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.initialize(L, W, H, device)
    print("Finish loading model")
    print("----------------------------")
    
    allocated = []
    allocated.append(torch.cuda.memory_allocated())
    
    """training"""
    print("Start training {} epochs ...".format(config.n_epoch))    
    n = 4
            
    for epoch in tqdm(range(config.n_epoch+1)):
        avg_loss = []
        allocated.append(torch.cuda.memory_allocated())

        for idx, data in enumerate(trainloader):
            allocated.append(torch.cuda.memory_allocated())
            ##

            
            # Choose n (e.g., n = 4 for a quarter)
            # Iterate over the segment
            for i in range(n):
                print('i',i)
                ##
                volume_in, v_gt, f_gt, v_in, f_in,_subj = data
                allocated.append(torch.cuda.memory_allocated())
                # Calculate the size of each segment
                segment_size = v_in.shape[1] // n
            
                # Choose the segment you want to iterate over (e.g., the first quarter)
                segment_start = i * segment_size
                segment_end = segment_start + segment_size

                volume_in = volume_in.to(device)
                allocated.append(torch.cuda.memory_allocated())
                v_gt = v_gt[:,segment_start:segment_end,:].to(device)
                v_in = v_in.to(device)
                f_in = f_in.to(device)
                
                allocated.append(torch.cuda.memory_allocated())

                optimizer.zero_grad()
                
                print('vin tr',v_in.shape)
                print('fin tr',f_in.shape)
                v_out = model(v=v_in, f=f_in, volume=volume_in,
                            n_smooth=config.n_smooth, lambd=config.lambd,
                            start = segment_start,end = segment_end)

                allocated.append(torch.cuda.memory_allocated())

                loss  = nn.MSELoss()(v_out[:,segment_start:segment_end,:], v_gt) * 1e+3
                avg_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                allocated.append(torch.cuda.memory_allocated())


        if config.report_training_loss:
            print("Epoch:{}, training loss:{}".format(epoch, np.mean(avg_loss)))

        
        if epoch % config.ckpts_interval == 0:
            print("----------------------------")
            print("Start validation ...")
            with torch.no_grad():
                error = []
                for idx, data in enumerate(validloader):
                    for i in range(n):
                        ##
                        volume_in, v_gt, f_gt, v_in, f_in,_subj = data
                        allocated.append(torch.cuda.memory_allocated())
                        # Calculate the size of each segment
                        segment_size = v_in.shape[1] // n

                        # Choose the segment you want to iterate over (e.g., the first quarter)
                        segment_start = i * segment_size
                        segment_end = segment_start + segment_size
                        volume_in = volume_in.to(device)
                        v_gt = v_gt[:,segment_start:segment_end,:].to(device)
                        v_in = v_in.to(device)
                        f_in = f_in.to(device)

                        v_out = model(v=v_in, f=f_in, volume=volume_in,
                                    n_smooth=config.n_smooth, 
                                    lambd=config.lambd,
                                    start = segment_start,
                                    end = segment_end)
                        error.append(nn.MSELoss()(v_out[:,segment_start:segment_end,:], v_gt).item() * 1e+3)
                
                print("Validation error:{}".format(np.mean(error)))
                allocated.append(torch.cuda.memory_allocated())

            if config.save_model:
                print('Save model checkpoints ... ')
                path_save_model = "./ckpts/model/pialnn_model_"+config.hemisphere+"_"+str(epoch)+"epochs.pt"
                torch.save(model.state_dict(), path_save_model)

            allocated.append(torch.cuda.memory_allocated())

            if config.save_mesh_train:
                print('Save pial surface mesh ... ')
                path_save_mesh = "./ckpts/mesh/pialnn_mesh_"+config.hemisphere+"_"+str(epoch)+"epochs.obj"

                normal = compute_normal(v_out, f_in)
                v_gm = v_out[0].cpu().numpy() * LWHmax/2  + [L/2,W/2,H/2]
                f_gm = f_in[0].cpu().numpy()
                n_gm = normal[0].cpu().numpy()

                save_mesh_obj(v_gm, f_gm, n_gm, path_save_mesh)


            allocated.append(torch.cuda.memory_allocated())
            max_memory_usage = max(allocated)

            # Print the maximum allocated GPU memory in GiB
            max_memory_usage_gib = max_memory_usage / (1024 ** 3)
            print(f"Maximum allocated GPU memory: {max_memory_usage_gib:.2f} GiB")

            print("Finish validation.")
            print("----------------------------")

    print("Finish training.")
    print("----------------------------")