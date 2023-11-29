import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import load_config
from data.dataload import load_data, BrainDataset
from model.cortexGNN import CortexGNN
from model.pialnn import PialNN
from utils import compute_normal, save_mesh_obj

from pytorch3d.loss import chamfer_distance

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes


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
    mse=True
    all_data = load_data(data_path=config.data_path,
                          hemisphere=config.hemisphere,fsWIn=mse)

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

    best_val_error = float('inf')  # Initialize the best validation error

    # batch size can only be 1
    trainloader = DataLoader(train_set, batch_size=1, shuffle=True)
    validloader = DataLoader(valid_set, batch_size=1, shuffle=False)
    sf = .1
    
    if mse:
        print('MSE')
    else:
        print('Chamfer etc')
    print("Finish loading dataset. There are total {} subjects.".format(n_data))
    print("Training data length",len(train_data))
    print("Validation data length",len(valid_data))
    print('scaling factor ',sf)
    print("----------------------------")
        
    # vertices_clone and faces_clone are now tensors that can be used with PyTorch3D
    
    """load model"""
    print("Start loading model ...")
    
    model = None
    num_blocks = None        
    if config.cortexGNN:
        num_blocks = 1
        print("Model is CortexGNN")
        model = CortexGNN(config.nc, config.K, config.n_scale,num_blocks,sf,config.gnn_layers,config.gnnVersion).to(device)#todo:revise num_blocks
    else:
        num_blocks = 1    
        print("Model is PialNN")
        model = PialNN(config.nc, config.K, config.n_scale).to(device)#todo:revise 7
    
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.initialize(L, W, H, device)
    print("Finish loading model")
    print("----------------------------")
    
    allocated = []
    allocated.append(torch.cuda.memory_allocated())
    
    """training"""
    print("Start training {} epochs ...".format(config.n_epoch))    
    n = 1
    for epoch in tqdm(range(config.n_epoch+1)):
        avg_loss = []
        allocated.append(torch.cuda.memory_allocated())

        for idx, data in enumerate(trainloader):
            allocated.append(torch.cuda.memory_allocated())
            ##
            v_out = None#Todo: check support for submesh training.
                    
            for bl in range(num_blocks):
                # Choose n (e.g., n = 4 for a quarter)
                # Iterate over the segment
                for i in range(n):
                    ##
                    volume_in = None
                    v_gt = None
                    f_gt = None
                    v_in = None
                    f_in = None
                    _subj= None
                    if bl == 0:
                        volume_in, v_gt, f_gt, v_in, f_in,_subj = data
                    else:
                        volume_in, v_gt, f_gt, _, f_in,_subj = data
                        v_in = v_out.detach()    
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
                    v_out = None
                    if config.cortexGNN:
                        v_out = model(v=v_in, f=f_in, volume=volume_in,
                                    n_smooth=config.n_smooth, lambd=config.lambd,
                                    start = segment_start,end = segment_end)
                    else:
                        v_out = model(v=v_in, f=f_in, volume=volume_in,
                                    n_smooth=config.n_smooth, lambd=config.lambd)
                    
                    allocated.append(torch.cuda.memory_allocated())

                    # Assuming v_out and v_gt are your vertex sets in R^3
                    # And they are PyTorch tensors of shape [N, 3] where N is the number of vertices

                    # Slicing the segment of interest from v_out
                    v_out_segment = v_out[:, segment_start:segment_end, :]


                    # Since Chamfer Distance can be on a different scale compared to MSE,
                    # you might want to adjust the scaling factor (here it's left as is)
                    loss = None
                    if mse:
                        loss = nn.MSELoss()(v_out[:,segment_start:segment_end,:], v_gt) * 1e+3
                    else:
                        # Compute the Chamfer Distance
                        chamfer_dist, _ = chamfer_distance(v_out_segment, v_gt)
                        loss = chamfer_dist * 1e+3

                    if bl == (num_blocks-1):
                        avg_loss.append(loss.item())
                    
                    loss.backward()
                    optimizer.step()
                    v_out = v_out.detach()
                    allocated.append(torch.cuda.memory_allocated())


        if config.report_training_loss:
            print("Epoch:{}, training loss:{}".format(epoch, np.mean(avg_loss)))

        
        if epoch % config.ckpts_interval == 0:
            print("----------------------------")
            print("Start validation ...")
            with torch.no_grad():
                error = []
                for idx, data in enumerate(validloader):
                    v_out = None#Todo: check support for submesh training.
                    for bl in range(num_blocks):
                        for i in range(n):
                            ##
                            volume_in = None
                            v_gt = None
                            f_gt = None
                            v_in = None
                            f_in = None
                            _subj= None
                            if bl == 0:
                                volume_in, v_gt, f_gt, v_in, f_in,_subj = data
                            else:
                                volume_in, v_gt, f_gt, _, f_in,_subj = data
                                v_in = v_out.detach()    

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


                            if config.cortexGNN:
                                v_out = model(v=v_in, f=f_in, volume=volume_in,
                                            n_smooth=config.n_smooth, lambd=config.lambd,
                                            start = segment_start,end = segment_end)
                            else:
                                v_out = model(v=v_in, f=f_in, volume=volume_in,
                                            n_smooth=config.n_smooth, lambd=config.lambd)
                            # Assuming v_out and v_gt are your vertex sets in R^3
                            # And they are PyTorch tensors of shape [N, 3] where N is the number of vertices

                            # Slicing the segment of interest from v_out
                            v_out_segment = v_out[:, segment_start:segment_end, :]


                            # Since Chamfer Distance can be on a different scale compared to MSE,
                            # you might want to adjust the scaling factor (here it's left as is)
                            loss = None
                            if mse:
                                loss = nn.MSELoss()(v_out[:,segment_start:segment_end,:], v_gt) * 1e+3
                            else:
                                # Compute the Chamfer Distance
                                chamfer_dist, _ = chamfer_distance(v_out_segment, v_gt)
                                loss = chamfer_dist * 1e+3
                            
                            if bl == (num_blocks-1):
                                error.append(loss.item() )
        
                            v_out = v_out.detach()
                            
                
                print("Validation error:{}".format(np.mean(error)))
                allocated.append(torch.cuda.memory_allocated())

            gnnVersion = "pialnn"
            layers = "NA"
            if config.gnnVersion==0 and config.cortexGNN:
                gnnVersion="PialGCN"
            elif config.gnnVersion==1 and config.cortexGNN:
                gnnVersion="PialGAT" 
            elif not config.cortexGNN:
                gnnVersion="PialNN"
            else:
                assert False,'unsupported'
            
            if config.cortexGNN:
                layers = config.gnn_layers
                
            current_val_error = np.mean(error)
            print("Validation error:{}".format(current_val_error))

            # Check if the current validation error is less than the best validation error
            if current_val_error < best_val_error:
                best_model_path = f"./ckpts/model/{gnnVersion}_GNNlayers{config.gnn_layers}_mse_whitein_full_model_"+config.hemisphere+"_best.pt"
            
                best_val_error = current_val_error  # Update the best validation error
                # Save the model as the new best model
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved at epoch {epoch} with validation error {best_val_error}")

            
            if config.save_model:
                print('Save model checkpoints ... ')
                path_save_model = f"./ckpts/model/{gnnVersion}_GNNlayers{config.gnn_layers}_mse_whitein_full_model_"+config.hemisphere+"_"+str(epoch)+"epochs.pt"
                torch.save(model.state_dict(), path_save_model)

            allocated.append(torch.cuda.memory_allocated())

            if config.save_mesh_train:
                print('Save pial surface mesh ... ')
                path_save_mesh = f"./ckpts/mesh/{gnnVersion}_GNNlayers{config.gnn_layers}_mse_whitein_full_mesh_"+config.hemisphere+"_"+str(epoch)+"epochs.obj"
                normal = compute_normal(v_out, f_in)#Todo:remove unsqueeze.
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