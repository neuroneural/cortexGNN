import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import load_config
from data.dataload import load_data, BrainDataset
from model.pialnn import PialNN
from utils import compute_normal, save_mesh_obj, compute_distance

import datetime
import nvidia_smi
import time

from csv import writer

def write_time2csv(model_name, t_sec, loading = False):
    filename = ''
    if loading is False:
        filename='/data/users2/washbee/speedrun/bm.events.csv'
    else:
        filename='/data/users2/washbee/speedrun/bm.loading.csv' 
    List = [model_name, t_sec]
    with open(filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()


nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()

# Helper Function
def printModelSize(model):
    # print(dir(model))
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('\n\n\n\n')
    print('model size: {:.3f}MB'.format(size_all_mb))
    print('\n\n\n\n')

def printSpaceUsage():
    nvidia_smi.nvmlInit()
    msgs = ""
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        #print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
        msgs += '\n'
        msgs += "Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used)
    nvidia_smi.nvmlShutdown()
    return msgs

if __name__ == '__main__':
    a = datetime.datetime.now()

    # log for GPU utilization
    
    GPU_msgs = []

    ### Set Stage
    stage = '0 - set device'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')


    """set device"""
    if torch.cuda.is_available():
        device_name = "cuda:0"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    ### Set Stage
    stage = '0 - load configuration'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')


    """load configuration"""
    config = load_config()


    ### Set Stage
    stage = '0 - load dataset'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')


    """load dataset"""
    print("----------------------------")
    print("Start loading dataset ...")
    test_data = load_data(data_path = config.data_path,
                          hemisphere = config.hemisphere)
    n_data = len(test_data)
    L,W,H = test_data[0].volume[0].shape  # shape of MRI
    LWHmax = max([L,W,H])

    test_set = BrainDataset(test_data)
    testloader = DataLoader(test_set, batch_size=1, shuffle=True)
    print("Finish loading dataset. There are total {} subjects.".format(n_data))
    print("----------------------------")


    ### Set Stage
    stage = '0 - load model'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')
    
    """load model"""
    print("Start loading model ...")
    
    torch.cuda.empty_cache()

    print(printSpaceUsage())
    
    model = PialNN(config.nc, config.K, config.n_scale).to(device)
    model.load_state_dict(torch.load("./ckpts/model/pialnn_model_lh_200epochs.pt", map_location=device))
    model.initialize(L, W, H, device)
    print("Finish loading model")
    print(printSpaceUsage())
    print("----------------------------")
    
    print('pial model')
    printModelSize(model)
    ### Set Stage
    stage = '0 - evaluation'
    msgs = printSpaceUsage()
    GPU_msgs.append(stage + msgs + '\n\n\n')

    for msg in GPU_msgs:
        print(msg)


    """evaluation"""
    print("Start evaluation ...")
    print(printSpaceUsage())
    
    b = datetime.datetime.now()
    t_sec = (b-a).total_seconds()
    
    write_time2csv('PialNN', t_sec,loading = True)
    
    
    with torch.no_grad():
        CD = []
        AD = []
        HD = []
        for idx, data in tqdm(enumerate(testloader)):
            a = datetime.datetime.now()
            volume_in, v_gt, f_gt, v_in, f_in = data

            volume_in = volume_in.to(device)
            v_gt = v_gt.to(device)
            f_gt = f_gt.to(device)
            v_in = v_in.to(device)
            f_in = f_in.to(device)
            
            torch.cuda.empty_cache()
            print('before v_pred = model(...)\n',printSpaceUsage())
            # set n_smooth > 1 if the mesh quality is not good
            v_pred = model(v=v_in, f=f_in, volume=volume_in,
                           n_smooth=config.n_smooth, lambd=config.lambd)
            print('after v_pred = model(...)\n',printSpaceUsage())
            
            v_pred_eval = v_pred[0].cpu().numpy() * LWHmax/2 + [L/2,W/2,H/2]
            f_pred_eval = f_in[0].cpu().numpy()
            #v_gt_eval = v_gt[0].cpu().numpy() * LWHmax/2 + [L/2,W/2,H/2]
            #f_gt_eval = f_gt[0].cpu().numpy()

            #compute distance-based metrics
            #cd, assd, hd = compute_distance(v_pred_eval, v_gt_eval,
            #                                f_pred_eval, f_gt_eval, config.n_test_pts)
            #CD.append(cd)
            #AD.append(assd)
            #HD.append(hd)

            if config.save_mesh_eval:
                path_save_mesh = "./ckpts/bm/pialnn_mesh_eval_"\
                        +config.hemisphere+"_subject"+str(idx)+".obj"

                normal = compute_normal(v_pred, f_in)
                n_pred_eval = normal[0].cpu().numpy()
                save_mesh_obj(v_pred_eval, f_pred_eval, n_pred_eval, path_save_mesh)
                
                #path_save_mesh = "./ckpts/eval/pialnn_mesh_eval_"\
                #        +config.hemisphere+"_subject"+str(idx)+"_gt.obj"

                #normal = compute_normal(v_gt, f_gt)
                #n_pred_eval = normal[0].cpu().numpy()
                #save_mesh_obj(v_gt_eval, f_gt_eval, n_pred_eval, path_save_mesh)

            b = datetime.datetime.now()
            t_sec = (b-a).total_seconds()
            write_time2csv('PialNN', t_sec)
            print('total seconds for one batch is {}'.format(t_sec))
            ### Set Stage
            stage = '0 - END'
            msgs = printSpaceUsage()
            GPU_msgs.append(stage + msgs + '\n\n\n')
             
            for msg in GPU_msgs:
                print(msg)

            #exit()

    #print("CD: Mean={}, Std={}".format(np.mean(CD), np.std(CD)))
    #print("AD: Mean={}, Std={}".format(np.mean(AD), np.std(AD)))
    #print("HD: Mean={}, Std={}".format(np.mean(HD), np.std(HD)))
    #print("Finish evaluation.")
    #print("----------------------------")

