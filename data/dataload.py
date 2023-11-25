import os
import numpy as np
import torch
from tqdm import tqdm
import nibabel as nib
from torch.utils.data import Dataset
import re

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from utils import compute_normal, save_mesh_obj
"""
volume: brain MRI volume
v_in: vertices of input white matter surface
f_in: faces of ground truth pial surface
v_gt: vertices of input white matter surface
f_gt: faces of ground truth pial surface
"""

class BrainData():
    def __init__(self, volume, v_in, v_gt, f_in, f_gt,subject_id,fsWIn = True):
        self.v_in = torch.Tensor(v_in)
        self.v_gt = torch.Tensor(v_gt)
        self.f_in = torch.LongTensor(f_in)
        self.f_gt = torch.LongTensor(f_gt)
        self.volume = torch.Tensor(volume).unsqueeze(0)
        self.sub_id = int(subject_id)
        self.fsWIn = fsWIn
class BrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        brain = self.data[i]
        return brain.volume, brain.v_gt, \
                brain.f_gt, brain.v_in, brain.f_in, brain.sub_id

    
def load_mri(path):
    
    brain = nib.load(path)
    brain_arr = brain.get_fdata()
    brain_arr = brain_arr / 255.
    
    # ====== change to your own transformation ======
    # transpose and clip the data to [192,224,192]
    brain_arr = brain_arr.transpose(1,2,0)
    brain_arr = brain_arr[::-1,:,:]
    brain_arr = brain_arr[:,:,::-1]
    brain_arr = brain_arr[32:-32, 16:-16, 32:-32]
    #================================================
    
    return brain_arr.copy()


def load_surf(path):
    v, f = nib.freesurfer.io.read_geometry(path)

    # ====== change to your own transformation ======
    # transpose and clip the data to [192,224,192]
    v = v[:,[0,2,1]]
    v[:,0] = v[:,0] - 32
    v[:,1] = - v[:,1] - 15
    v[:,2] = v[:,2] - 32

    # normalize to [-1, 1]
    v = v + 128
    v = (v - [96, 112, 96]) / 112
    f = f.astype(np.int32)
    #================================================

    return v, f

def load_surf_pytorch3D(path):
    device = torch.device("cpu")
    mesh = load_objs_as_meshes([path], device=device)
    v = mesh.verts_list()[0].numpy()
    f = mesh.faces_list()[0].numpy()
           
    v = v[:,[0,2,1]]*.9
    v[:,0] = v[:,0] - 32
    v[:,1] = - v[:,1] - 15
    v[:,2] = v[:,2] - 32

    # normalize to [-1, 1]
    v = v + 128
    v = (v - [96, 112, 96]) / 112
    f = f.astype(np.int32)
    #================================================

    return v, f
    
    
def load_data(data_path, hemisphere,fsWIn=True):
    """
    data path: path of dataset
    """
    
    subject_lists = sorted(os.listdir(data_path))

    dataset = []
    
    for i in tqdm(range(len(subject_lists))):
        
        subid = subject_lists[i]
        

        # load brain MRI
        volume = load_mri(data_path + subid + '/mri/orig.mgz')

        # load ground truth pial surface
        v_gt, f_gt = load_surf(data_path + subid + '/surf/' + hemisphere + '.pial')
        #v_gt, f_gt = load_surf(data_path + subid + '/surf/' + hemisphere + '.pial.deformed')

        # load input white matter surface
        v_in = None
        f_in = None
        if fsWIn:
            v_in, f_in = load_surf(data_path + subid + '/surf/' + hemisphere + '.white')
            #         v_in, f_in = load_surf(data_path + subid + '/surf/' + hemisphere + '.white.deformed')
        else:#Todo:cleanup, debug
            v_in, f_in = load_surf_pytorch3D("/pialnn/resources/mods/lh_white_smooth_140k_mod_pial.obj")

        subid = str(subid)
        numbers = re.findall('\d+', subid)[0]

        print("numbers",numbers)  # Outputs: ['100', '20']

        #subid = subid.replace("'", "").replace("\\", "").replace("(", "").replace(")", "")
        braindata = BrainData(volume=volume, v_gt=v_gt, f_gt=f_gt,
                              v_in=v_in, f_in=f_in, subject_id=numbers,fsWIn=fsWIn)#Todo: make configurable
        dataset.append(braindata)

    return dataset


