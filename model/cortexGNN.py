import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import compute_normal
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops

from torch_geometric.nn import GATConv

#N layer GCN using pyg
class NLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,num_layers):
        super(NLayerGCN, self).__init__()
        assert num_layers > 1, "Number of layers should be greater than 1"

        # Initialize the GAT layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(pyg_nn.GCNConv(in_features, hidden_features))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(pyg_nn.GCNConv(hidden_features, hidden_features))

        # Output layer
        self.layers.append(pyg_nn.GCNConv(hidden_features, out_features))
    
    def forward(self, x, edge_index):
        # Create a subset of vertices
        x = x.squeeze()
        
        # Pass the subset of vertices and edges to the GCN layers
        # Assuming self.layers is a list of GCN layers in your model
        for layer in self.layers:
            x = layer(x, edge_index)

        print('x gcn',x.shape)
        return x


class NLayerGAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(NLayerGAT, self).__init__()
        assert num_layers > 1, "Number of layers should be greater than 1"

        # Initialize the GAT layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GATConv(in_features, hidden_features))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_features, hidden_features))

        # Output layer
        self.layers.append(GATConv(hidden_features, out_features))

    def forward(self, x, edge_index):
        x = x.squeeze()
        

        assert x.dim() == 2, f"Input should be 2D, but got shape {x.shape}"

        
        # Apply all but the last GAT layer with ReLU activation
        for layer in self.layers[:-1]:
            
            x = F.relu(layer(x, edge_index))

        # Apply the last GAT layer with tanh activation
        x = torch.tanh(self.layers[-1](x, edge_index))
        

        return x


"""
Extract node features for two layer gcn
"""

class NodeFeatureNet(nn.Module):
    def __init__(self, nc=64, K=5, n_scale=1):
        super(NodeFeatureNet, self).__init__()

        # mlp layers
        self.fc1 = nn.Linear(6, nc)
        
        # for local convolution operation
        self.localconv = nn.Conv3d(n_scale, nc, (K, K, K))
        # self.localfc = nn.Linear(nc, nc)#remove
        
        self.n_scale = n_scale
        self.nc = nc
        self.K = K

    def forward(self, v, f, volume,start,end):
        #normal = compute_normal(v, f)    # compute normal

        #normal = normal[:,start:end,:]
        
        # point feature
        v = v[:,start:end,:]
        
        #x = torch.cat([v, normal], 2)
        x = torch.cat([v, torch.zeros_like(v)], 2)
        

        x = F.leaky_relu(self.fc1(x), 0.15)
        
        
        # local feature
        cubes = self.cube_sampling(v, volume)    # extract K^3 cubes
        
        x_local = self.localconv(cubes)#Todo:consider reducing
        

        x_local = x_local.view(1, v.shape[1], self.nc)
        

        # x_local = self.localfc(x_local)#Todo:consider removing
        

        # fusion
        x = torch.cat([x, x_local], 2)
        

        return x    # node features
    
    def initialize(self, L, W, H, device=None):
        """initialize necessary constants"""
        
        LWHmax = max([L,W,H])
        

        self.LWHmax = LWHmax
        

        # rescale to [-1, 1]
        self.rescale = torch.Tensor([L/LWHmax, W/LWHmax, H/LWHmax]).to(device)
        

        # shape of mulit-scale image pyramid
        self.pyramid_shape = torch.zeros([self.n_scale, 3]).to(device)
        

        for i in range(self.n_scale):
            self.pyramid_shape[i] = torch.Tensor([L/(2**i),
                                                  W/(2**i),
                                                  H/(2**i)]).to(device)
        

        # for threshold
        self.lower_bound = torch.tensor([(self.K-1)//2,
                                         (self.K-1)//2,
                                         (self.K-1)//2]).to(device)
        

        # for storage of sampled cubes
        self.cubes_holder = torch.zeros([1, self.n_scale,
                                         self.K, self.K, self.K]).to(device)
        


    def cube_sampling(self, v, volume):

        # for storage of sampled cubes
        cubes = self.cubes_holder.repeat(v.shape[1],1,1,1,1)
        

        # 3D MRI volume
        vol_ = volume.clone()
        

        for n in range(self.n_scale):   # multi scales
            if n > 0:
                vol_ = F.avg_pool3d(vol_, 2)    # down sampling
            vol = vol_[0,0]
            
            
            # find corresponding position
            indices = (v[0] + self.rescale) * self.LWHmax / (2**(n+1))
            

            indices = torch.round(indices).long()
            

            indices = torch.max(torch.min(indices, self.pyramid_shape[n]-3),
                                self.lower_bound).long()
            
            
            # sample values of each cube
            for i in [-2,-1,0,1,2]:
                for j in [-2,-1,0,1,2]:
                    for k in [-2,-1,0,1,2]:
                        cubes[:,n,2+i,2+j,2+k] = vol[indices[:,2]+i,
                                                     indices[:,1]+j,
                                                     indices[:,0]+k]
                        

        return cubes

"""
Deformation Block
nc: number of channels
K: kernal size for local conv operation
n_scale: num of layers of image pyramid
"""

class DeformBlockGNN(nn.Module):
    def __init__(self, nc=64, K=5, n_scale=1,sf=.1,gnn_layers=2,gnnVersion=1):
        super(DeformBlockGNN, self).__init__()
        self.nodeFeatureNet = NodeFeatureNet(nc=nc,K=K,n_scale=n_scale)
        #chatgpt,declare a custom 2 layer gcn here that takes features from nodeFeatureNet to predict node features, use a tanh nonlinearity at the end instead of softmax. 
        self.n_scale = n_scale
        self.nc = nc
        self.K = K
        self.sf = sf
        if gnnVersion == 1:
            self.gcn = NLayerGAT(nc*2, nc, 3,gnn_layers)  # Adjust dimensions as needed
            print('NLayerGAT',gnn_layers)
        else:
            self.gcn = NLayerGCN(nc*2, nc, 3,gnn_layers)  # Adjust dimensions as needed
            print('NLayerGCN',gnn_layers)
            
    def forward(self, v, f, volume,start,end,edge_index):
        coord = v.clone()
        print('edge index',edge_index.shape)
        # Filter edge_index to include only edges within the specified range
        # and adjust the indices to align with the new vertex indexing
        mask = (edge_index[0] >= start) & (edge_index[0] < end) & \
            (edge_index[1] >= start) & (edge_index[1] < end)
        edge_index_subset = edge_index[:, mask] - start
        print('edge_index_subset',edge_index_subset.shape)
        x_subset = self.nodeFeatureNet(v, f, volume,start,end)

        print('x.shape')
        x = self.gcn(x_subset, edge_index_subset)*self.sf #threshold the deformation like before
        print('x.shape',x.shape)

        coord[:,start:end,:] = coord[:,start:end,:] + x
        
        return coord    # v=v+dvreturn    # node features
    
    def initialize(self, L, W, H, device=None):
        """initialize necessary constants"""
        self.nodeFeatureNet.initialize(L,W,H,device = device)
        



"""
PialNN with 3 deformation blocks + 1 Laplacian smoothing layer
"""
class CortexGNN(nn.Module):
    def __init__(self, nc=64, K=5, n_scale=1, num_blocks=3,sf=.1,gnn_layers=2,gnnVersion=1):
        super(CortexGNN, self).__init__()
        
        self.block1 = DeformBlockGNN(nc, K, n_scale,sf,gnn_layers=gnn_layers,gnnVersion=gnnVersion)
        #self.block2 = DeformBlockGNN(nc, K, n_scale,sf,gnn_layers=gnn_layers,gnnVersion=gnnVersion)
        #self.block3 = DeformBlockGNN(nc, K, n_scale,sf,gnn_layers=gnn_layers,gnnVersion=gnnVersion)
        #self.smooth = LaplacianSmooth(3, 3, aggr='mean')

    def forward(self, v, f, volume, n_smooth=1, lambd=1.0,start = None, end = None):
        assert v.shape[0] == 1
        assert f.shape[0] == 1
        assert v.shape[1] != 1
        assert f.shape[1] != 1
        edge_list = torch.cat([f[0,:,[0,1]],
                               f[0,:,[1,2]],
                               f[0,:,[2,0]]], dim=0).transpose(1,0)#moved from after x
        

        edge_index_with_loops = add_self_loops(edge_list)[0]
        

        x = self.block1(v, f, volume,start,end,edge_index_with_loops)
        #x = self.block2(x, f, volume,start,end,edge_index_with_loops)
        
        
        # for i in range(n_smooth):
        #      x = self.smooth(x, edge_list, lambd=lambd)
             

        return x
    
    def initialize(self, L=256, W=256, H=256, device=None):
        # for i in range(len(self.blocks)):
        #     self.blocks[i].initialize(L,W,H,device)
        self.block1.initialize(L,W,H,device)
        #self.block2.initialize(L,W,H,device)
        



"""
LaplacianSmooth() is a differentiable Laplacian smoothing layer.
The code is implemented based on the torch_geometric.nn.conv.GraphConv.
For original GraphConv implementation, please see
https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/graph_conv.py


x: the coordinates of the vertices, (|V|, 3).
edge_index: the list of edges, (2, |E|), e.g. [[0,1],[1,3],...]. 
lambd: weight for Laplacian smoothing, between [0,1].
out: the smoothed vertices, (|V|, 3).
"""

from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class LaplacianSmooth(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int,
                                                     int]], out_channels: int,
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super(LaplacianSmooth, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None, lambd=0.5) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        out = lambd * out 

        x_r = x[1]

        if x_r is not None:
            out += (1-lambd) * x_r

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)