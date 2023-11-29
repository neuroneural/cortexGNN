import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNet(nn.Module):
    def __init__(self,nc = 128,k=5):
        super(HyperNet, self).__init__()
        # Hypernetwork takes a 3D vertex norm vector as input
        self.fc1 = nn.Linear(3, nc)  # Intermediate layer, can adjust size
        self.fc2 = nn.Linear(nc, nc * k * k * k)  # Output layer
        self.k = k 
        self.nc = nc
    def forward(self, vertex_norms):
        x = F.relu(self.fc1(vertex_norms))  # Apply non-linearity
        print('x',x.shape)
        conv_weights = self.fc2(x)
        print('conv_weights',conv_weights.shape)
        # Reshape weights to match the convolution dimensions
        conv_weights = conv_weights.view(-1,self.nc, self.k, self.k, self.k)
        return conv_weights

class ConditionalConv3D(nn.Module):
    def __init__(self, nc):
        super(ConditionalConv3D, self).__init__()
        self.hypernet1 = HyperNet(nc=128, k=5)
        self.hypernet2 = HyperNet(nc=3, k=5)
        
    def forward(self, x, vertex_norms):
        # Generate dynamic weights from the hypernet
        dynamic_weights = self.hypernet1(vertex_norms)
        print('dynamic_weights',dynamic_weights.shape)
        # Apply 3D convolution with dynamic weights
        return F.conv3d(x, torch.randn(128,1,5,5,5),stride=(1,1,1))  # padding=2 for same output size
        
# Example usage
conditional_conv = ConditionalConv3D(nc=128)

# Example input data
input_data = torch.randn(12000,1, 5, 5, 5)  # [batch, channel, depth, height, width]

# Example vertex norm
vertex_norm = torch.randn(1,12000, 3)  # 3D vertex norm

# Apply the conditional convolution
output = conditional_conv(input_data, vertex_norm)

print('output',output.shape)
