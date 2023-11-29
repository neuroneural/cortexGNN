import torch
import gc


import nvidia_smi

def get_memory_usage_in_gb():
    nvidia_smi.nvmlInit()  # Initialize NVML
    device_count = nvidia_smi.nvmlDeviceGetCount()

    used_memory = []
    for i in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        used_memory.append(info.used / 1024**3)  # Memory in GB

    nvidia_smi.nvmlShutdown()  # Shutdown NVML
    return used_memory[0]

# used_memory_gbs = get_gpu_used_memory()
# print(used_memory_gbs)


# def get_memory_usage_in_gb():
#     torch.cuda.synchronize()  # Wait for CUDA operations to finish
#     allocated = torch.cuda.memory_allocated()  # Get the current GPU memory usage
#     return allocated / 1024**3  # Convert bytes to gigabytes


device = torch.device('cuda')
# Before the prediction
mem_alloc = []

# Loading the tensors from the file
loaded_data = torch.load('../tensors.pt')

# Extracting individual tensors
volume_in = loaded_data['volume_in'].to(device)
v_gt = loaded_data['v_gt'].to(device)
f_gt = loaded_data['f_gt'].to(device)
v_in = loaded_data['v_in'].to(device)
f_in = loaded_data['f_in'].to(device)

print("volume_in",volume_in.shape)
print("v_gt",v_gt.shape)
print("f_gt",f_gt.shape)
print("v_in",v_in.shape)
print("f_in",f_in.shape)

from pialnn_mod import PialNN
import torch.optim as optim

train = True
model = None
optimizer = None
if train:
    model = PialNN().to(device)
    # Assume 'model' is your neural network model
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # lr is the learning rate

else:
    model = PialNN()
    model.eval()
    model.to(device)

model.initialize(192,224,192,device)


print(get_memory_usage_in_gb())
torch.cuda.empty_cache()


for i in range(1):
        # After prediction, before loss
    #mem_alloc.append(get_memory_usage_in_gb())

    v_out = model(v=v_in, f=f_in, volume=volume_in,
                            n_smooth=1, lambd=1.0)
    print ("v_out", v_out.shape)

    # Calculate loss
    if train:    
        loss  = torch.nn.MSELoss()(v_out, v_gt) * 1e+3
                    
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # After backpropagation
    mem_alloc.append(get_memory_usage_in_gb())
    torch.cuda.empty_cache()
        

# Convert list to a PyTorch tensor
tensor = torch.tensor(mem_alloc)

# Find the maximum value
max_value = torch.median(tensor)

print(f"Peak Gpu allocation: {max_value.item()}")


