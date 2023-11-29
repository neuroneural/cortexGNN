import torch
import gc
from torch.profiler import profile, record_function, ProfilerActivity


def get_memory_usage_in_gb():
    torch.cuda.synchronize()  # Wait for CUDA operations to finish
    allocated = torch.cuda.memory_allocated()  # Get the current GPU memory usage
    return allocated / 1024**3  # Convert bytes to gigabytes


device = torch.device('cuda')
# Before the prediction
mem_before_pred = get_memory_usage_in_gb()

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

model = PialNN().to(device)

# Assume 'model' is your neural network model
optimizer = optim.Adam(model.parameters(), lr=0.001)  # lr is the learning rate

model.initialize(192,224,192,device)
# for i in range(10):
#     v_out = model(v=v_in, f=f_in, volume=volume_in,
#                             n_smooth=1, lambd=1.0)
#     print ("v_out", v_out.shape)


#     # After prediction, before loss
#     mem_after_pred = get_memory_usage_in_gb()

#     # Calculate loss
#     loss  = torch.nn.MSELoss()(v_out, v_gt) * 1e+3
                
#     # After loss calculation, before backpropagation
#     mem_after_loss = get_memory_usage_in_gb()

#     # Backward pass and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # After backpropagation
#     mem_after_backprop = get_memory_usage_in_gb()

#     # Print memory usage
#     print(f"Memory Usage: Prediction - {mem_after_pred - mem_before_pred} GB, "
#             f"Loss Calculation - {mem_after_loss - mem_after_pred} GB, "
#             f"Backpropagation - {mem_after_backprop - mem_after_loss} GB")


# Assuming the rest of your setup code is the same

# Start the profiling
with profile(activities=[ProfilerActivity.CPU, 
                         ProfilerActivity.CUDA], 
             record_shapes=True, 
             profile_memory=True,  # Set this to True to report memory usage
             with_stack=True) as prof:

    for i in range(10):
        # Forward pass
        with record_function("MODEL_FORWARD"):
            v_out = model(v=v_in, f=f_in, volume=volume_in, n_smooth=1, lambd=1.0)

        # Calculate loss
        with record_function("LOSS_CALCULATION"):
            loss = torch.nn.MSELoss()(v_out, v_gt) * 1e+3

        # Backward pass and optimize
        with record_function("BACKWARD_AND_OPTIMIZE"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# End of profiling
prof.export_chrome_trace("trace.json")  # Exports the trace to a file


import json

def sum_memory_usage(trace_file):
    with open(trace_file, 'r') as file:
        trace_data = json.load(file)
    
    total_memory_usage = 0
    for event in trace_data['traceEvents']:
        # Check if memory is reported in the event
        if 'args' in event and 'allocated_memory' in event['args']:
            # Sum up the memory usage
            total_memory_usage += event['args']['allocated_memory']

    return total_memory_usage

# Path to the trace file
trace_file_path = 'trace.json'

# Calculate the total memory usage
total_memory_usage = sum_memory_usage(trace_file_path)
print(f"Total Memory Usage: {total_memory_usage} bytes")
