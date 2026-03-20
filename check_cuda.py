import torch, sys
print('torch version:', torch.__version__)
print('CUDA compiled into torch:', torch.version.cuda)
print('CUDA available?', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print('Total memory (GB):', round(props.total_memory/1e9,2))
else:
    print('CUDA not available')
