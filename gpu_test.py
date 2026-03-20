import torch
print('torch version:', torch.__version__)
print('CUDA compiled into torch:', torch.version.cuda)
print('CUDA available?', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    # allocate a tensor on GPU
    x = torch.randn(1000, 1000, device='cuda')
    print('Allocated tensor on GPU')
    print('Memory allocated (GB):', round(torch.cuda.memory_allocated(0)/1e9, 2))
else:
    print('CUDA not available')
