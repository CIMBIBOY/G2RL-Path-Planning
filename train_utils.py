import torch
import random
import numpy as np
import time

# Function to set the seed for reproducibility
def set_seed(seed, determism):
    print(f"Runtime is seeded '{seed}' for all operations")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = determism

def print_cuda_info():
    if torch.cuda.is_available():
        print(f"CUDA is available. Version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available, training on CPU")

# Functions for debugging computation time
def debug_start(timestep = 0, str = 'a'):
    print(f"At timestep: {timestep} form debug operation num: {str}")
    start_time = time.time()
    return start_time

def debug_end(stime):
    end_time = time.time()
    operation_time = end_time - stime
    if operation_time > 0.1: print(f"Operation took {operation_time} s to complete")
