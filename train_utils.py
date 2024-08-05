import torch
import random
import numpy as np

# Function to set the seed for reproducibility
def set_seed(seed):
    print(f"Runtime is seeded '{seed}' for all operations")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior in CUDA (optional but useful for debugging)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def print_cuda_info():
    if torch.cuda.is_available():
        print(f"CUDA is available. Version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")
