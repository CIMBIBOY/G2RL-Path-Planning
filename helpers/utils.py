import numpy as np
import math
import torch
import random
import time
 
# TODO : check this method is local guidance correctly padded to the rgb chanels 
def symmetric_pad_array(input_array: np.ndarray, target_shape: tuple, pad_val: int) -> np.ndarray:

    for dim_in, dim_target in zip(input_array.shape, target_shape):
        if dim_target < dim_in:
            raise Exception("`target_shape` should be greater or equal than `input_array` shape for each axis.")

    pad_width = []
    for dim_in, dim_target  in zip(input_array.shape, target_shape):
        if (dim_in-dim_target)%2 == 0:
            pad_width.append((int(abs((dim_in-dim_target)/2)), int(abs((dim_in-dim_target)/2))))
        else:
            pad_width.append((int(abs((dim_in-dim_target)/2)), (int(abs((dim_in-dim_target)/2))+1)))
    
    return np.pad(input_array, pad_width, 'constant', constant_values=pad_val)

def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

def calculate_max_steps(path_len):
    min_path_length = 1
    max_path_length = 100
    min_steps = 42
    max_steps = 516

    if path_len <= min_path_length:
        return min_steps
    elif path_len >= max_path_length:
        return max_steps

    # Normalize the path length to a value between 0 and 1
    normalized_length = (path_len - min_path_length) / (max_path_length - min_path_length)

    # Use a combination of logarithmic and power functions with adjusted parameters
    log_component = math.log(1 + 5 * normalized_length) / math.log(6)
    power_component = normalized_length ** 0.8

    # Combine the components with adjusted weights
    combined = 0.7 * log_component + 0.3 * power_component

    # Scale to our desired range and round to nearest integer
    return round(min_steps + (max_steps - min_steps) * combined)

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

print("Path Length | Max Steps")
print("------------------------")
for length in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]:
    max_steps = calculate_max_steps(length)
    print(f"{length:11d} | {max_steps:9d}")