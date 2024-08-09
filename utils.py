import numpy as np
import math
 
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