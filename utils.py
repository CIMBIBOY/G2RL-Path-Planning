import numpy as np

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