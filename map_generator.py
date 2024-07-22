from PIL import Image
import numpy as np
import random

'''
This script is responsible for generating maps and converting map data. 
It includes functions to:

	1.	Create random maps with obstacles placed randomly.
	2.	Generate guide maps with specific coordinates marked.
	3.	Convert maps to value maps where obstacles are represented by 1 and free spaces by 0.
	4.	Generate start and end points for dynamic obstacles.
	5.	Create global and local guidance maps based on the paths of dynamic obstacles.
	6.	Generate heuristic values for the A* algorithm based on Manhattan distances.
'''

def random_map(w,h, n_static, map_name = "random-1", color_coord = [50,205,50]):

    static_coord_width = [random.randint(0,w-1) for i in range(n_static)]
    static_coord_height = [random.randint(0,h-1) for i in range(n_static)]

    data = np.ones((h, w, 3), dtype=np.uint8)*255

    for i in range(n_static):
        data[static_coord_height[i], static_coord_width[i]] = color_coord
    
    img = Image.fromarray(data, 'RGB')
    img.save(f'data/{map_name}.png')



def guide_map(w,h,h_coord,w_coord, map_name = "guide-1", color_coord = [50,205,50]):

    assert len(h_coord) == len(w_coord), "Coordinates length is not same"
    data = np.ones((h, w, 3), dtype=np.uint8)*255

    for i in range(len(h_coord)):
        data[h_coord[i], w_coord[i]] = color_coord
    
    img = Image.fromarray(data, 'RGB')
    img.save(f'data/{map_name}.png')


def map_to_value(arr):
    """
    Generate an array of the map and convert the RGB values ​​into 0 and 1. 
    0 means passable and 1 means static obstacles (black)
    """
    h,w = arr.shape[:2]
    new_arr = np.zeros(shape=(h,w), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            cell_coord = arr[i,j]
            if cell_coord[0] == 0 and cell_coord[1] == 0 and cell_coord[2] == 0:
                new_arr[i,j] = 1

    return new_arr

def start_end_points(obs_coords, arr):
    """
    Generate start and end coordinates

    Input: coordinates of all dynamic obstacles and the 0, 1 value map

    Output: list of end point coordinates of each dynamic obstacle
    the list is [dynamic obstacle id, [start point coordinates, end point coordinates]  
    
    """
    coords = []
    # Map size
    h,w = arr.shape[:2]
    for i,c in enumerate(obs_coords):
        while True:
            h_new = random.randint(0,h-1)
            w_new = random.randint(0,w-1)
            # The end point is randomly selected. 
            # The end point must be optional (not a static obstacle) and cannot be the starting point. 
            # TODO: Might be an issue here. There shouldn't be another endpoint, not in coords.
            if arr[h_new][w_new] == 0 and [h_new, w_new] not in obs_coords and [h_new, w_new] != c:
                c.extend([h_new, w_new])
                coords.append([i, c])
                break
        # print(f"Generated for {i} obstacle.")
    return coords

def global_guidance(paths, arr):

    guidance = np.ones((len(arr), len(arr[0])), np.uint8)*255
    for x,y in paths:
        guidance[x,y] = 105

    return guidance

def local_guidance(paths, arr, idx):
    if idx < len(paths):
        arr[paths[idx]] = [255,255,255]
        
    return arr

def heuristic_generator(arr, end):
    """
    Generate a table of heuristic function values
    """
    # print(f"Input arr shape: {arr.shape}")
    
    if len(arr.shape) == 2:
        h, w = arr.shape
    elif len(arr.shape) == 3:
        h, w, _ = arr.shape
    else:
        raise ValueError("Invalid input array shape")
    
    # Check if end coordinates are within bounds
    if not (0 <= end[0] < h and 0 <= end[1] < w):
        raise ValueError(f"End coordinates {end} are out of bounds for the map size ({h}, {w})")

    # Initialize heuristic map with zeroes
    h_map = [[0 for _ in range(w)] for _ in range(h)]
    
    # Compute Manhattan distances
    for i in range(h):
        for j in range(w):
            h_map[i][j] = abs(end[0] - i) + abs(end[1] - j)

    return h_map