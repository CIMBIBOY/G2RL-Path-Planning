from PIL import Image
import numpy as np
import matplotlib as plt 

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

def random_map(w, h, n_static, map_name="random-1", color_coord=[50, 205, 50], rng=None):
    if rng is None:
        rng = np.random  # Use the global numpy RNG if none is provided

    static_coord_width = [rng.integers(0, w) for i in range(n_static)]
    static_coord_height = [rng.integers(0, h) for i in range(n_static)]

    data = np.ones((h, w, 3), dtype=np.uint8) * 255

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

    # Eval debug map_to_value conversion call
    # print("Converting map to value array")
    
    h, w = arr.shape[:2]
    new_arr = np.zeros(shape=(h,w), dtype=np.int8)
    obstacle_count = 0
    for i in range(h):
        for j in range(w):
            cell_coord = arr[i,j]
            if cell_coord[0] == 0 and cell_coord[1] == 0 and cell_coord[2] == 0:
                new_arr[i,j] = 1
                obstacle_count += 1
    
    if np.all(new_arr == 0):
        print("Warning: All-zero value map")
    # Eval Debug static + dynamic object count along with cell num
    # print(f"Identified {obstacle_count} obstacles out of {h*w} cells")
    
    return new_arr


def start_end_points(obs_coords, arr, rng=None):
    """
    Generate start and end coordinates for dynamic obstacles.

    Input: 
    - obs_coords: coordinates of all dynamic obstacles
    - arr: the 0, 1 value map

    Output: list of [dynamic obstacle id, [start point coordinates, end point coordinates]]
    """

    if rng is None:
        rng = np.random  # Use the global numpy RNG if none is provided

    coords = []
    h, w = arr.shape[:2]
    end_points = set()

    for i, start in enumerate(obs_coords):
        attempts = 0
        while attempts < 1000:
            h_new = rng.integers(0, h)
            w_new = rng.integers(0, w)
            new_point = [h_new, w_new]
            
            if (arr[h_new][w_new] == 0 and 
                new_point not in obs_coords and 
                new_point != start and 
                tuple(new_point) not in end_points):
                
                coords.append([i, start + new_point])
                end_points.add(tuple(new_point))
                break
            
            attempts += 1
        
        if attempts == 1000:
            print(f"Warning: Could not find valid end point for obstacle {i} after 1000 attempts")
            return None  # Return None if we can't find a valid configuration

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

