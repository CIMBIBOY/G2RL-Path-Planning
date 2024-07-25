# script to store temporary copies, which are under repair

import random

def start_end_points(obs_coords, arr):
    """
    Generate start and end coordinates for dynamic obstacles.

    Input: 
    - obs_coords: coordinates of all dynamic obstacles
    - arr: the 0, 1 value map

    Output: list of [dynamic obstacle id, [start point coordinates, end point coordinates]]
    """
    coords = []
    h, w = arr.shape[:2]
    end_points = set()  # To keep track of all end points

    for i, start in enumerate(obs_coords):
        attempts = 0
        while attempts < 1000:
            h_new = random.randint(0, h-1)
            w_new = random.randint(0, w-1)
            new_point = [h_new, w_new]
            
            if (arr[h_new][w_new] == 0 and 
                new_point not in obs_coords and 
                new_point != start and  # This check ensures start != end
                tuple(new_point) not in end_points):
                
                coords.append([i, start + new_point])
                end_points.add(tuple(new_point))
                break
            
            attempts += 1
        
        if attempts == 1000:
            print(f"Warning: Could not find valid end point for obstacle {i} after 1000 attempts")
            return None  # Return None if we can't find a valid configuration

    return coords