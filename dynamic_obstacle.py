from collections import defaultdict
import os
from PIL import Image
from numpy import array, asarray
import numpy as np
import random

def initialize_objects(arr, n_dynamic_obst = 10):
    """
    Input: array of initial map, number of dynamic obstacles

    Output: array of initial positions of all dynamic obstacles and images after adding dynamic obstacles

    """
    arr = arr.copy()
    coord = []
    h,w = arr.shape[:2]

    while n_dynamic_obst > 0:
        h_obs = random.randint(0,h-1)
        w_obs = random.randint(0,w-1)

        cell_coord = arr[h_obs, w_obs]
        # When RGB is 0,0,0, it is black, indicating a static obstacle and cannot be generated.
        if cell_coord[0] != 0 and cell_coord[1] != 0 and cell_coord[2] != 0:
            # Dynamic obstacles are orange
            arr[h_obs, w_obs] = [255,165,0]
            n_dynamic_obst -= 1
            coord.append([h_obs,w_obs])
    
    return coord, arr

def manhattan_distance(x_st, y_st, x_end, y_end):
    # Returns the Manhattan distance
    return abs(x_end - x_st) + abs(y_end - y_st)

def update_coords(coords, inst_arr, agent, time_idx, width, global_map, direction, agent_old_coordinates, cells_skipped, dist):
    
    """ 
    Update coordinates

    Input: all paths, a map containing all information, agent id, time, local field of view size, global navigation map, 
        movement direction [x, y], coordinates at the last moment, number of grids skipped, distance

    Output: local field of view, local navigation map, global navigation map, whether to act, reward, 
        number of skipped grids, updated map, updated coordinates, distance

    """

    h,w = inst_arr.shape[:2]
    
    local_obs = np.array([])
    local_map = np.array([])
    agent_reward = 0

    # Get the path of the agent
    coord = coords[agent]

    agentDone = False
    h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
    h_new, w_new = h_old + direction[0], w_old + direction[1]

    # Upon reaching the end
    if (h_new == coord[-1][0] and w_new == coord[-1][1]):
        print("Agent Reached Goal")
        agentDone = True

    # out of bounds
    if (h_new >= h or w_new >= w) or (h_new < 0 or w_new < 0):
        agent_reward += rewards_dict('1')
        agentDone = True

    else:
        # If you hit an obstacle (orange or black)
        if (inst_arr[h_new,w_new][0] == 255 and inst_arr[h_new,w_new][1] == 165 and inst_arr[h_new,w_new][2] == 0) or \
            (inst_arr[h_new,w_new][0] == 0 and inst_arr[h_new,w_new][1] == 0 and inst_arr[h_new,w_new][2] == 0):

            agent_reward += rewards_dict('1')
            agentDone = True

        # If the global navigation point (white) is not encountered
        if (global_map[h_new, w_new] == 255) and (0<=h_new<h and 0<=w_new<w):
            agent_reward += rewards_dict('0')

            # TODO 
            # The statistics here are wrong. 
            # They are not counted by the number of empty walks but by the skipped global navigation points.

            cells_skipped += 1
        
        # If you hit the global navigation point (gray)
        if (global_map[h_new, w_new] != 255 and cells_skipped >= 0) and (0<=h_new<h and 0<=w_new<w):
            agent_reward += rewards_dict('2',cells_skipped)
            cells_skipped = 0

    # Cross-border return 
    # TODO: Problem here. 
    # If you encounter an obstacle, you cannot move.

    if 0 > h_new or h_new>=h or 0>w_new or w_new>= w:
        h_new, w_new = h_old, w_old

    # Calculate new distance
    if manhattan_distance(h_new, w_new, coord[-1][0], coord[-1][1]) < dist:
        # agent_reward += rewards_dict('3')
        dist = manhattan_distance(h_new, w_new, coord[-1][0], coord[-1][1])
    
    # Update diagram
    inst_arr[h_old, w_old] = [255,255,255]
    inst_arr[h_new, w_new] = [255,0,0]

    # if idx == agent:
    local_obs = inst_arr[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]
    global_map[h_old, w_old] = 255
    local_map = global_map[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]

    # TODO Dynamic obstacles are not updated
    
        # else:
        #     isEnd = False
        #     if time_idx < len(coord):
        #         h_old, w_old = coord[time_idx-1]
        #         h_new, w_new = coord[time_idx]
            
        #     else:
        #         h_old, w_old = coord[-1]
        #         h_new, w_new = coord[-1]
        #         isEnd = True

        #     if not isEnd:
        #         inst_arr[h_new, w_new] = [255,165,0]
        #         inst_arr[h_old, w_old] = [255,255,255]
    
    return np.array(local_obs), np.array(local_map), global_map, agentDone, agent_reward, cells_skipped, inst_arr, [h_new, w_new], dist


def rewards_dict(case, N = 0):

    """
    Return reward value
    r1 indicates that the robot reaches the free point of non-global navigation
    r2 means the robot hit an obstacle
    r3 indicates that the robot reaches the global navigation point
    r4 available for additional reward option
    """
    r1,r2,r3,r4 = -0.01, -0.1, 0.1, 0.05
    rewards = {
        '0':r1,
        '1':r1 + r2,
        '2': r1 + N*r3,
        '3': r4
    }

    return rewards[case]

    
    
    

    

        




