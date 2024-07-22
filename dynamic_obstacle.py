from collections import defaultdict
import os
from PIL import Image
from numpy import array, asarray
import numpy as np
import random
from global_mapper import a_star
from map_generator import heuristic_generator

'''
1.	Initialize dynamic objects on the map.
2.	Move dynamic obstacles based on the A* algorithm or other logic.
3.	Update coordinates of obstacles during simulation steps.
'''

def initialize_objects(arr, n_dynamic_obst = 20):
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
    h, w = inst_arr.shape[:2]
    
    local_obs = np.array([])
    local_map = np.array([])
    agent_reward = 0

    # Get the path of the agent
    agent_path = coords[agent]

    agentDone = False
    h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
    h_new, w_new = h_old + direction[1], w_old + direction[0]

    # Upon reaching the end
    if (h_new == agent_path[-1][0] and w_new == agent_path[-1][1]):
        print("Agent Reached Goal")
        agentDone = True

    # Check for out of bounds or obstacles
    if (h_new >= h or w_new >= w) or (h_new < 0 or w_new < 0) or \
       (inst_arr[h_new,w_new][0] == 255 and inst_arr[h_new,w_new][1] == 165 and inst_arr[h_new,w_new][2] == 0) or \
       (inst_arr[h_new,w_new][0] == 0 and inst_arr[h_new,w_new][1] == 0 and inst_arr[h_new,w_new][2] == 0):
        agent_reward += rewards_dict('1')
        agentDone = True
        h_new, w_new = h_old, w_old
    else:
        # If the global navigation point (white) is not encountered
        if (global_map[h_new, w_new] == 255):
            agent_reward += rewards_dict('0')
            cells_skipped += 1
        
        # If you hit the global navigation point (gray)
        if (global_map[h_new, w_new] != 255 and cells_skipped >= 0):
            agent_reward += rewards_dict('2', cells_skipped)
            cells_skipped = 0

    # Calculate new distance
    new_dist = manhattan_distance(h_new, w_new, agent_path[-1][0], agent_path[-1][1])
    if new_dist < dist:
        dist = new_dist

    # Clear the previous agent position
    inst_arr[h_old, w_old] = [255, 255, 255]

    # Update dynamic obstacles
    for idx, path in enumerate(coords):
        if idx == agent:
            continue  # Skip the agent, we'll update it separately
        
        if time_idx < len(path):
            h_old, w_old = path[time_idx - 1]
            h_new, w_new = path[time_idx]
        else:
            h_old, w_old = path[-1]
            h_new, w_new = path[-1]

        if h_new >= h or w_new >= w:
            continue

        cell_coord = inst_arr[h_new, w_new]

        if np.array_equal(cell_coord, [255, 255, 255]):  # If the cell is white (free)
            inst_arr[h_new, w_new] = [255, 165, 0]  # Move to new position
            inst_arr[h_old, w_old] = [255, 255, 255]  # Clear old position
        else:
            # If the next cell is occupied, stay in place
            h_new, w_new = h_old, w_old

        coords[idx] = coords[idx][:time_idx] + [[h_new, w_new]] + coords[idx][time_idx+1:]

    # Clear the previous agent position only if it's red
    if np.array_equal(inst_arr[h_old, w_old], [255, 0, 0]):
        inst_arr[h_old, w_old] = [255, 255, 255]

    # Update agent position after moving obstacles
    if not agentDone:
        inst_arr[h_new, w_new] = [255, 0, 0]

    # Update the agent's path in coords
    coords[agent] = coords[agent][:time_idx] + [[h_new, w_new]] + coords[agent][time_idx+1:]

    print(f"Agent position: old = {(h_old, w_old)}, new = {(h_new, w_new)}")
    moving_obstacles = sum(1 for idx, path in enumerate(coords) if idx != agent and time_idx < len(path))
    print(f"Number of moving obstacles: {moving_obstacles}")

    # Update local observation and global map
    local_obs = inst_arr[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]
    global_map[h_old, w_old] = 255
    local_map = global_map[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]

    return np.array(local_obs), np.array(local_map), global_map, agentDone, agent_reward, cells_skipped, inst_arr, [h_new, w_new], dist, coords


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
        '0': r1,
        '1': r1 + r2,
        '2': r1 + N * r3,
        '3': r4
    }

    return rewards[case]
