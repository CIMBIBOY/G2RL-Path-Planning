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

# Function to initialize dynamic obstacles on the map
def initialize_objects(arr, n_dynamic_obst = 20):
    """
    Input: array of initial map, number of dynamic obstacles

    Output: array of initial positions of all dynamic obstacles and images after adding dynamic obstacles

    """
    arr = arr.copy()
    coord = []
    h, w = arr.shape[:2]

    while n_dynamic_obst > 0:
        h_obs = random.randint(0, h - 1)
        w_obs = random.randint(0, w - 1)

        cell_coord = arr[h_obs, w_obs]
        if np.array_equal(cell_coord, [255, 255, 255]):
            arr[h_obs, w_obs] = [255, 165, 0]
            n_dynamic_obst -= 1
            coord.append([h_obs, w_obs])

    return coord, arr

# Function to calculate the Manhattan distance
def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

# Function to update the coordinates of the agent and dynamic obstacles
def update_coords(coords, inst_arr, agent, time_idx, width, global_map, direction, agent_old_coordinates, cells_skipped, dist, env):
    
    """ 
    Update coordinates

    Input: all paths, a map containing all information, agent id, time, local field of view size, global navigation map, 
        movement direction [x, y], coordinates at the last moment, number of grids skipped, distance

    Output: local field of view, local navigation map, global navigation map, whether to act, reward, 
        number of skipped grids, updated map, updated coordinates, distance

    """
    deb = 0

    h, w = inst_arr.shape[:2]

    local_obs = np.array([])
    local_map = np.array([])
    agent_reward = 0

    # Get the path of the agent
    agent_path = coords[agent]

    agentDone = False
    h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
    print(f"Old cordinates {h_old, w_old}")
    h_new, w_new = h_old + direction[1], w_old + direction[0]
    print(f"Old cordinates {h_new, h_old}")

    # Check if the agent has reached its goal
    if (h_new == agent_path[-1][0] and w_new == agent_path[-1][1]):
        print("Agent Reached Goal")
        agentDone = True
        inst_arr[h_old, w_old] = [0, 255, 0] # mark "before collision cell" as green

    # Check for out of bounds or collisions with obstacles
    if (h_new >= h or w_new >= w) or (h_new < 0 or w_new < 0) or \
       (inst_arr[h_new, w_new][0] == 255 and inst_arr[h_new, w_new][1] == 165 and inst_arr[h_new, w_new][2] == 0) or \
       (inst_arr[h_new, w_new][0] == 0 and inst_arr[h_new, w_new][1] == 0 and inst_arr[h_new, w_new][2] == 0):
        agent_reward += rewards_dict('1')
        agentDone = True
        h_new, w_new = h_old, w_old
    else:
        if global_map[h_new, w_new] == 255:
            agent_reward += rewards_dict('0')
            cells_skipped += 1
        
        if global_map[h_new, w_new] != 255 and cells_skipped >= 0:
            agent_reward += rewards_dict('2', cells_skipped)
            cells_skipped = 0

    # Calculate new distance
    new_dist = manhattan_distance(h_new, w_new, agent_path[-1][0], agent_path[-1][1])
    if new_dist < dist:
        print(f"Old dist: {dist}, new dist: {new_dist}")
        dist = new_dist

    env.render_forvideo(0,deb) # env image debugger
    deb = deb + 1

    # Update dynamic obstacles
    for idx, path in enumerate(coords):
        if idx == agent:
            continue  # Skip the agent, we'll update it separately

        if time_idx < len(path):
            h_old_obs, w_old_obs = path[time_idx - 1]
            h_new_obs, w_new_obs = path[time_idx]
        else:
            continue  # Skip obstacles that have reached their goal

        # Check if the next position is occupied or out of bounds
        is_occupied = (h_new_obs >= h or w_new_obs >= w or h_new_obs < 0 or w_new_obs < 0 or
                       not np.array_equal(inst_arr[h_new_obs, w_new_obs], [255, 255, 255]))

        if is_occupied:
            if random.random() < 0.9:
                # Stay in current position
                h_new_obs, w_new_obs = h_old_obs, w_old_obs
            else:
                # Reverse direction and move back to start
                coords[idx] = path[:time_idx][::-1] + [[h_old_obs, w_old_obs]]
                h_new_obs, w_new_obs = coords[idx][1]  # Move to the next position in the reversed path
        
        # Update the obstacle's position
        print(f"Dyn Obs: {idx} was at pos: {h_old_obs, w_old_obs} and now at {h_new_obs, w_new_obs}" )
        env.render_forvideo(0,deb) # env image debugger
        deb = deb + 1
        # if np.array_equal(inst_arr[h_old_obs, w_old_obs], [255, 165, 0]):  # only move if it's a yellow dynamic object
        inst_arr[h_new_obs, w_new_obs] = [255, 165, 0]  # Move to new position
        inst_arr[h_old_obs, w_old_obs] = [255, 255, 255]  # Clear old position
        coords[idx] = path[:time_idx] + [[h_new_obs, w_new_obs]] + path[time_idx+1:]
        env.render_forvideo(0,deb) # env image debugger
        deb = deb + 1

    # Update agent position after moving obstacles
    if not agentDone:
        # Clear the previous agent position only if it's red
        inst_arr[h_old, w_old] = [255, 255, 255]
        inst_arr[h_new, w_new] = [255, 0, 0]

    env.render_forvideo(0,deb) # env image debugger
    deb = deb + 1
    # Update the agent's path in coords
    coords[agent] = coords[agent][:time_idx] + [[h_new, w_new]] + coords[agent][time_idx+1:]
    env.render_forvideo(0,deb) # env image debugger
    deb = deb + 1

    print(f"Agent position: old = {(h_old, w_old)}, new = {(h_new, w_new)}")
    moving_obstacles = sum(1 for idx, path in enumerate(coords) if idx != agent and time_idx < len(path))
    print(f"Number of moving obstacles: {moving_obstacles}")

    # Update local observation and global map
    local_obs = inst_arr[max(0, h_new - width):min(h - 1, h_new + width), max(0, w_new - width):min(w - 1, w_new + width)]
    global_map[h_old, w_old] = 255
    local_map = global_map[max(0, h_new - width):min(h - 1, h_new + width), max(0, w_new - width):min(w - 1, w_new + width)]

    # Check for collision with dynamic obstacles
    for idx, path in enumerate(coords):
        if idx == agent:
            continue
        if [h_new, w_new] in path:
            print("Collision with dynamic obstacle")
            agent_reward += rewards_dict('1')
            inst_arr[h_old, w_old] = [0, 255, 0] # mark collision cell as green
    
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
