import numpy as np
import random

'''
1.	Initialize dynamic objects on the map.
2.	Move dynamic obstacles based on the A* algorithm or other logic.
3.	Update coordinates of obstacles during simulation steps.
'''

# Function to initialize dynamic obstacles on the map
def initialize_objects(arr, n_dynamic_obst = 20, rng=None):
    """
    Input: array of initial map, number of dynamic obstacles

    Output: array of initial positions of all dynamic obstacles and images after adding dynamic obstacles

    """
    arr = arr.copy()
    coord = []
    h, w = arr.shape[:2]

    if rng is None:
        rng = np.random  # Use the global numpy RNG if none is provided

    while n_dynamic_obst > 0:
        h_obs = rng.integers(0, h)
        w_obs = rng.integers(0, w)

        cell_coord = arr[h_obs, w_obs]
        if cell_coord[0] != 0 and cell_coord[1] != 0 and cell_coord[2] != 0:
            arr[h_obs, w_obs] = [255, 165, 0]
            n_dynamic_obst -= 1
            coord.append([h_obs, w_obs])
    
    return coord, arr

# Function to calculate the Manhattan distance
def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

# Function to update the coordinates of the agent and dynamic obstacles
def update_coords(coords, inst_arr, agent, time_idx, width, global_map, direction, agent_old_coordinates, cells_skipped, dist, agent_goal, terminations, stayed_array, info):
    
    """ 
    Update coordinates

    Input: all paths, a map containing all information, agent id, time, local field of view size, global navigation map, 
        movement direction [x, y], coordinates at the last moment, number of grids skipped, distance

    Output: local field of view, local navigation map, global navigation map, whether to act, reward, 
        number of skipped grids, updated map, updated coordinates, distance

    """
    h, w = inst_arr.shape[:2]

    local_obs = np.array([])
    local_map = np.array([])
    agent_reward = 0
    arrived = 0

    # Get the path of the agent
    agent_path = coords[agent]

    done = False
    trunc = False
    h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
    h_new, w_new = h_old + direction[0], w_old - direction[1]

    # debug to monitor the agent's movement
    # print(f"At time index: {time_idx}\nAgent position: ({agent_old_coordinates[0]}, {agent_old_coordinates[1]})")
    # print(f"New agent position: ({h_new}, {w_new})\n")

    # Marking agent's goal cell as green 
    # inst_arr[agent_goal[0], agent_goal[1]] = [0, 255, 0]

    # Check if the agent has reached its goal
    if (h_new, w_new) == (agent_goal[0], agent_goal[1]):
        print(f"")
        print(f"From start position: {agent_path[0]}, agent reached it's goal at: {agent_goal} in {time_idx} timesteps")
        done = True
        info['goal_reached'] = True
        inst_arr[h_new, w_new] = [0, 255, 0]  # mark goal cell as green
        arrived = True
        agent_reward += rewards_dict('3', manhattan_distance(agent_path[0][0], agent_path[0][1], agent_path[-1][0], agent_path[-1][1]))
        terminations[0] += 1

    # Check for out of bounds or collisions with obstacles
    if (h_new >= h or w_new >= w) or (h_new < 0 or w_new < 0) or \
       (inst_arr[h_new, w_new][0] == 255 and inst_arr[h_new, w_new][1] == 165 and inst_arr[h_new, w_new][2] == 0) or \
       (inst_arr[h_new, w_new][0] == 0 and inst_arr[h_new, w_new][1] == 0 and inst_arr[h_new, w_new][2] == 0):
        agent_reward += rewards_dict('1')
        # print("Reward for collision")
        trunc = True
        info['collision'] = True
        # print("Collision with obstacles or out of bounds")
        terminations[3] += 1
        h_new, w_new = h_old, w_old
    else:
        if direction[0] == 0 and direction[1] == 0: 
            agent_reward += rewards_dict('0')
            # print("Reward for non global navigation")
        else:
            if global_map[h_new, w_new] == 255:
                agent_reward += rewards_dict('0')
                # print("Reward for non global navigation")
                cells_skipped += 1

            if global_map[h_new, w_new] != 255 and cells_skipped == 0:
                agent_reward += rewards_dict('4')
                # print("Reward for staying on the global path")
            
            if global_map[h_new, w_new] != 255 and cells_skipped > 0:
                agent_reward += rewards_dict('2', cells_skipped)
                # print("Reward for retruning to global path")
                cells_skipped = 0

    # Calculate new distance
    new_dist = manhattan_distance(h_new, w_new, agent_path[-1][0], agent_path[-1][1])
    if new_dist < dist:
        dist = new_dist

    # At reset state objects don't move
    if time_idx != 1: 
        # Update dynamic obstacles
        for idx, path in enumerate(coords):
            if idx == agent:
                continue  # Skip the agent, we'll update it separately
            if time_idx < len(path) + 1:
                h_old_obs, w_old_obs = path[time_idx-2-stayed_array[idx]]
                h_new_obs, w_new_obs = path[time_idx-1-stayed_array[idx]]
            else:
                continue  # Skip obstacles that have reached their goal

            # Check if the next position is occupied or out of bounds
            is_occupied = (h_new_obs >= h or w_new_obs >= w or h_new_obs < 0 or w_new_obs < 0 or
                        not np.array_equal(inst_arr[h_new_obs, w_new_obs], [255, 255, 255]))

            if is_occupied:
                if random.random() < 0.9:
                    stayed_array[idx] += 1
                    # Stay in current position
                    h_new_obs, w_new_obs = h_old_obs, w_old_obs
                else:
                    # Reverse direction and move back to start
                    coords[idx] = path[:time_idx-stayed_array[idx]][::-1]
                    stayed_array[idx] = time_idx - 2
                    h_new_obs, w_new_obs = coords[idx][1]  # Move to the next position in the reversed path
            
            # Update the obstacle's position
            # print(f"Dyn Obs: {idx} was at pos: {h_old_obs, w_old_obs} and now at {h_new_obs, w_new_obs}" )
            if h_old_obs != h_new_obs or w_old_obs != w_new_obs:
                inst_arr[h_old_obs, w_old_obs] = [255, 255, 255]  # Clear old position
                inst_arr[h_new_obs, w_new_obs] = [255, 165, 0]  # Move to new position
                # coords[idx] = path[:time_idx-1] + [[h_new_obs, w_new_obs]] + path[time_idx:]
            # elif not reversed:
                # If the obstacle didn't move, don't change its path
                # coords[idx] = path    
        
    # Update agent position after moving obstacles
    if not done:
        # Clear the previous agent position
        inst_arr[h_old, w_old] = [255, 255, 255]
        inst_arr[h_new, w_new] = [255, 0, 0]

    # Update local observation and global map
    local_obs = inst_arr[max(0, h_new - width):min(h - 1, h_new + width), max(0, w_new - width):min(w - 1, w_new + width)]
    global_map[h_old, w_old] = 255
    local_map = global_map[max(0, h_new - width):min(h - 1, h_new + width), max(0, w_new - width):min(w - 1, w_new + width)]

    return np.array(local_obs), np.array(local_map), global_map, done, trunc, info, agent_reward, cells_skipped, inst_arr, [h_new, w_new], dist, arrived, terminations, stayed_array


def rewards_dict(case, N = 0, path_len = 0):

    """
    Return reward value
    r1 indicates that the robot reaches the free point of non-global navigation
    r2 means the robot hit an obstacle
    r3 indicates that the robot reaches the global navigation point
    r4 agent reaches it's goal
    r5 agent follows it's global guidance path
    """
    r1,r2,r3,r4,r5= -0.01, -0.1, 0.1, 0.1 * path_len/10, 0.16
    rewards = {
        '0': r1,
        '1': r1 + r2,
        '2': r1 + N * r3,
        '3': r4,
        '4': r5
    }

    return rewards[case]
