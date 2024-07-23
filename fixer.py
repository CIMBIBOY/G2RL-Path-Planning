yet again I start off with 2 red marks on the map marking agents. They are always beside each other at the first reset of the env: from PIL import Image
import numpy as np
from dynamic_obstacle import initialize_objects, update_coords
from map_generator import start_end_points, map_to_value, global_guidance
from global_mapper import find_path, return_path
from utils import symmetric_pad_array
import os
import imageio

def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

class WarehouseEnvironment:

    def __init__(self,height = 48, width = 48, amr_count = 20, agent_idx = 1, local_fov = 15):

        assert height == 48 and width == 48, "We are not currently supporting other dimensions"
        # Initial map address
        self.map_path = "data/cleaned_empty/empty-48-48-random-10_60_agents.png" 
        self.amr_count = amr_count
        # Convert png image to array, three layers of RGB
        self.map_img_arr = np.asarray(Image.open(self.map_path))
        # state space dimension
        self.n_states = height * width
        # action space dim
        self.n_actions = len(self.action_space())
        # Agent id
        self.agent_idx = agent_idx
        # Partial field of view size
        self.local_fov = local_fov
        self.time_idx = 1
        self.init_arr = []
        # Array for dynamic objects
        self.dynamic_coords = []
        self.frames = []  # To store frames for .gif visualization
    
    def reset(self):
        # Initialize all dynamic obstacles
        self.dynamic_coords, self.init_arr = initialize_objects(self.map_img_arr, self.amr_count)
        
        # Generate destinations and routes
        print(f"Number of dynamic obstacles after initialization: {len(self.dynamic_coords)}")
        self.generate_end_points_and_paths()
        print(f"Number of dynamic obstacles after generating paths: {len(self.dynamic_coords)}")
        
        # The dynamic obstacle corresponding to agent_idx is regarded as the controlled agent
        self.agent_prev_coord = self.dynamic_coords[self.agent_idx][0]  # Take the first position of the path
        
        # The agent is modified to red
        self.init_arr[self.agent_prev_coord[0], self.agent_prev_coord[1]] = [255, 0, 0]  # Mark the agent's initial position in red
        
        self.time_idx = 1
        self.scenes = []
        self.cells_skipped = 0
        
        # initialization state
        reset_state = self.dynamic_coords[self.agent_idx]
        
        # initial distance
        start = reset_state[0]
        end = reset_state[-1]
        self.dist = manhattan_distance(start[0], start[1], end[0], end[1])
        
        graphical_state, _, _, _ = self.step(4)
        return start[0] * start[1], graphical_state
    
    def generate_end_points_and_paths(self):
        """
        Generate destinations and routes
        """
        value_map = map_to_value(self.init_arr.squeeze())
        start_end_coords = start_end_points(self.dynamic_coords, value_map)

        self.agents_paths = []
        for idx, idx_coords in start_end_coords:
            start = idx_coords[:2]
            end = idx_coords[2:]
            
            if isinstance(start, (list, tuple)) and len(start) == 2 and isinstance(end, (list, tuple)) and len(end) == 2:
                path, fov = find_path(value_map, start, end)
                if path:  # Check if a valid path was found
                    short_path = return_path(path)
                    self.agents_paths.append(short_path)
                else:
                    print(f"No valid path found for obstacle {idx}. Keeping it stationary.")
                    self.agents_paths.append([start])  # Keep the obstacle at its start position
            else:
                raise ValueError("start and end must be lists or tuples of length 2")
        
        # Debug
        valid_paths = sum(1 for path in self.agents_paths if len(path) > 1)
        print(f"Number of valid paths: {valid_paths}")
        
        self.dynamic_coords = self.agents_paths
        self.global_mapper_arr = global_guidance(self.agents_paths[self.agent_idx], self.map_img_arr.squeeze())


    def step(self, action):
        if len(self.init_arr) == 0:
            print("Run env.reset() first")
            return

        self.time_idx += 1
        conv, x, y = self.action_dict[action]
        # print(f'Action taken: {conv}')
        
        target_array = (2*self.local_fov, 2*self.local_fov, 4)

        print(f"Number of dynamic obstacles before update: {len(self.dynamic_coords)}")
        # Update coordinates - last (if working) is: , self.dynamic_coords
        local_obs, local_map, self.global_mapper_arr, isAgentDone, rewards, \
        self.cells_skipped, self.init_arr, new_agent_coord, self.dist, self.dynamic_coords = \
        update_coords(
            self.dynamic_coords, self.init_arr, self.agent_idx, self.time_idx,
            self.local_fov, self.global_mapper_arr, [x,y], self.dynamic_coords[self.agent_idx][self.time_idx-1],
            self.cells_skipped, self.dist
        )

        self.agent_prev_coord = new_agent_coord

        print(f"Number of dynamic obstacles after update: {len(self.dynamic_coords)}")
        print(f"Agent color at position {self.agent_prev_coord}: {self.init_arr[self.agent_prev_coord[0], self.agent_prev_coord[1]]}")

        combined_arr = np.array([])
        if len(local_obs) > 0:
            self.scenes.append(Image.fromarray(local_obs, 'RGB'))
            local_map = local_map.reshape(local_map.shape[0],local_map.shape[1],1)
            combined_arr = np.dstack((local_obs, local_map))
            combined_arr = symmetric_pad_array(combined_arr, target_array, 255)
            combined_arr = combined_arr.reshape(1,1,combined_arr.shape[0], combined_arr.shape[1], combined_arr.shape[2])

        return combined_arr, self.agent_prev_coord[0] * self.agent_prev_coord[1], rewards, isAgentDone
    
    def render_forvideo(self, train_index, image_index):
        assert len(self.init_arr) != 0, "Run env.reset() before proceeding"
        img = Image.fromarray(self.init_arr, 'RGB')

        # Ensure the base directory 'training_images' exists
        base_dir = '/Users/czimbermark/Documents/SZTAKI/G2RL/G2RL-Path-Planning/training_images'
        os.makedirs(base_dir, exist_ok=True)

        # Create train_{train_index}_images directory if it does not exist
        train_dir = os.path.join(base_dir, f"train_{train_index}_images")
        os.makedirs(train_dir, exist_ok=True)

        # Save the image with a unique filename
        img_path = os.path.join(train_dir, f"train_{train_index}_{int(image_index)}.png")
        img.save(img_path)

    def render(self):
        """
        Renders the current state of the environment in gif format for real-time visualization. 
        This method should be called after each step.
        """
        assert len(self.init_arr) != 0, "Run env.reset() before proceeding"
        
        # Convert the environment state to an image
        img = Image.fromarray(self.init_arr.astype('uint8'), 'RGB')
        
        # Resize the image if needed (optional, for better visualization)
        img = img.resize((200, 200), Image.NEAREST)
        
        # Convert PIL Image to numpy array
        frame = np.array(img)
        
        # Append the frame to our list of frames
        self.frames.append(frame)
        
        # Update the GIF file
        self._update_gif()

    def _update_gif(self):
        """
        Updates the GIF file with all frames collected so far.
        """
        # Save the frames as a GIF
        imageio.mimsave("data/g2rl.gif", self.frames, duration=0.5, loop=0)

    def create_scenes(self, path = "data/agent_locals.gif", length_s = 100):
        if len(self.scenes) > 0:
            self.scenes[0].save(path,
                 save_all=True, append_images=self.scenes[1:], optimize=False, duration=length_s*4, loop=0)
        else:
            pass

    def action_space(self):
        # action space
        self.action_dict = {
            0:['up',0,1],
            1:['down',0,-1],
            2:['left',-1,0],
            3:['right',1,0],
            4:['idle',0,0]
        }
        return list(self.action_dict.keys())


env = WarehouseEnvironment()
_, state = env.reset() # image of first reset

print(state.shape)

env.render_forvideo(0,1) - after training starts only some dynamic object starts to move from the 19 object even tho debug states all of them are moving: (base) Hero:G2RL-Path-Planning czimbermark$ /Users/czimbermark/.pyenv/versions/3.10.0/bin/python /Users/czimbermark/Documents/SZTAKI/G2RL/G2RL-Path-Planning/WarehouseEnv.py
Number of dynamic obstacles after initialization: 20
Number of valid paths: 20
Number of dynamic obstacles after generating paths: 20
Number of dynamic obstacles before update: 20
Agent position: old = (37, 3), new = (37, 3)
Number of moving obstacles: 19
Number of dynamic obstacles after update: 20
Agent color at position [37, 3]: [255   0   0]
(1, 1, 30, 30, 4). If I look correctly at my code dyn obstacles should mostly move cause they only stay stationary when random.random() is > 0.9.. Also environment resets random when there is nothing near the agent, which is a very nasty bug. So I currently have these 3 issues: 2 agents adjesent to each other at the first step and only 1 moves. So wrong init a red dot stays on the map. Movement of dynamic obstacles is wrong, only some moves. And environment resets random, when it shouldn't: from collections import defaultdict
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

    # Check if the agent has reached its goal
    if (h_new == agent_path[-1][0] and w_new == agent_path[-1][1]):
        print("Agent Reached Goal")
        agentDone = True
        inst_arr[h_new, w_new] = [0, 255, 0] # mark succes as green

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
        dist = new_dist

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
            if random.random() > 0.9:
                # Stay in current position
                h_new_obs, w_new_obs = h_old_obs, w_old_obs
            else:
                # Reverse direction and move back to start
                coords[idx] = path[:time_idx][::-1] + [[h_old_obs, w_old_obs]]
                h_new_obs, w_new_obs = coords[idx][1]  # Move to the next position in the reversed path

        # Update the obstacle's position
        if np.array_equal(inst_arr[h_old_obs, w_old_obs], [255, 165, 0]):  # only move if it's a yellow dynamic object
            inst_arr[h_new_obs, w_new_obs] = [255, 165, 0]  # Move to new position
            inst_arr[h_old_obs, w_old_obs] = [255, 255, 255]  # Clear old position
            coords[idx] = path[:time_idx] + [[h_new_obs, w_new_obs]] + path[time_idx+1:]

    # Update agent position after moving obstacles
    if not agentDone:
        # Clear the previous agent position only if it's red
        inst_arr[h_old, w_old] = [255, 255, 255]
        inst_arr[h_new, w_new] = [255, 0, 0]

    # Update the agent's path in coords
    coords[agent] = coords[agent][:time_idx] + [[h_new, w_new]] + coords[agent][time_idx+1:]

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
            print("Collision with dynamic obstacle, resetting environment.")
            return np.array(local_obs), np.array(local_map), global_map, True, agent_reward, cells_skipped, inst_arr, [h_new, w_new], dist, coords

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
