from PIL import Image
import numpy as np
from environment.dynamic_obstacle import initialize_objects, update_coords
from environment.map_generator import start_end_points, map_to_value, global_guidance
from environment.global_mapper import find_path, return_path
from helpers.utils import symmetric_pad_array, calculate_max_steps
import os
import imageio
import pygame
from collections import deque
import torch
from gym.spaces import Box, Discrete
from gym.utils import seeding
import random

def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

class WarehouseEnvironment:

    def __init__(self,height = 48, width = 48, amr_count = 2, max_amr = 25, agent_idx = 0, local_fov = 15, time_dimension = 1, pygame_render = True, seed = None):

        assert height == 48 and width == 48, "We are not currently supporting other dimensions"
        # Initial map address
        self.map_path = "data/cleaned_empty/empty-48-48-random-10_60_agents.png" 
        # Dynamic objects start and max number
        self.max_amr = max_amr
        self.amr_count = amr_count
        self.fast_obj = 4
        self.fast_obj_idx = self.max_amr - self.fast_obj
        # Convert png image to array, three layers of RGB
        self.map_img_arr = np.asarray(Image.open(self.map_path))
        # map size
        self.height = height
        self.width = width
        # state space dimension
        self.n_states = height * width
        # Number of historical observations to use
        self.Nt = time_dimension
        # Buffer to store past observations to store last 4 observations
        self.observation_history = deque(maxlen=self.Nt)
        self.initial_random_steps = False
        # observation space
        self.observation_space = Box(low=0, high=255, shape=(1, self.Nt, 30, 30, 4), dtype=np.uint8)
        # action space dim
        self.n_actions = len(self.f_action_space())
        # Define action space
        self.action_space = Discrete(self.n_actions)
        # Agent id
        self.agent_idx = agent_idx
        # Agent's path length 
        self.agent_path_len = 100
        self.agent_goal = None 
        self.steps = 0 
        # Partial field of view size
        self.local_fov = local_fov
        self.time_idx = 0
        self.init_arr = []
        # Array for dynamic objects
        self.dynamic_coords = []
        self.stays = []
        self.terminations = np.zeros(4, dtype=int)
        self.last_action = 4

        # Agent reached end position count 
        self.arrived = 0
        self.episode_count = -1
        self.horizon = 'short'
        self.max_step = 42
       
        self.frames = []  # To store frames for .gif visualization

        self.metadata = {
            'render.modes': ['human'],  # Specifies that 'human' render mode is supported
            'video.frames_per_second': 30  # Frame rate for rendering videos, adjust as needed
        }
        self.pygame_render = pygame_render
        self.screen = None
        self.clock = None

        self.info = {
            'R_max_step': False,
            'no_global_guidance': False,
            'goal_reached': False,
            'collision': False,
            'blocked': False,
            'steps': 0,
            # 'path': [],
            'reward': 0,
        }

        self.seed(seed)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)  # Generate a valid seed if none is provided
        elif not (0 <= seed < 2**32):
            raise ValueError("Seed must be between 0 and 2**32 - 1")
        # Ensure seed is an integer
        seed = int(seed)
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]

    def reset(self): 
        # Increment the episode count
        self.episode_count += 1
        
        # Reset step count for maximum timesteps
        self.steps = 0
        self.info['steps'] = 0
        
        # Generate new coordinates and paths every 50 episodes
        if self.episode_count == 0 or self.episode_count % 50 == 1:

            self.seed(self.np_random.integers(0, 1000000))
            # Implementing curriculum learning
            if self.arrived >= 25 and self.amr_count < self.max_amr: 
                self.amr_count += 1
                print(f"Dynamic object added, current count: {self.amr_count}")
                if self.amr_count == self.fast_obj_idx:
                    print(f"First fast dynamic object added, task got harder!")

            # Initialize all dynamic obstacles
            self.dynamic_coords, self.init_arr = initialize_objects(self.map_img_arr, self.amr_count, rng=self.np_random)

            if self.init_arr is None or self.init_arr.size == 0:
                raise ValueError("Initialization failed, init_arr is empty or None")

            # Generate destinations and routes
            self.generate_end_points_and_paths()
            # Set dones for new start-goal pair
            self.arrived = 0
        else:
            # If not generating new paths, reset the init_arr to its original state
            self.init_arr = self.map_img_arr.copy() 
            # Reset dynamic obstacles to their initial positions
            for idx, path in enumerate(self.dynamic_coords):
                initial_pos = path[0]
                if idx < self.fast_obj_idx: 
                    self.init_arr[initial_pos[0], initial_pos[1]] = [255, 165, 0]  # Orange color for dynamic obstacles
            self.global_mapper_arr = global_guidance(self.agents_paths[self.agent_idx], self.map_img_arr.squeeze())
        
        self.stays = np.zeros(self.amr_count, dtype=int)
        # The dynamic obstacle corresponding to agent_idx is regarded as the controlled agent
        self.agent_prev_coord = self.dynamic_coords[self.agent_idx][0]  # Take the first position of the path
        # The agent is modified to red
        self.init_arr[self.agent_prev_coord[0], self.agent_prev_coord[1]] = [255, 0, 0]  # Mark the agent's initial position in red

        # Implementation of fasterer dynamic obstacles after curriculum learning reaches self.max_amr green fast objects are added.
        if self.amr_count > self.fast_obj_idx:
            for idx, path in enumerate(self.dynamic_coords[self.fast_obj_idx:]):
                # Randomly decide whether to keep every second or every third coordinate
                if self.np_random.random() < 0.8:  
                    # 90% chance to remove every second or third coordinate
                    if self.np_random.random() < 0.5:
                        # 50% chance to remove every second coordinate
                        self.dynamic_coords[self.fast_obj_idx + idx] = path[::2]  # Keep only even-indexed positions
                    else: 
                        # 50% chance to remove every third coordinate
                        self.dynamic_coords[self.fast_obj_idx + idx] = [pos for i, pos in enumerate(path) if (i + 1) % 3 != 0]
                else: 
                    # 10% chance to keep every third coordinate
                    self.dynamic_coords[self.fast_obj_idx + idx] = path[::3]  # Keep every third position
                
                initial_pos = path[0]
                self.init_arr[initial_pos[0], initial_pos[1]] = [0, 255, 0]  # Green color for fast objects

        # TODO: Implement a blue agent, which follows A* path, choosing idle action for every fifth time_idx. 
        # Additional reward for agent if stays close or surpasses blue agent. + reward dict element
        # Collision sholdn't be allowed with blue agent, because it will stay on A* path, for which agent recives icreased reward. (Potentially high collision risk.)

        self.time_idx = 0
        self.scenes = []
        self.leave_idx = -1
        
        # initialization state
        reset_state = self.dynamic_coords[self.agent_idx]
        
        # calc maximum steps
        if self.horizon != 'long':
            self.max_step = calculate_max_steps(self.agent_path_len)

        # initial distance
        start = reset_state[0]
        end = reset_state[-1]
        self.dist = manhattan_distance(start[0], start[1], end[0], end[1])
        self.agent_path_len = self.dist

        self.agent_path = self.agents_paths[self.agent_idx]
        self.agent_goal = self.agent_path[-1]  # Get the goal from the agent's path

        self.observation_history.clear()
        self.initial_random_steps = False

        # Take initial step to get the first real observation
        graphical_state, _, _, _, _ = self.step(4)  # Assuming 4 is a valid initial action

        return graphical_state, self.info

    def step(self, action):
        if len(self.init_arr) == 0:
            print("Run env.reset() first")
            return None, None, None, False

        conv, x, y = self.action_dict[action]
        
        target_array = (2*self.local_fov, 2*self.local_fov, 4)
        
        self.time_idx += 1

        # Update coordinates 
        local_obs, local_map, self.global_mapper_arr, done, trunc, self.info, rewards, \
        self.leave_idx, self.init_arr, new_agent_coord, self.dist, reached_goal, self.terminations, self.stays = \
        update_coords(
            self.dynamic_coords, self.init_arr, self.agent_idx, self.time_idx,
            self.local_fov, self.global_mapper_arr, [x,y], self.agent_prev_coord,
            self.leave_idx, self.dist, self.agent_goal, self.terminations, self.stays, self.info, self.fast_obj_idx, self.agent_path_len
        )

        self.steps += 1
        self.agent_prev_coord = new_agent_coord

        # Update info
        self.info['steps'] += 1
        # self.info['path'].append((self.agent_prev_coord[0], self.agent_prev_coord[1]))
        self.info['reward'] += rewards

        if reached_goal == True:
            self.arrived += 1

        # Check if there's global guidance in the local FOV
        if not self.has_global_guidance() and done == False:
            trunc = True
            self.info['no_global_guidance'] = True
            self.terminations[1] += 1

        if self.steps > self.max_step and done == False:
            # print(f"Max steps reached with steps: {self.steps} for path length: {self.agent_path_len}, decay: {self.decay}")
            trunc = True
            self.info['R_max_step'] = True
            self.terminations[2] += 1

        if done or trunc: self.agent_last_coord = new_agent_coord

        combined_arr = np.array([])
        if len(local_obs) > 0:
            self.scenes.append(Image.fromarray(local_obs, 'RGB'))
            local_map = local_map.reshape(local_map.shape[0],local_map.shape[1],1)
            combined_arr = np.dstack((local_obs, local_map))
            combined_arr = symmetric_pad_array(combined_arr, target_array, 255)
            combined_arr = combined_arr.reshape(1,1,combined_arr.shape[0], combined_arr.shape[1], combined_arr.shape[2])
        
        if len(combined_arr) > 0:
            if len(self.observation_history) < self.Nt:
                for _ in range(self.Nt):
                    self.observation_history.append(combined_arr)
                self.initial_random_steps = True
            else:
                # Remove the oldest observation and add the new one
                self.observation_history.popleft()
                self.observation_history.append(combined_arr)
    
        
        if self.initial_random_steps == False:
            # Return the single observation during initial steps
            return_values = (combined_arr, rewards, done, trunc, self.info)
        else:
            # Return the stacked state after we have enough observations
            stacked_state = self.get_stacked_state()
            return_values = (stacked_state, rewards, done, trunc, self.info)
            
        return return_values
    
    def get_stacked_state(self):
        # Ensure we have exactly Nt observations
        assert len(self.observation_history) == self.Nt, f"Expected {self.Nt} observations, but got {len(self.observation_history)}"
        
        # Stack the observations along the second axis (axis=1)
        stacked_state = np.concatenate(list(self.observation_history), axis=1)

        return stacked_state
    
    def f_action_space(self):
        # action space
        self.action_dict = {
            0:['up',0,1],
            1:['down',0,-1],
            2:['left',-1,0],
            3:['right',1,0],
            4:['idle',0,0]
        }
        return list(self.action_dict.keys())
    
    def action_mask(self, device):
        return self.get_action_mask(device)
    
    def get_action_mask(self, device):
        """Return a mask of valid actions, where 1 indicates a valid action and 0 indicates an invalid action."""
        mask = torch.ones(len(self.action_dict), dtype=torch.float32, device=device)

        # Get the current position of the agent
        agent_position = self.agent_prev_coord
        h, w = agent_position

        # Check each possible action and set mask to 0 for invalid actions
        if w < 0 or not self.is_position_valid(h, w - 1):  # up
            mask[0] = 0
            # print(f"Invalid action: Up (h={h}, w={w - 1})")
        if w >= self.width or not self.is_position_valid(h, w + 1):  # down
            mask[1] = 0
            # print(f"Invalid action: Down (h={h}, w={w + 1})")
        if h < 0 or not self.is_position_valid(h - 1, w):  # left
            mask[2] = 0
            # print(f"Invalid action: Left (h={h - 1}, w={w})")
        if h >= self.height or not self.is_position_valid(h + 1, w):  # right
            mask[3] = 0
            # print(f"Invalid action: Right (h={h + 1}, w={w})")

        if self.last_action == 4:
            mask[4] = 0
        else: # Idle action is only valid if last 3 wasn't idle
            mask[4] = 1

        # print(f"Action mask: {mask}")
        return mask

    def is_position_valid(self, h, w):
    
        if h < 0 or h >= self.height or w < 0 or w >= self.width:
            # print(f"Position ({h}, {w}) is out of bounds")
            return False
        
        if (self.init_arr[h, w] == [0, 255, 0]).all():
            # print(f"Position ({h}, {w}) contains a dynamic obstacle")
            return False
        
        if (self.init_arr[h, w] == [255, 165, 0]).all():
            # print(f"Position ({h}, {w}) contains a dynamic obstacle")
            return False
        
        if (self.init_arr[h, w] == [0, 0, 0]).all():
            # print(f"Position ({h}, {w}) contains a static obstacle")
            return False
        
        # print(f"Position ({h}, {w}) is valid")
        return True
    
    def generate_end_points_and_paths(self):
        """
        Generate destinations and routes
        """
        value_map = map_to_value(self.init_arr.squeeze())
        start_end_coords = start_end_points(self.dynamic_coords, value_map, self.np_random)

        self.agents_paths = []
        for idx, idx_coords in start_end_coords:
            start = idx_coords[:2]
            end = idx_coords[2:]
            assert start != end, "Start and end coordinates cannot be indenticial"
            
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
        # valid_paths = sum(1 for path in self.agents_paths if len(path) > 1)
        # print(f"Number of valid paths: {valid_paths}")
        
        self.dynamic_coords = self.agents_paths
        self.global_mapper_arr = global_guidance(self.agents_paths[self.agent_idx], self.map_img_arr.squeeze())

    def has_global_guidance(self):
        local_guidance = self.global_mapper_arr[
            max(0, self.agent_prev_coord[0] - self.local_fov):min(self.width-1, self.agent_prev_coord[0] + self.local_fov),
            max(0, self.agent_prev_coord[1] - self.local_fov):min(self.width-1, self.agent_prev_coord[1] + self.local_fov)
        ]
        
        # Check if there's any global guidance information (value less than 255) in the local observation
        has_guidance = np.any(local_guidance < 255)
        
        return has_guidance
    
    def render(self):
        if self.pygame_render:  # Check if rendering is enabled
            if self.screen is None:  # Initialize only if not already initialized
                pygame.init()
                print("Pygame screen constructed")
                self.screen = pygame.display.set_mode((800, 800))
                self.clock = pygame.time.Clock()
                pygame.display.set_caption(f"Warehouse Environment")

        if self.pygame_render == False:
            pygame.quit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Create a surface from the numpy array
        surf = pygame.surfarray.make_surface(self.init_arr)

        # Define the window size
        new_width, new_height = 800, 800  # Changable

        # Scale the surface to the size
        surf = pygame.transform.scale(surf, (new_width, new_height))

        # If the screen size doesn't match the new size, recreate it
        if self.screen.get_size() != (new_width, new_height):
            self.screen = pygame.display.set_mode((new_width, new_height))

        # Draw the path under the agent
        for x, y in self.agent_path:
            center_x = (x + 0.5) * new_width // self.init_arr.shape[1]
            center_y = (y + 0.5) * new_height // self.init_arr.shape[0]
            pygame.draw.circle(surf, (255, 0, 0), (center_x, center_y), 5)

        # Draw a purple square representing the local field of view around the agent
        if hasattr(self, 'agent_prev_coord'):
            # Calculate the top-left corner of the FOV square
            top_left_x = max(0, self.agent_prev_coord[0] - self.local_fov) * new_width // self.init_arr.shape[1]
            top_left_y = max(0, self.agent_prev_coord[1] - self.local_fov) * new_height // self.init_arr.shape[0]

            # Calculate the bottom-right corner of the FOV square
            bottom_right_x = min(self.init_arr.shape[1]-1, self.agent_prev_coord[0] + self.local_fov) * new_width // self.init_arr.shape[1]
            bottom_right_y = min(self.init_arr.shape[0]-1, self.agent_prev_coord[1] + self.local_fov) * new_height // self.init_arr.shape[0]

            # Draw the purple square
            pygame.draw.rect(surf, (128, 0, 128), pygame.Rect(top_left_x, top_left_y, bottom_right_x - top_left_x, bottom_right_y - top_left_y), 2)  # The last parameter is the thickness

        # Blit the scaled surface to the screen
        self.screen.blit(surf, (0, 0))

        # Display the global guidance map as a semi-transparent red overlay
        if hasattr(self, 'global_mapper_arr'):
            # Convert the global map to a surface
            guidance_surf = pygame.surfarray.make_surface(self.global_mapper_arr)
            guidance_surf = pygame.transform.scale(guidance_surf, (new_width, new_height))
            guidance_surf.set_alpha(128)  # Set transparency level
            self.screen.blit(guidance_surf, (0, 0))

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        self.clock.tick(20)  # 30 FPS

    def close(self):
        pygame.quit()
    
    def render_video(self, train_name, image_index):
        assert len(self.init_arr) != 0, "Run env.reset() before proceeding"
        # Get the most recent observation (last channel of the stacked state)
        img = Image.fromarray(self.init_arr, 'RGB')

        # Ensure the base directory 'training_images' exists
        base_dir = 'eval/training_images'
        os.makedirs(base_dir, exist_ok=True)

        # Create train_{train_index}_images directory if it does not exist
        train_dir = os.path.join(base_dir, f"train_name")
        os.makedirs(train_dir, exist_ok=True)

        # Save the image with a unique filename
        img_path = os.path.join(train_dir, f"{train_name}_{int(image_index)}.png")
        img.save(img_path)

    def render_gif(self):
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

# Testing Environment

'''
env = WarehouseEnvironment()
_, state = env.reset() # image of first reset

print(state.shape)

env.render_video(0,420)
'''