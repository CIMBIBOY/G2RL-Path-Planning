from PIL import Image
import numpy as np
from dynamic_obstacle import initialize_objects, update_coords
from map_generator import start_end_points, map_to_value, global_guidance
from global_mapper import find_path, return_path
from utils import symmetric_pad_array
import os
import imageio
import pygame
from collections import deque
import torch
import time

def manhattan_distance(x_st, y_st, x_end, y_end):
    return abs(x_end - x_st) + abs(y_end - y_st)

class WarehouseEnvironment:

    def __init__(self,height = 48, width = 48, amr_count = 20, agent_idx = 1, local_fov = 15, pygame_render = True):

        assert height == 48 and width == 48, "We are not currently supporting other dimensions"
        # Initial map address
        self.map_path = "data/cleaned_empty/empty-48-48-random-10_60_agents.png" 
        self.amr_count = amr_count
        # Convert png image to array, three layers of RGB
        self.map_img_arr = np.asarray(Image.open(self.map_path))
        # map size
        self.height = height
        self.width = width
        # state space dimension
        self.n_states = height * width
        # action space dim
        self.n_actions = len(self.action_space())
        # Number of historical observations to use
        self.Nt = 4 
        self.initial_random_steps = 0
        # Buffer to store past observations to store last 4 observations
        self.observation_history = deque(maxlen=self.Nt)
        # Agent id
        self.agent_idx = agent_idx
        # Agent's path length 
        self.agent_path_len = 100    
        self.steps = 0 
        # Partial field of view size
        self.local_fov = local_fov
        self.time_idx = 1
        self.init_arr = []
        # Array for dynamic objects
        self.dynamic_coords = []
        self.episode_count = 0

        # Agent reached end position count 
        self.arrived = 0
        self.episode_count = 0

        self.frames = []  # To store frames for .gif visualization

        self.pygame_render = pygame_render
        self.screen = None
        self.clock = None

    
    def reset(self):
        # reset step count for maximum timesteps
        self.steps = 0
        # Initialize all dynamic obstacles
        self.dynamic_coords, self.init_arr = initialize_objects(self.map_img_arr, self.amr_count)
        
        if self.init_arr is None or self.init_arr.size == 0:
            raise ValueError("Initialization failed, init_arr is empty or None")

        # Generate destinations and routes
        self.generate_end_points_and_paths()
       
        # The dynamic obstacle corresponding to agent_idx is regarded as the controlled agent
        self.agent_prev_coord = self.dynamic_coords[self.agent_idx][0]  # Take the first position of the path
        # print(self.agent_prev_coord)

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

        self.agent_path = self.agents_paths[self.agent_idx]

        self.observation_history.clear()
        self.initial_random_steps = 0

        # Take initial step to get the first real observation
        graphical_state, _, _, _ = self.step(4)  # Assuming 4 is a valid initial action

        # Increment the episode count
        self.episode_count += 1
        
        # Re-randomize the start and goal cells of all dynamic obstacles after 50 episodes
        if self.episode_count % 50 == 0:
            self.dynamic_coords, self.init_arr = initialize_objects(self.map_img_arr, self.amr_count)
            self.generate_end_points_and_paths()
            self.agent_prev_coord = self.dynamic_coords[self.agent_idx][0]
            self.init_arr[self.agent_prev_coord[0], self.agent_prev_coord[1]] = [255, 0, 0]
            self.agent_path = self.agents_paths[self.agent_idx]

        return start[0] * start[1], graphical_state
    

    def get_stacked_state(self):
        # Ensure we have exactly Nt observations
        assert len(self.observation_history) == self.Nt, f"Expected {self.Nt} observations, but got {len(self.observation_history)}"
        
        # Stack the observations along the second axis (axis=1)
        stacked_state = np.concatenate(list(self.observation_history), axis=1)

        return stacked_state
    

    def step(self, action):
        self.steps += 1
        if len(self.init_arr) == 0:
            print("Run env.reset() first")
            return None, None, None, False

        self.time_idx += 1
        conv, x, y = self.action_dict[action]
        # print(f'Action taken: {conv}')
        
        target_array = (2*self.local_fov, 2*self.local_fov, 4)

        agent_goal = self.agent_path[-1]  # Get the goal from the agent's path
        self.agent_path_len = len(self.agent_path)
        
        # Update coordinates 
        local_obs, local_map, self.global_mapper_arr, isAgentDone, rewards, \
        self.cells_skipped, self.init_arr, new_agent_coord, self.dist, self.dynamic_coords, reached_goal = \
        update_coords(
            self.dynamic_coords, self.init_arr, self.agent_idx, self.time_idx,
            self.local_fov, self.global_mapper_arr, [x,y], self.agent_prev_coord,
            self.cells_skipped, self.dist, agent_goal
        )

        self.agent_prev_coord = new_agent_coord
        if reached_goal == True:
            self.arrived += 1

        # Check if there's global guidance in the local FOV
        if not self.has_global_guidance():
            isAgentDone = True 

        # maximum allowed steps for a single epoch
        if self.steps > self.agent_path_len * 4:
            isAgentDone = True 

        combined_arr = np.array([])
        if len(local_obs) > 0:
            self.scenes.append(Image.fromarray(local_obs, 'RGB'))
            local_map = local_map.reshape(local_map.shape[0],local_map.shape[1],1)
            combined_arr = np.dstack((local_obs, local_map))
            combined_arr = symmetric_pad_array(combined_arr, target_array, 255)
            combined_arr = combined_arr.reshape(1,1,combined_arr.shape[0], combined_arr.shape[1], combined_arr.shape[2])
        
        if len(combined_arr) > 0:
            if len(self.observation_history) < self.Nt:
                self.observation_history.append(combined_arr)
                self.initial_random_steps += 1
            else:
                # Remove the oldest observation and add the new one
                self.observation_history.popleft()
                self.observation_history.append(combined_arr)

        if self.initial_random_steps < self.Nt:
            # Return the single observation during initial steps
            return_values = (combined_arr, self.agent_prev_coord[0] * self.agent_prev_coord[1], rewards, isAgentDone)
        else:
            # Return the stacked state after we have enough observations
            stacked_state = self.get_stacked_state()
            return_values = (stacked_state, self.agent_prev_coord[0] * self.agent_prev_coord[1], rewards, isAgentDone)
        
        return return_values
    
    def get_action_mask(self, device):
        """Return a mask of valid actions, where 1 indicates a valid action and 0 indicates an invalid action."""
        mask = np.ones(len(self.action_dict), dtype=np.float32)

        # Get the current position of the agent
        agent_position = self.agent_prev_coord
        h, w = agent_position

        # Check each possible action and set mask to 0 for invalid actions
        if h <= 0 or not self.is_position_valid(h, w - 1):  # up
            mask[0] = 0
            print(f"Invalid action: Up (h={h}, w={w - 1})")
        if h >= self.height - 1 or not self.is_position_valid(h, w + 1):  # down
            mask[1] = 0
            print(f"Invalid action: Down (h={h}, w={w + 1})")
        if w <= 0 or not self.is_position_valid(h - 1, w):  # left
            mask[2] = 0
            print(f"Invalid action: Left (h={h - 1}, w={w})")
        if w >= self.width - 1 or not self.is_position_valid(h + 1, w):  # right
            mask[3] = 0
            print(f"Invalid action: Right (h={h + 1}, w={w})")

        # Idle action is always valid
        mask[4] = 1

        print(f"Action mask: {mask}")
        return torch.tensor(mask, device=device)

    def is_position_valid(self, h, w):
        # Check if the position is within the map boundaries
        if h < 0 or h >= self.height or w < 0 or w >= self.width:
            return False
        # Collision with:
        if (self.init_arr[h, w] == [255, 165, 0]).all():  # Dynamic obstacle
            return False
        if (self.init_arr[h, w] == [0, 0, 0]).all():  # Static obstacle
            return False
        return True
    
    
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
            assert start != end, "Kb lehetetlen, de mégis megtörténik..."
            
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
        half_fov = self.local_fov // 2
        
        local_guidance = self.global_mapper_arr[
            max(0, self.agent_prev_coord[0] - half_fov):min(self.width, self.agent_prev_coord[0] + half_fov + 1),
            max(0, self.agent_prev_coord[1] - half_fov):min(self.width, self.agent_prev_coord[1] + half_fov + 1)
        ]
        
        # Check if there's any global guidance information (value less than 255) in the local observation
        has_guidance = np.any(local_guidance < 255)
        
        return has_guidance
    

    def render_video(self, train_index, image_index):
        assert len(self.init_arr) != 0, "Run env.reset() before proceeding"
        # Get the most recent observation (last channel of the stacked state)
        img = Image.fromarray(self.init_arr, 'RGB')

        # Ensure the base directory 'training_images' exists
        base_dir = 'training_images'
        os.makedirs(base_dir, exist_ok=True)

        # Create train_{train_index}_images directory if it does not exist
        train_dir = os.path.join(base_dir, f"train_{train_index}_images")
        os.makedirs(train_dir, exist_ok=True)

        # Save the image with a unique filename
        img_path = os.path.join(train_dir, f"train_{train_index}_{int(image_index)}.png")
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

    def render(self):
        if self.pygame_render:  # Check if rendering is enabled
            if self.screen is None:  # Initialize only if not already initialized
                pygame.init()
                print("Pygame screen constructed")
                self.screen = pygame.display.set_mode((200, 200))
                self.clock = pygame.time.Clock()

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
        
        # Blit the scaled surface to the screen
        self.screen.blit(surf, (0, 0))

        # Update the display
        pygame.display.flip()

        # Control the frame rate
        self.clock.tick(20)  # 30 FPS

    def close(self):
        pygame.quit()

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

# Testing Environment

'''
env = WarehouseEnvironment()
_, state = env.reset() # image of first reset

print(state.shape)

env.render_video(0,420)
'''