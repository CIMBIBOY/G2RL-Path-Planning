from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
from A_star import a_star_search
import random


class StaticEnvironment:

    def __init__(self, agent_idx = 0, local_fov = 10, num_dyna = 0):
        # Initial map address
        self.map_path = "G2RL-Path-Planning/data/cleaned_empty/empty-48-48-random-10_60_agents.png"
        # Convert png image to array, three layers of RGB
        self.map_origin = np.array(Image.open(self.map_path))
        # state space dimension
        self.dim_states = [2 * local_fov + 1, 2 * local_fov + 1, 4]
        self.n_states = self.dim_states[0] * self.dim_states[1] * self.dim_states[2]
        # action space dimension
        self.n_actions = len(self.action_space())
        # Agent id
        self.agent_idx = agent_idx
        # Partial field of view size
        self.local_fov = local_fov
        # time step
        self.time_idx = 0
        # Global navigation steps
        self.guidance_idx = 0
        # Agent location
        self.agent_pos = {}
        # Visualization
        self.is_render = False
        # Number of dynamic obstacles
        self.num_dyna = num_dyna
        self.dyna_obst = {}

    def gen_c_flags(self):
        """
        Generate start and end points   
        """

        # Fixed generation of start and end coordinates
        start_pos = [0, 0]
        end_pos = [len(self.map_global)-1, len(self.map_global[0])-1]
        self.map_global[start_pos[0], start_pos[1], :] = np.array([255, 0, 0])  # Red
        self.map_global[end_pos[0], end_pos[1], :] = np.array([0, 0, 255])  # Blue
        self.map_agent[start_pos[0], start_pos[1], :] = np.array([255, 0, 0])  # Red 
        self.agent_pos[self.agent_idx] = start_pos
        self.end_pos = end_pos
        
        return start_pos

    def generate_path(self):
        _, self.map_path, self.path_matrix = a_star_search(self.map_global)

    def reset(self):
        self.map_global = copy.deepcopy(self.map_origin)
        self.map_agent = copy.deepcopy(self.map_origin)
        self.gen_c_flags()
        self.generate_path()
        self.generate_dyna_obst()
        self.guidance_idx = 0
        self.time_idx = 0
        return self.get_state()

    def is_free(self, pos):
        # Can't go beyond the boundary
        if pos[0] < 0 or pos[0] > len(self.map_origin) - 1 or \
            pos[1] < 0 or pos[1] > len(self.map_origin[0]) - 1:
            return False
        # Static obstacles cannot be walked
        if self.map_origin[pos[0], pos[1], 0] == 0 and \
            self.map_origin[pos[0], pos[1], 1] == 0 and \
            self.map_origin[pos[0], pos[1], 2] == 0:
            return False
        # Dynamic obstacles cannot be walked
        if self.map_agent[pos[0], pos[1], 0] == 255 and \
            self.map_agent[pos[0], pos[1], 1] == 165 and \
            self.map_agent[pos[0], pos[1], 2] == 0:
            return False
        # For dynamic obstacles, the agent cannot walk
        if self.map_agent[pos[0], pos[1], 0] == 255 and \
            self.map_agent[pos[0], pos[1], 1] == 0 and \
            self.map_agent[pos[0], pos[1], 2] == 0:
            return False
        return True
    
    def generate_dyna_obst(self):
        # Generate dynamic obstacles
        n_dynamic_obst = self.num_dyna
        h, w = self.map_origin.shape[:2]
        while n_dynamic_obst > 0:
            h_obs = random.randint(0,h-1)
            w_obs = random.randint(0,w-1)
            if self.is_free([h_obs, w_obs]):
                self.map_agent[h_obs, w_obs, :] = np.array([255,165,0])
                n_dynamic_obst -= 1
                self.dyna_obst[self.num_dyna-n_dynamic_obst] = [h_obs, w_obs]
    
    def dyna_obst_update(self):
        for idx in self.dyna_obst:
            pos = self.dyna_obst[idx]
            action_num = random.randint(0, self.n_actions-1)
            action = self.action_dict[action_num]
            pos_next = [pos[0] + action[1], pos[1] + action[2]]
            if self.is_free(pos_next):
                self.dyna_obst[idx] = pos_next
                self.map_agent[pos[0], pos[1], :] = np.array([255,255,255])
                self.map_agent[pos_next[0], pos_next[1], :] = np.array([255,165,0])
            else:
                pass

    def step(self, action_num):
        # Action selection
        action = self.action_dict[action_num]
        # Perform action 
        agent_pos = self.agent_pos[self.agent_idx]
        agent_pos_next = [agent_pos[0] + action[1], agent_pos[1] + action[2]]
        if self.is_free(agent_pos_next):
            self.agent_pos[self.agent_idx] = agent_pos_next
            self.map_agent[agent_pos[0], agent_pos[1], :] = np.array([255,255,255])
            self.map_agent[agent_pos_next[0], agent_pos_next[1], :] = np.array([255,0,0])
            is_collided = False
        else:
            is_collided = True
        # dynamic obstacle course
        self.dyna_obst_update()
        
        reward = self.get_reward(is_collided)
        state = self.get_state()
        terminated = self.is_terminated()
        self.state = state
        if self.is_render:
            self.render()
        self.time_idx += 1
        return reward, state, terminated

    def get_reward(self, is_collided):
        if is_collided:
            return self.rewards_dict('1')
        elif self.path_matrix[self.agent_pos[self.agent_idx][0], self.agent_pos[self.agent_idx][1]] != -1:
            N = self.path_matrix[self.agent_pos[self.agent_idx][0], self.agent_pos[self.agent_idx][1]] - self.guidance_idx
            if N > 0:
                self.guidance_idx = self.path_matrix[self.agent_pos[self.agent_idx][0], self.agent_pos[self.agent_idx][1]]
                return self.rewards_dict('2', N)
        return self.rewards_dict('0')

    def get_state(self):
        # With the agent as the center, intercept the local field of view
        agent_pos = self.agent_pos[self.agent_idx]
        local_fov = self.local_fov
        
        # Calculate the coordinates of the upper left corner and lower right corner of the local field of view
        top = max(0, agent_pos[0] - local_fov)
        bottom = min(len(self.map_origin) - 1, agent_pos[0] + local_fov)
        left = max(0, agent_pos[1] - local_fov)
        right = min(len(self.map_origin[0]) - 1, agent_pos[1] + local_fov)
        
        # Create a blank image of a partial field of view
        local_map = np.zeros(self.dim_states, dtype=np.uint8)

        # Copy the corresponding area in the global map to the local view
        local_map_start_row = local_fov - (agent_pos[0] - top)
        local_map_start_col = local_fov - (agent_pos[1] - left)
        local_map_end_row = local_fov + (bottom - agent_pos[0])
        local_map_end_col = local_fov + (right - agent_pos[1])
        local_map[local_map_start_row:local_map_end_row+1, local_map_start_col:local_map_end_col+1, 0:3] = \
            self.map_agent[top:bottom+1, left:right+1,:]
        
        # Copy path information into local view
        map_path_now = copy.deepcopy(self.map_path)
        map_path_now[self.path_matrix <= self.guidance_idx] = 0
        local_map[local_map_start_row:local_map_end_row+1, local_map_start_col:local_map_end_col+1, 3] = \
            map_path_now[top:bottom+1, left:right+1]
        
        # Parts that are not filled enough are black
        if local_map_start_row > 0:
            local_map[:local_map_start_row,:, 0:3] = 0
        if local_map_end_row < local_fov - 1:
            local_map[local_map_end_row+1:,:,0:3] = 0
        if local_map_start_col > 0:
            local_map[:, :local_map_start_col,0:3] = 0
        if local_map_end_col < local_fov - 1:
            local_map[:, local_map_end_col+1:, 0:3] = 0

        # The center of the local field of view is red
        local_map[local_fov, local_fov, 0:4] = np.array([255, 0, 0, 255])
        
        return local_map

    def is_terminated(self):
        if self.agent_pos[self.agent_idx] == self.end_pos:
            return True
        else:
            return False

    def render(self):
        if not hasattr(self, 'fig'):
            # Create a figure and axis
            self.fig, self.ax = plt.subplots()

        # Clear the previous plot
        self.ax.clear()

        # Plot the map_agent
        self.ax.imshow(self.map_agent)

        # Show the plot
        plt.pause(0.001)

        if not hasattr(self, 'fig_1'):
            # Create a figure and axis
            self.fig_1, self.ax_1 = plt.subplots()

        # Clear the previous plot
        self.ax_1.clear()

        # Plot the grayscale map
        self.ax_1.imshow(self.state[:,:,3], cmap='gray')

        # Show the plot
        plt.pause(0.001)

        if not hasattr(self, 'fig_2'):
            # Create a figure and axis
            self.fig_2, self.ax_2 = plt.subplots()

        # Clear the previous plot
        self.ax_2.clear()

        # Plot the grayscale map
        self.ax_2.imshow(self.state[:,:,:3])

        # Show the plot
        plt.pause(0.001)

    def action_space(self):
        # Action space
        self.action_dict = {
            0:['up',-1,0],
            1:['down',1,0],
            2:['left',0,-1],
            3:['right',0,1],
            4:['idle',0,0]
        }
        return list(self.action_dict.keys())
    
    def rewards_dict(self, case, N = 1):
        """
        Return reward value
        r1 indicates that the robot reaches the free point of non-global navigation
        r2 means the robot hit an obstacle
        r3 indicates that the robot reaches the global navigation point
        r4 for additional reward option
        """
        r1,r2,r3,r4 = -1, -100, 10, 5
        rewards = {
            '0': r1,
            '1': r1 + r2,
            '2': r1 + N*r3,
            '3': r4
        }
        return rewards[case]

if __name__=="__main__":
    env = StaticEnvironment()
    env.reset()
    for _ in range(100):
        env.step(3)
        time.sleep(0.3)