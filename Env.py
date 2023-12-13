from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
from algorithm import a_star_search
import utils


class StaticEnvironment:

    def __init__(self, height = 48, width = 48, amr_count = 20, agent_idx = 0, local_fov = 4):
        assert height == 48 and width == 48, "We are not currently supporting other dimensions"
        # 初始地图地址
        self.map_path = "data/cleaned_empty/empty-48-48-random-10_60_agents.png"
        self.amr_count = amr_count
        # 把png图片转为array，三层RGB
        self.map_origin = np.array(Image.open(self.map_path))
        # 状态空间维数
        self.n_states = height*width
        # 动作空间维数
        self.n_actions = len(self.action_space())
        # 智能体id
        self.agent_idx = agent_idx
        # 局部视野范围大小
        self.local_fov = local_fov
        # 时间步
        self.time_idx = 1
        # 全局导航步数
        self.guidance_idx = 0
        # 智能体位置
        self.agent_pos = {}

    def generate_start_and_end_points(self):
        """生成起点和终点    
        """

        # 固定生成起点和终点坐标
        start_pos = [0, 0]
        end_pos = [len(self.map_global)-1, len(self.map_global[0])-1]
        self.map_global[start_pos[0], start_pos[1], :] = np.array([255, 0, 0])  # Red
        self.map_global[end_pos[0], end_pos[1], :] = np.array([0, 0, 255])  # Blue
        self.map_agent[start_pos[0], start_pos[1], :] = np.array([255,165,0])  # Orange
        self.agent_pos[self.agent_idx] = start_pos
        self.end_pos = end_pos
        
        return start_pos

    def generate_path(self):
        _, self.map_path, self.path_matrix = a_star_search(self.map_global)

    def reset(self):
        self.map_global = copy.deepcopy(self.map_origin)
        self.map_agent = copy.deepcopy(self.map_origin)
        self.generate_start_and_end_points()
        self.generate_path()

    def is_free(self, pos):
        # 越界不能走
        if pos[0] < 0 or pos[0] > len(self.map_origin) - 1 or \
            pos[1] < 0 or pos[1] > len(self.map_origin[0]) - 1:
            return False
        # 静态障碍物不能走
        if self.map_origin[pos[0], pos[1], 0] == 0 and \
            self.map_origin[pos[0], pos[1], 1] == 0 and \
            self.map_origin[pos[0], pos[1], 2] == 0:
            return False
        return True
        

    def step(self, action):
        # 选择动作
        action = self.action_dict[action]
        # 执行动作
        agent_pos = self.agent_pos[self.agent_idx]
        agent_pos_next = [agent_pos[0] + action[1], agent_pos[1] + action[2]]
        if self.is_free(agent_pos_next):
            self.agent_pos[self.agent_idx] = agent_pos_next
            self.map_agent[agent_pos[0], agent_pos[1], :] = np.array([255,255,255])
            self.map_agent[agent_pos_next[0], agent_pos_next[1], :] = np.array([255,165,0])
            is_collided = False
        else:
            is_collided = True
        
        reward = self.get_reward(is_collided)
        state = self.get_state()
        terminated = self.is_terminated()
        self.state = state
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
        # 以agent为中心，截取局部视野
        agent_pos = self.agent_pos[self.agent_idx]
        local_fov = self.local_fov
        
        # 计算局部视野的左上角和右下角坐标
        top = max(0, agent_pos[0] - local_fov)
        bottom = min(len(self.map_origin) - 1, agent_pos[0] + local_fov)
        left = max(0, agent_pos[1] - local_fov)
        right = min(len(self.map_origin[0]) - 1, agent_pos[1] + local_fov)
        
        # 创建局部视野的空白图像
        local_map = np.zeros((local_fov*2+1, local_fov*2+1, 3), dtype=np.uint8)
        
        # 将全局地图中对应的区域复制到局部视野中
        local_map_start_row = local_fov - (agent_pos[0] - top)
        local_map_start_col = local_fov - (agent_pos[1] - left)
        local_map_end_row = local_fov + (bottom - agent_pos[0])
        local_map_end_col = local_fov + (right - agent_pos[1])
        local_map[local_map_start_row:local_map_end_row+1, local_map_start_col:local_map_end_col+1] = \
            self.map_origin[top:bottom+1, left:right+1]
        
        # 填充不够的部分为黑色
        if local_map_start_row > 0:
            local_map[:local_map_start_row,:, :] = 0
        if local_map_end_row < local_fov - 1:
            local_map[local_map_end_row+1:,:,:] = 0
        if local_map_start_col > 0:
            local_map[:, :local_map_start_col,:] = 0
        if local_map_end_col < local_fov - 1:
            local_map[:, local_map_end_col+1:, :] = 0
        
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
        # self.ax.imshow(self.state)

        # Show the plot
        plt.pause(0.001)

    def action_space(self):
        # 动作空间
        self.action_dict = {
            0:['up',-1,0],
            1:['down',1,0],
            2:['left',0,-1],
            3:['right',0,1],
            4:['idle',0,0]
        }
        return list(self.action_dict.keys())
    
    def rewards_dict(self, case, N = 1):
        """返回奖励值
        r1表示机器人到达非全局导航的自由点
        r2表示机器人撞在障碍物上
        r3表示机器人到达全局导航点
        r4?
        """
        r1,r2,r3,r4 = -0.01, -0.1, 0.1, 0.05
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
        env.step(4)
        time.sleep(0.3)