from collections import defaultdict
import os
from PIL import Image
from numpy import array, asarray
import numpy as np
import random

def initialize_objects(arr, n_dynamic_obst = 10):
    """
    输入：初始地图的array，动态障碍物数量
    输出：所有动态障碍物初始位置和加入动态障碍后图的array
    """
    arr = arr.copy()
    coord = []
    h,w = arr.shape[:2]

    while n_dynamic_obst > 0:
        h_obs = random.randint(0,h-1)
        w_obs = random.randint(0,w-1)

        cell_coord = arr[h_obs, w_obs]
        # RGB为0,0,0时为黑色，表示静态障碍，不能生成
        if cell_coord[0] != 0 and cell_coord[1] != 0 and cell_coord[2] != 0:
            # 动态障碍物为橙色
            arr[h_obs, w_obs] = [255,165,0]
            n_dynamic_obst -= 1
            coord.append([h_obs,w_obs])
    
    return coord, arr

def manhattan_distance(x_st, y_st, x_end, y_end):
    # 返回曼哈顿距离
    return abs(x_end - x_st) + abs(y_end - y_st)

def update_coords(coords, inst_arr, agent, time_idx, width, global_map, direction, agent_old_coordinates, cells_skipped, dist):
    """更新坐标
    输入：全部路径，包含全部信息的图，智能体id，时间，局部视野大小，全局导航图，运动方向[x,y]，上一时刻坐标，跳过的格子数，距离
    返回：局部视野，局部导航图，全局导航图，是否行动，奖励，跳过的格子数，更新后的图，更新后的坐标，距离
    """
    h,w = inst_arr.shape[:2]
    
    local_obs = np.array([])
    local_map = np.array([])
    agent_reward = 0
    # 取出智能体的路径
    coord = coords[agent]

    agentDone = False
    h_old, w_old = agent_old_coordinates[0], agent_old_coordinates[1]
    h_new, w_new = h_old + direction[0], w_old + direction[1]

    # 如果到达终点
    if (h_new == coord[-1][0] and w_new == coord[-1][1]):
        print("Agent Reached Gole")
        agentDone = True

    # 如果越界
    if (h_new >= h or w_new >= w) or (h_new < 0 or w_new < 0):
        agent_reward += rewards_dict('1')
        agentDone = True

    else:
        # 如果碰到障碍物（橙色或黑色）
        if (inst_arr[h_new,w_new][0] == 255 and inst_arr[h_new,w_new][1] == 165 and inst_arr[h_new,w_new][2] == 0) or \
            (inst_arr[h_new,w_new][0] == 0 and inst_arr[h_new,w_new][1] == 0 and inst_arr[h_new,w_new][2] == 0):

            agent_reward += rewards_dict('1')
            agentDone = True

        # 如果没有碰到全局导航点（白色）
        if (global_map[h_new, w_new] == 255) and (0<=h_new<h and 0<=w_new<w):
            agent_reward += rewards_dict('0')
            # TODO 这里的统计不对，不是按空走的次数算而是靠跳过的全局导航点算
            cells_skipped += 1
        
        # 如果碰到全局导航点（灰色）
        if (global_map[h_new, w_new] != 255 and cells_skipped >= 0) and (0<=h_new<h and 0<=w_new<w):
            agent_reward += rewards_dict('2',cells_skipped)
            cells_skipped = 0

    # 越界回归 TODO：这里有问题，如果碰到障碍物也不能走
    if 0 > h_new or h_new>=h or 0>w_new or w_new>= w:
        h_new, w_new = h_old, w_old

    # 计算新的距离
    if manhattan_distance(h_new, w_new, coord[-1][0], coord[-1][1]) < dist:
        # agent_reward += rewards_dict('3')
        dist = manhattan_distance(h_new, w_new, coord[-1][0], coord[-1][1])
    
    # 更新图
    inst_arr[h_old, w_old] = [255,255,255]
    inst_arr[h_new, w_new] = [255,0,0]

    # if idx == agent:
    local_obs = inst_arr[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]
    global_map[h_old, w_old] = 255
    local_map = global_map[max(0,h_new - width):min(h-1,h_new + width), max(0,w_new - width):min(w-1,w_new + width)]

    # TODO 没有更新动态障碍物
    
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
    """返回奖励值
    r1表示机器人到达非全局导航的自由点
    r2表示机器人撞在障碍物上
    r3表示机器人到达全局导航点
    r4?
    """
    r1,r2,r3,r4 = -0.01, -0.1, 0.1, 0.05
    rewards = {
        '0':r1,
        '1':r1 + r2,
        '2': r1 + N*r3,
        '3': r4
    }

    return rewards[case]

    
    
    

    

        




