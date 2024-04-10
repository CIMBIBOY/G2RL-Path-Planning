import numpy as np
import random
import matplotlib.pyplot as plt

# 创建一个大小为60x60的数组，用于表示地图，并使用RGB通道
map_size = 60
map_array = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255

# 在地图上随机设置一些障碍物
# 定义障碍物或特征点的范围
obstacle_ranges = []
# obstacle_ranges = [(i, j) for i in list(range(0, 6)) + list(range(10, 16)) + list(range(20, 26)) + list(range(30, 36)) + list(range(40, 46)) + list(range(50, 56))
#                     for j in list(range(0, 6)) + list(range(10, 16)) + list(range(20, 26)) + list(range(30, 36)) + list(range(40, 46)) + list(range(50, 56))]
for i in range(0, 8):
    for j in range(0, 8):
          obstacle_ranges.append((i,j))
for i in range(12, 22):
    for j in range(20, 28):
          obstacle_ranges.append((i,j))
for i in range(32, 50):
    for j in range(19, 38):
          obstacle_ranges.append((i,j))
for i in range(40, 49):
    for j in range(11, 18):
          obstacle_ranges.append((i,j))
for i in range(10, 28):
    for j in range(0, 10):
          obstacle_ranges.append((i,j))
for i in range(0, 10):
    for j in range(50, 60):
          obstacle_ranges.append((i,j))
for i in range(50, 60):
    for j in range(50, 60):
          obstacle_ranges.append((i,j))
for i in range(40, 45):
    for j in range(50, 60):
          obstacle_ranges.append((i,j))          
      

for i, j in obstacle_ranges:
        map_array[i, j, :] = [0, 0, 0]  # 设置障碍物的颜色为黑色

pos = [26, 20]
map_array[pos[0], pos[1], :] = [255, 0, 0]
map_array[47,59,:] = [0, 255, 255]

global_guide = []
for i in range(21, 50):
    global_guide.append([26,i])
for i in range(26,48):
    global_guide.append([i,49])
for i in range(50,59):
    global_guide.append([47,i]) 

for i, j in global_guide:
    map_array[i, j, :] = [105, 105, 105]

obs_dyna = [(12, 11),(31, 24), (6, 21), (11, 47), (58, 2), (19, 17), (21, 18), (46, 8), (14, 42), (25, 45), (22, 45), (43, 1), (19, 10), (22, 34), (25, 18), (42, 0), (25, 14), (10, 48), (53, 25), (49, 40), (52, 25)]
# for i, j in obs_dyna:
#     map_array[i, j, :] = [255, 165, 0]

local_map = map_array[pos[0]-14:pos[0]+14, pos[1]-14:pos[1]+14, :]

# 使用 matplotlib 绘制地图
plt.figure(figsize=(map_size/10, map_size/10))  # 设置图像大小为60x60像素
plt.imshow(map_array, origin='lower')
plt.axis('off')  # 关闭坐标轴显示
plt.savefig('grid_map_rgb.png', bbox_inches='tight', pad_inches=0)  # 将地图保存为 PNG 文件，不留白
plt.show()