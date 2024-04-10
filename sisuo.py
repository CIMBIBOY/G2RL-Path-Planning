import numpy as np
import random
import matplotlib.pyplot as plt

# 创建一个大小为60x60的数组，用于表示地图，并使用RGB通道
map_size = 30
map_array = np.ones((map_size, map_size, 3), dtype=np.uint8) * 255

obstacle_ranges = []

for i in range(3, 25):
    for j in range (3, 25):
        obstacle_ranges.append((i,j))

for i, j in obstacle_ranges:
    map_array[i, j, :] = [0, 0, 0]

for i in range(15, 25):
    map_array[i, 15, :] = [255, 255, 255]

map_array[15, 15, :] = [255, 0, 0]
map_array[0, 15, :] = [0, 0, 255]

# 使用 matplotlib 绘制地图
plt.figure(figsize=(map_size/10, map_size/10))  # 设置图像大小为60x60像素
plt.imshow(map_array, origin='lower')
plt.axis('off')  # 关闭坐标轴显示
plt.savefig('死锁.png', bbox_inches='tight', pad_inches=0)  # 将地图保存为 PNG 文件，不留白
plt.show()