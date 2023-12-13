from PIL import Image
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from network import MLP, CNN


def read_image(image_path):
    # Load the image
    image = Image.open(image_path)
    return np.array(image)

def set_start_end(image, start_pos, end_pos):
    # Set the start and end positions
    image[start_pos[0], start_pos[1], :] = np.array([255, 0, 0])  # Red
    image[end_pos[0], end_pos[1], :] = np.array([0, 0, 255])  # Blue
    return image


if __name__ == '__main__':
    # Load the image
    image = read_image('data/cleaned_empty/empty-48-48-random-10_60_agents.png')
    # 起点
    start_pos = (0, 0)
    # 终点
    end_pos = (len(image)-1, len(image[0])-1)
    # 设置起点和终点
    image = set_start_end(image, start_pos, end_pos)

    # Create the DQN model
    num_actions = 4  # Replace with the number of actions in your path planning problem
    state_shape = [14, 14, 3]
    model = MLP(state_shape, num_actions)