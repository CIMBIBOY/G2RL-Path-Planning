import numpy as np
from heapq import heappop, heappush


def a_star_search(image):
    # Define the start and goal colors
    start_color = np.array([255, 0, 0])  # Red
    goal_color = np.array([0, 0, 255])  # Blue

    # Find the start and goal positions
    start_pos = tuple(np.argwhere(np.all(image == start_color, axis=-1))[0])
    goal_pos = tuple(np.argwhere(np.all(image == goal_color, axis=-1))[0])

    # Define the possible movements
    movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Define the cost function
    def cost(current_pos, next_pos):
        return 1

    # Define the heuristic function (Manhattan distance)
    def heuristic(current_pos, goal_pos):
        return abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])

    # Initialize the data structures
    open_list = [(0, start_pos)]
    closed_set = set()
    # Dictionary to keep track of the path
    came_from = {}

    # Initialize the cost and path matrices
    cost_matrix = np.full_like(image[:, :, 0], -1, dtype=np.int32)
    path_matrix = np.full_like(image[:, :, 0], -1, dtype=np.int32)

    # Set the start position cost and path
    cost_matrix[start_pos[0], start_pos[1]] = 0
    path_matrix[start_pos[0], start_pos[1]] = 0

    # A* search algorithm
    while open_list:
        current_cost, current_pos = heappop(open_list)

        if current_pos[0] == goal_pos[0] and current_pos[1] == goal_pos[1]:
            break

        if current_pos in closed_set:
            continue

        closed_set.add(current_pos)

        for movement in movements:
            next_pos = (current_pos[0] + movement[0], current_pos[1] + movement[1])

            if not (0 <= next_pos[0] < image.shape[0] and 0 <= next_pos[1] < image.shape[1]):
                continue

            if np.array_equal(image[next_pos[0], next_pos[1]], np.array([0, 0, 0])):  # Obstacle
                continue

            new_cost = cost_matrix[current_pos[0], current_pos[1]] + cost(current_pos, next_pos)

            if cost_matrix[next_pos[0], next_pos[1]] == -1 or new_cost < cost_matrix[next_pos[0], next_pos[1]]:
                cost_matrix[next_pos[0], next_pos[1]] = new_cost
                priority = new_cost + heuristic(next_pos, goal_pos)
                heappush(open_list, (priority, next_pos))
                came_from[next_pos] = current_pos

    # Reconstruct the path
    current_pos = goal_pos
    path = []

    while current_pos[0] != start_pos[0] or current_pos[1] != start_pos[1]:
        path.append(current_pos)
        current_pos = came_from[current_pos]

    path.append(start_pos)
    path.reverse()

    # Modify the original image to mark the path
    path_image = np.full_like(image[:, :, 0], 255, dtype=np.int32)
    for pos in path:
        image[pos[0], pos[1]] = np.array([0, 255, 0])  # Green
        path_image[pos[0], pos[1]] = 105

    # Modify the path matrix
    for i, row in enumerate(path_matrix):
        for j, _ in enumerate(row):
            if (i, j) in path:
                path_matrix[i, j] = path.index((i, j))
            elif np.array_equal(image[i, j], np.array([0, 0, 0])):
                path_matrix[i, j] = -1
    return path, path_image, path_matrix