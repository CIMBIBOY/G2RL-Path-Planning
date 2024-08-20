from environment.map_generator import heuristic_generator

'''
1.	Pathfinding algorithm implementations.
2.	Support functions for the A* algorithm, such as the heuristic generator and others for path reconstruction.
'''

class Node:
    def __init__(self, parent, position) -> None:
        self.parent = parent
        self.pos = position

    def __eq__(self, node):
        if node == None:
            return self.pos == node

        return self.pos == node.pos

    def __gt__(self, node):
        return self.pos > node.pos
    
    def __lt__(self, node):
        return self.pos < node.pos
    
def a_star(grid, init, goal, cost, delta, heuristic):
    """
    :param grid: 2D matrix of the world, with the obstacles marked as '1', rest as '0'
    :param init: list of x and y co-ordinates of the robot's initial position
    :param goal: list of x and y co-ordinates of the robot's intended final position
    :param cost: cost of moving one position in the grid
    :param delta: list of all the possible movements
    :param heuristic: 2D matrix of same size as grid, giving the cost of reaching the goal from each cell
    :return: path: list of the cost of the minimum path, and the goal position
    :return: extended: 2D matrix of same size as grid, for each element, the count when it was expanded or -1 if
             the element was never expanded.
    """

    # Debug print statements to check init and goal
    # print(f"init: {init}")
    # print(f"goal: {goal}")

    # Ensure that init and goal are in the form [x, y]
    if not (isinstance(init, (list, tuple)) and len(init) == 2):
        print(f"Error: init is not a list or tuple of length 2: {init}, type: {type(init)}")
        raise ValueError("init must be a list or tuple of length 2")
    if not (isinstance(goal, (list, tuple)) and len(goal) == 2):
        print(f"Error: goal is not a list or tuple of length 2: {goal}, type: {type(goal)}")
        raise ValueError("goal must be a list or tuple of length 2")
    
    # Edge case where path is blocked immediately
    if grid[init[0]][init[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        print(f"Start or goal is blocked. Start: {init}, Goal: {goal}")
        return 'fail', []
    
    # open list
    path = []
    routes = []
    val = 1
    # Closed set, representing the nodes that have been visited, initially 0 and 1 after the visit
    visited = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    visited[init[0]][init[1]] = 1
    # Path sequence diagram, initially -1, means no walking, the starting point is 1, and then increases sequentially
    expand = [[-1 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    expand[init[0]][init[1]] = 0
    # starting point
    init_node = Node(None, init)
    g = 0

    # f = g + h, h is the heuristic function, indicating the estimated cost (Manhattan distance)
        # from the current node to the end point
    f = g + heuristic[init_node.pos[0]][init_node.pos[1]]

    minList = [f, g, init_node]

    while [minList[2].pos[0], minList[2].pos[1]] != goal:
        # Traverse all possible directions of movement
        for i in range(len(delta)):
            # moved node
            point = Node(init_node, [init_node.pos[0] + delta[i][0], init_node.pos[1] + delta[i][1]])
            # Judgment has not crossed the line
            if 0 <= point.pos[0] < len(grid) and 0 <= point.pos[1] < len(grid[0]):
                # Judgment has not been visited
                if visited[point.pos[0]][point.pos[1]] == 0 and grid[point.pos[0]][point.pos[1]] == 0:
                    # Calculate new f, g
                    g2 = g + cost
                    f2 = g2 + heuristic[point.pos[0]][point.pos[1]]
                    path.append([f2, g2, point])
                    visited[point.pos[0]][point.pos[1]] = 1

        # If there are no expandable nodes, return failure
        if not path:
            print(f"A* failed: No expandable nodes")
            print(f"Current position: {init_node.pos}")

            # Printing map if no path is foundable 
            '''
            print(f"Visited cells:")
            for row in visited:
                print(row)
            '''
            return 'fail', expand

        # Clear minList
        del minList[:]
        # Find the node where f is the smallest
        minList = min(path)
        # Add the node to routes
        routes.append(minList)
        # Remove the node from path
        path.remove(minList)
        # update init, g
        init_node = minList[2]
        g = minList[1]
        # Recording order
        expand[init_node.pos[0]][init_node.pos[1]] = val
        val += 1

    # print(routes)
    return routes, expand

def return_path(path):
    """
   return path
    """
    coord = []
    try:
        dest = path[-1][2]
        while True:
            if dest != None:
                coord.append(dest.pos)
                dest = dest.parent
            else:
                break

        return coord[::-1]
    except:
        return coord

def find_path(maze, start, end):
    """
    Generate paths through A* algorithm
    Input: value map, starting point coordinates, end point coordinates
    Output: Global Guidance
    """
    # Ensure start and end are lists or tuples of length 2
    if not (isinstance(start, (list, tuple)) and len(start) == 2):
        raise ValueError("start must be a list or tuple of length 2")
    if not (isinstance(end, (list, tuple)) and len(end) == 2):
        raise ValueError("end must be a list or tuple of length 2")
    
    # Eval debug of A*
    '''
    print(f"Finding path from {start} to {end}")
    print(f"Maze shape: {maze.shape}")
    #'''
    
    h_map = heuristic_generator(maze, end)
    # cost is the cost of moving once
    cost = 1
    # Optional movement direction
    delta = [[0, -1],  # go up
             [-1, 0],  # go left
             [0, 1],   # go down
             [1, 0]]   # go right
    
    '''
    A* algorithm 
    - input: 
    value map, starting point coordinates, end point coordinates, cost, optional movement direction, heuristic function
    '''
    # print(f"start: {start}")
    # print(f"end: {end}")
    
    path, expand = a_star(maze, start, end, cost, delta, h_map)

    # Eval debug of A*
    '''
    if path == 'fail':
        print("A* failed to find a path")
        print(f"Maze:\n{maze}")
    else:
        print(f"Path found with {len(path)} steps")
    #'''
    
    return path, expand
