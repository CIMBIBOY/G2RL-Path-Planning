from map_generator import heuristic_generator

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
    # 开放列表
    path = []
    routes = []
    val = 1
    # 闭集，表示已经访问过的节点，初始为0，访问后为1
    visited = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    visited[init[0]][init[1]] = 1
    # 路径顺序图，初始为-1，表示不走，起点为1，而后依次增加
    expand = [[-1 for _ in range(len(grid[0]))] for _ in range(len(grid))]
    expand[init[0]][init[1]] = 0
    # 起点
    init = Node(None, init)
    g = 0
    # f = g + h，h为启发函数，表示从当前节点到终点的估计代价（曼哈顿距离）
    f = g + heuristic[init.pos[0]][init.pos[1]]

    minList = [f, g, init]

    while [minList[2].pos[0], minList[2].pos[1]] != goal:
        # 遍历所有可能的移动方向
        for i in range(len(delta)):
            # 移动后的节点
            point = Node(init, [init.pos[0] + delta[i][0], init.pos[1] + delta[i][1]])
            # 判断没有越界
            if 0 <= point.pos[0] < len(grid) and 0 <= point.pos[1] < len(grid[0]):
                # 判断没有访问过
                if visited[point.pos[0]][point.pos[1]] == 0 and grid[point.pos[0]][point.pos[1]] == 0:
                    # 计算新的f，g
                    g2 = g + cost
                    f2 = g2 + heuristic[point.pos[0]][point.pos[1]]
                    path.append([f2, g2, point])
                    visited[point.pos[0]][point.pos[1]] = 1
        # 如果没有可扩展的节点，返回失败
        if not path:
            return 'fail', expand

        # 清空minList
        del minList[:]
        # 找到f最小的节点
        minList = min(path)
        # 将该节点加入到routes中
        routes.append(minList)
        # 从path中删除该节点
        path.remove(minList)
        # 更新init，g
        init = minList[2]
        g = minList[1]
        # 记录顺序
        expand[init.pos[0]][init.pos[1]] = val
        val += 1

    # print(routes)
    return routes, expand

def return_path(path):
    """返回路径
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
    """通过a*算法生成路径
    输入：value map，起点坐标，终点坐标
    输出：
    """
    h_map = heuristic_generator(maze, end)
    # cost是移动一次的代价
    cost = 1
    # 可选的移动方向
    delta = [[0, -1],  # go up
             [-1, 0],  # go left
             [0, 1],  # go down
             [1, 0]]  # go right
    # a*算法，输入：value map，起点坐标，终点坐标，cost，可选的移动方向，启发函数
    path, expand = a_star(maze, start, end, cost, delta, h_map)
    return path, expand
