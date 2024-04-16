class MazeWorld:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.actions = ['move_up', 'move_down', 'move_left', 'move_right']

    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < len(self.grid) and 0 <= y < len(self.grid[0]) and self.grid[x][y] != '1'

    def get_successors(self, state):
        successors = []
        for action in self.actions:
            if action.startswith('move'):
                new_pos = self.get_new_position(state, action)
                if self.is_valid(new_pos):
                    successors.append((new_pos, action))

        return successors

    def get_new_position(self, state, action):
        x, y = state
        if action == 'move_right':
            return x, y + 1
        elif action == 'move_down':
            return x + 1, y
        elif action == 'move_left':
            return x, y - 1
        elif action == 'move_up':
            return x - 1, y


def get_heuristic(state, goal):
    x1, y1 = state
    x2, y2 = goal
    return abs(x2 - x1) + abs(y2 - y1)


def A_star(MazeWorld):
    start = MazeWorld.start
    goal = MazeWorld.goal
    p_q = [(0 + get_heuristic(start, goal), start, [])]
    visited = []
    while p_q:
        p_q.sort()
        cost, node, path = p_q.pop(0)
        if node == goal:
            return path + [node]
        if node not in visited:
            visited.append(node)
            for successor, action in MazeWorld.get_successors(node):
                if successor not in visited:
                    p_q.append((cost + 1 + get_heuristic(successor, goal), successor, path + [action]))
    return None


grid = [
    ['.', '1', '.', '.', '.'],
    ['.', '.', '.', '1', '.'],
    ['.', '1', '.', '1', '.'],
    ['.', '1', '.', '.', '.'],
    ['.', '1', '.', '1', 'G'],
]
start = (0, 0)
goal = (4, 4)

world = MazeWorld(grid, start, goal)

Astar_path = A_star(world)
print(Astar_path)