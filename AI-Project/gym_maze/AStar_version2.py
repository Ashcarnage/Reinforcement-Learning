from envs.maze_view_2d import MazeView2D
import time
import numpy as np 
class MazeWorld:
    def __init__(self, grid, start, goal, maze_file = None,QTable = None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.actions = ['N', 'S', 'W', 'E']
        self.maze_view = MazeView2D(maze_name="Amigo - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
                                        enable_render=True)
        self.QTable= QTable
        self.maze_view.reset_robot()
        self.configure()
    
    def configure(self, display=None):
        self.display = display
        
    def is_game_over(self):
        return self.maze_view.game_over

    def render(self,mode = "Human", close = False):
        if close:
            self.maze_view.quit_game()
        return self.maze_view.update()

    def is_validup(self, pos):
        x, y = pos
        x1 = x-1
        return 0 <= x1 < len(self.grid) and 0 <= y < len(self.grid[0]) and (self.grid[x][y] == 1 or self.grid[x1][y] == 4)

    def is_validdown(self, pos):
        x, y = pos
        x1 = x + 1
        return 0 <= x1 < len(self.grid) and 0 <= y < len(self.grid[0]) and (self.grid[x][y] == 4 or self.grid[x1][y] == 1)

    def is_validright(self, pos):
        x, y = pos
        y1 = y + 1
        return 0 <= x < len(self.grid) and 0 <= y1 < len(self.grid[0]) and (self.grid[x][y] == 2 or self.grid[x][y1] == 8)

    def is_validleft(self, pos):
        x, y = pos
        y1 = y - 1

        return 0 <= x < len(self.grid) and 0 <= y1 < len(self.grid[0]) and (self.grid[x][y] == 8 or self.grid[x][y1] == 2)

    def get_successors(self, state):
        successors = []
        for action in self.actions:
            new_pos = self.get_new_position(state, action)
            if new_pos != None:
                successors.append((new_pos, action))

        return successors

    def get_new_position(self, state, action):
        x, y = state
        if action == 'E':
            if self.is_validright(state):
                return x, y + 1
            else:
                return None
        elif action == 'S':
            if self.is_validdown(state):
                return x + 1, y
            else:
                return None
        elif action == 'W':
            if self.is_validleft(state):
                return x, y - 1
            else:
                return None
        elif action == 'N':
            if self.is_validup(state):
                return x - 1, y
            else:
                return None

Dirs = ["N","S","E","W"]
def get_heuristic(state, goal):
    x1, y1 = state
    x2, y2 = goal
    return abs(x2 - x1) + abs(y2 - y1)

def sigmoid(X):
    return 1/(1+np.exp(X))*10

def A_star(MazeWorld,Qtable):
    start = MazeWorld.start
    goal = MazeWorld.goal
    p_q = [(0 + get_heuristic(start, goal), start, [],start)]
    visited = []
    while p_q:
        p_q.sort()
        cost, node, path,state = p_q.pop(0)
        cost =cost - get_heuristic(state, goal)
        if node == goal:
            return path 
        if node not in visited:
            visited.append(node)
            for successor, action in MazeWorld.get_successors(node):
                if successor not in visited:
                    Qtable[f'{successor[0]}{successor[1]}'][Dirs.index(action)] = sigmoid(Qtable[f'{successor[0]}{successor[1]}'][Dirs.index(action)])
                    cost1 = Qtable[f"{successor[0]}{successor[1]}"][Dirs.index(action)]
                    p_q.append((cost + cost1 + get_heuristic(state, goal), successor, path + [action],state))
    return None


def BestFS(MazeWorld):
    start = MazeWorld.start
    goal = MazeWorld.goal
    p_q = [(0, start, [])]
    visited = []
    while p_q:
        p_q.sort()
        cost, node, path = p_q.pop(0)
        if node == goal:
            return path 
        if node not in visited:
            visited.append(node)
            for successor, action in MazeWorld.get_successors(node):
                if successor not in visited:
                    p_q.append((get_heuristic(successor, goal), successor, path + [action]))

# grid = [
#     [2, 4, 8, 8],
#     [1, 2, 2, 4],
#     [1, 8, 8, 1],
#     [2, 2, 1, 8],
# ]
# np.save("/Users/ayushbhakat/Documents/Artifacto/AI-Project/gym_maze/models/kunalsmodel", np.array(grid).T,allow_pickle=False, fix_imports=True)
start = (0, 0)
goal = (9, 9)
grid = np.load("/Users/ayushbhakat/Documents/Artifacto/AI-Project/gym_maze/models/models_1.npy", allow_pickle=False, fix_imports=True).T
maze_file =  "/Users/ayushbhakat/Documents/Artifacto/AI-Project/gym_maze/models/models_1.npy"
Qtable = np.load("/Users/ayushbhakat/Documents/Artifacto/AI-Project/gym_maze/QTables/QTables.npy",allow_pickle=True, fix_imports=True).item()
world = MazeWorld(grid, start, goal,maze_file,QTable=Qtable)
Astar_path = A_star(world,Qtable)
print(Astar_path)

# BestFS_path = BestFS(world)
# print(BestFS_path)

for dir in Astar_path:
    world.render()
    time.sleep(0.3)
    world.maze_view.move_robot(dir)
