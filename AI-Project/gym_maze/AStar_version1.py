import numpy as np
import time
from envs.maze_view_2d import MazeView2D
class Wall:
    dirs =  {"N" : (0,-1),"S" : (0,1),"E" : (1,0),"W" : (-1,0)}
    def __init__(self, maze):
        self.maze  = maze
        self.maze_size  = len(self.maze)
        self.matrix = np.zeros((self.maze_size, self.maze_size))

    def __isBounded(self,cell,dir):
        x,y = cell
        x1 = x + self.dirs[dir][0]
        y1 = y + self.dirs[dir][1]
        if x1>=0 and x1<self.maze_size and y1>=0 and y1<self.maze_size :
            return True 
        return False


    def is_open(self,cell,dir):
        x,y = cell
        x1 = x + self.dirs[dir][0]
        y1 = y + self.dirs[dir][1]
        # print("---->",x1,y1)
        if self.__isBounded((x,y),dir):
            current_cell = bool(self.__get_wall_info(self.maze[x,y])[dir])
            next_cell = bool(self.__get_wall_info(self.maze[x1,y1])[self.__get_opposite_wall(dir)])
            # print(f"for {cell } the way in {dir} is  : ",next_cell or current_cell)
            return current_cell or next_cell
        return False


    def __get_opposite_wall(self, dirs):
        opposite_dirs = ""

        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            opposite_dirs += opposite_dir

        return opposite_dirs


    def __get_wall_info(self,cell):
        wall = {
            "N": (cell & 0x1) >> 0,
            "S": (cell & 0x4) >> 2,
            "E": (cell & 0x2) >> 1,
            "W": (cell & 0x8) >> 3,
        }
        return wall
    

class AgentAstar(Wall):
    Dirs = ["N","S","E","W"]
    COMPASS = {"N" : (0,-1),"S" : (0,1),"E" : (1,0),"W" : (-1,0)}
    def __init__(self, maze = None, QTable =  None, start = (0,0), goal = None, maze_file = None,enable_render = True ):
        super().__init__(maze)
        self.maze = maze
        self.maze_size = len(maze)
        self.QTable = QTable
        self.start = start
        self.goal = (len(maze)-1,len(maze)-1)
        self.maze_view = MazeView2D(maze_name="Amigo - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
                                        enable_render=enable_render)
        self.visited = []
        self.queue = [(0,0,[(0,0,'N')])]
        self.maze_view.reset_robot()
        self.configure()
    
    def configure(self, display=None):
        self.display = display

    def get_heuristic(self,cell):
        return abs(cell[0]-self.goal[0]) + abs(cell[1]-self.goal[1])

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self,mode = "Human", close = False):
        if close:
            self.maze_view.quit_game()
        return self.maze_view.update()

    def next_move(self,cell):
        l = []
        next_actions = {}
        x,y = cell
        self.QTable[f"{x}{y}"] = self.sigmoid(self.QTable[f"{x}{y}"])
        for dir in self.Dirs:
            if self.is_open(cell,dir):
                l.append(dir)
        for i in l:
            next_actions[i] = self.QTable[f"{x}{y}"][self.Dirs.index(i)]
        
        # cost, direction = self.get_greatest_dir(l2)
        # x1 = x+ self.COMPASS[direction][0]
        # y1 = y+ self.COMPASS[direction][1]

        return next_actions
    
    def sigmoid(self,X):
        X = np.array(X)
        return 1/(1+np.exp(X))*10
    
    def Insert_value(self,total_cost, inter_cost , node_path):
        k = 0
        for q_total,q_inter,q_path in self.queue:
            if q_path[-1][:-1]==node_path[-1][:-1]:
                if total_cost < q_total:
                    self.queue[k] = (total_cost,inter_cost,node_path)
                    return 
        self.queue.append((total_cost, inter_cost , node_path))
    
    def AStar(self):
        main_path = None
        while self.queue:
            self.queue.sort(key = lambda x : x[0])
            total_cost, inter_cost, node_path = self.queue.pop(0)

            if node_path[-1][:-1] == self.goal:
                print(node_path)
                main_path=node_path
                break
            if node_path[-1][:-1] not in self.visited:
                self.visited.append(node_path[-1][:-1])
                print(node_path[-1])
                next_actions = self.next_move(node_path[-1][:-1])
                for direc in next_actions:
                    x,y,_ = node_path[-1]
                    x1 = x+ self.COMPASS[direc][0]
                    y1 = y+ self.COMPASS[direc][1]
                    new_inter_cost = inter_cost + next_actions[direc]
                    new_total_cost = new_inter_cost + self.get_heuristic((x1,y1))
                    new_node_path = node_path + [(x1,y1,direc)]
                    self.Insert_value(new_total_cost,new_inter_cost,new_node_path)

        main_path=main_path[1 :]
        for d in main_path:
            self.render()
            time.sleep(0.2)
            self.maze_view.move_robot(d[2])


file_path = "/Users/ayushbhakat/Documents/Artifacto/AI-Project/gym_maze/models/models_1.npy"
file_path1 = "/Users/ayushbhakat/Documents/Artifacto/AI-Project/gym_maze/QTables/QTables.npy"
maze = np.load(file_path, allow_pickle=False, fix_imports=True)
print(maze)
Qtable = np.load(file_path1, allow_pickle=True, fix_imports=True).item()
game = AgentAstar(maze = maze,maze_file = file_path,QTable = Qtable)   
game.AStar()




        

