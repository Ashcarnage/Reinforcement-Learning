import gymnasium as gym 
import numpy as np
import cv2
from gymnasium import spaces
from envs.maze_view_2d import MazeView2D

class TangleMaze (gym.Env):
    metadata =  {"render_modes" : ["Human"],"render_fps" : 30}
    ACTION = ["N", "S", "E", "W"]
    def __init__(self, maze_file : np.ndarray = None, maze_size = None, start = None, goal = None, mode = None, enable_render = True):
        super().__init__()
        self.enable_render = enable_render
        self.maze =  maze_file
        self.info =  {}
        # self.height,self.width =  maze.shape 
        self.total_actions = 4
        self.action_space = spaces.Discrete(self.total_actions)
        if maze_file:
            self.maze_view = MazeView2D(maze_name="Amigo - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
                                        enable_render=enable_render)


        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)))
            else : 
                has_loops = False
                num_portals = 0

            self.maze_view =  MazeView2D(maze_name = f"Amigo {maze_size[0]} X {maze_size[1]}",maze_size=maze_size,
                                        screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render)

            
        self.maze_size = self.maze_view.maze_size
        self.target_goal = (self.maze_size[0]-1,self.maze_size[1]-1)

        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high,dtype=np.int64)
        self.state = None
        self.steps_beyond_done = None
        np.random.seed(2)
        self.reset()

        #initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display
    
    def manhattan_distance(self,cell):
        return abs(abs(cell[0]-self.target_goal[0])- abs(cell[1]-self.target_goal[1]))
    def is_invalid_action(self):
        pass

    def is_closer_to_goal(self,state, next_state):
        prev_dist = self.manhattan_distance(state)
        new_dist = self.manhattan_distance(next_state)
        # print("the distances are : ",prev_dist,new_dist)
        if new_dist < prev_dist:
            return True
        return False
    
    def sigmoid(self, reward):
        return 2/(1+np.e**(reward*0.01))-1
    

    def step(self, action, state, state_history):
        progress_reward = 0.5
        revisit_penalty = -0.25
        wall_penalty = -0.1
        state = tuple(state)


        if not self.maze_view.maze.is_open(state,self.ACTION[action]):
            print("\nwall : ")
            reward  = wall_penalty
            done = False
            return state, reward, done
        self.maze_view.move_robot(self.ACTION[action])
        new_cell = tuple(self.maze_view.robot)
        # prev_dist = self.manhattan_distance(self.state)
        # new_dist = self.manhattan_distance(new_cell)
        # print("MANHATTAN DISTANCE : ",prev_dist,new_dist)
        # print()
        # print(f"new cell  : {new_cell}    state_history : ",state_history)
        flag = True
        # print("holaaaa 3: ",state)
        # print("->>>>>>>>  ",new_cell,state,action)
        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            print("\ngoalllll : ")
            reward = 1
            done = True
            flag = False
            return state, reward, done

        if self.is_closer_to_goal(state,new_cell):
            print("\nA step closer : ")
            reward = progress_reward
            done  =  False
            flag = False

        if new_cell[0]==state[0] and new_cell[1]==state[1] :
            print("->>>>>>>>  ",new_cell,state)
            print("\nLoop state : ")
            reward = -0.2
            done = False
            flag = False

        if new_cell in state_history:
            count = 0
            for i in state_history:
                if new_cell==i:
                    count+=1
            print("\naaaaaaaaaaa  hahah   revisited : ")
            reward = revisit_penalty*count
            done = False
            flag = False

        if flag:
            print("\nnormal move : ")
            reward = 0.1
            done = False

        self.state = tuple(self.maze_view.robot)
        # print("next state : ",self.state)

        return self.state, reward, done

    def reset(self):
        self.maze_view.reset_robot()
        self.state = np.zeros(2, dtype= int)
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self,mode = "Human", close = False):
        if close:
            self.maze_view.quit_game()
        return self.maze_view.update()
    

class MazeEnvSample(TangleMaze):
    def __init__(self,file_path = None, maze_size = None, enable_render = True):
        super().__init__(maze_file = file_path ,maze_size=maze_size,enable_render = enable_render)

# m = TangleMaze(maze_size=(30,30))
# print("yooyo")
# print(m.observation_space.sample())
# print(m.is_game_over())
# print(m.render())
# print(m.is_game_over(),