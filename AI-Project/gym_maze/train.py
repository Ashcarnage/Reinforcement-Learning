import gymnasium as gym 
import numpy as np
import random 
import time
from envs.Tangle_maze import MazeEnvSample

class QlearningAgent:
    act_dir = ["N","S","E","W"]

    def __init__(self, env : MazeEnvSample, learning_rate =  0.6, discount_factor =  0.9, epsilon = 0.3):
        self.env =  env
        self.observation_space_size =  env.observation_space.high[1] + 1
        self.total_actions = env.total_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.state_history = [] 
        self.Qtable = {f"{i}{j}" : np.zeros(self.total_actions) for j in range(self.observation_space_size) for i in range(self.observation_space_size)}
    
    # def get_walls(self,state):
    #         x0, y0 = state
    #         # find neighbours of the current cells that actually exist
    #         neighbours = dict()
    #         for dir_key, dir_val in env.maze_view.maze.COMPASS.items():
    #             x1 = x0 + dir_val[0]
    #             y1 = y0 + dir_val[1]
    #             # if cell is within bounds
    #             if 0 <= x1 < self.env.maze_view.maze.MAZE_W and 0 <= y1 < self.env.maze_view.maze.MAZE_H:
    #                 # if all four walls still exist
    #                 if self.env.maze_view.maze.get_walls_status(self.env.maze_view.maze.maze_cells[x1, y1])[dir_key]:
    #                 #if self.num_walls_broken(self.maze_cells[x1, y1]) <= 1:
    #                     neighbours[dir_key] = (x1, y1)
    #         return neighbours

    # def get_walls(self,state):
    #     walls = self.env.maze_view.maze.get_walls_status(env.maze_view.maze.maze_cells[state[0],state[1]])
    #     l = []
    #     # print("walls of NORTH : ",walls)
    #     for i in self.act_dir:
    #         # print("i = ",i)
    #         if not bool(walls[i]):
    #             l.append(self.act_dir.index(i))
    #     return l

    def next_action(self, state):
        # print(f" {state}: ------- im ammmmm iron man  -------: ",self.get_walls(state))
        if np.random.rand()< self.epsilon:
            print("__________taking a random sample _________")
            action = self.env.action_space.sample()
            while True:
                if (self.env.maze_view.maze.is_open(state,self.act_dir[action])):
                    return action
                else :
                    action = self.env.action_space.sample()

        # elif len(self.get_walls(state))==3:
        #     print(": ------- im ammmmm iron man  -------: ",self.get_walls(state))
        #     for i in self.act_dir:
        #         if i not in self.get_walls(state):
        #             return self.act_dir.index(i)
        # elif len(self.get_walls(state))==3:
        #     for i in range(len(self.act_dir)):
        #         if i not in self.get_walls(state):
        #             print(f"@@@@@@@@@@@i = {i} : {self.get_walls(state)}")
        #             return i

        else :
            print("__________taking a fixed sample _________")
            # print("lalalal" , f"{state[0]}{state[1]}")
            # print("value : ",np.argmax(self.Qtable[f"{state[0]}{state[1]}"]))
            return np.argmax(self.Qtable[f"{state[0]}{state[1]}"])
        
        # print(f" {state}: ------- im ammmmm iron man closed  -------: ",self.get_walls(state))
    def learn(self, state, action, reward, next_state, done):
        if done:
            q_target = reward
        else:
            q_target =  reward + self.gamma * np.max(self.Qtable[f"{next_state[0]}{next_state[1]}"])

        self.Qtable[f"{state[0]}{state[1]}"][action] += self.lr*(q_target - self.Qtable[f"{state[0]}{state[1]}"][action])


    def train(self, episodes = 100):
        for episode in range(episodes):
            state = self.env.reset()
            score = 0
            self.state_history = []
            print(f"|-------------- Episode : {episode +1} ----------------| ")

            while True:
                action = self.next_action(state)
                # print(f"\nthis is the current state : {state}and the next chosen action {action} \n")
                next_state, reward, done  =  self.env.step(action,state,self.state_history)
                self.state_history.append(tuple(state))
                print(f"reward :{reward}")
                score+=reward 
                self.env.render()

                self.learn(state,action,reward,next_state,done)
                state = next_state

                if not done: 
                    print(f"QValue for pos {next_state} : {self.Qtable[f'{next_state[0]}{next_state[1]}']}")
                    # print("state history : ",self.state_history)
                else:
                    break
                
                time.sleep(0.15)

    def __call__(self):
        state = self.env.reset()
        done =  False 

        while not done :
            action = self.next_action(state)
            next_state, reward, done  =  self.env.step(action)
            state = next_state






# env = MazeEnvSample(maze_size=(30,30))
# a = QlearningAgent(env)
# print(a.observation_space_size)

# l = np.random.randint(0,10,size = (10,20))
# print(l[2])
if __name__ == "__main__":
    maze_size=(10,10)
    env = MazeEnvSample(file_path="/Users/ayushbhakat/Documents/Artifacto/AI-Project/gym_maze/models/models_1.npy")
    # env.maze_view.maze.save_maze(file_path = "/Users/ayushbhakat/Documents/Artifacto/AI-Project/gym_maze/models")
    game  =  QlearningAgent(env)
    game.train()