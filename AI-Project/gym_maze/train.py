import gymnasium as gym 
import numpy as np
import random 
import time
from envs.Tangle_maze import MazeEnvSample

class QlearningAgent:
    def __init__(self, env : MazeEnvSample, learning_rate =  0.6, discount_factor =  0.9, epsilon = 0.3):
        self.env =  env
        self.observation_space_size =  env.observation_space.high[1] + 1
        self.total_actions = env.total_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.state_history = [] 
        self.Qtable = {f"{i}{j}" : np.zeros(self.total_actions) for j in range(self.observation_space_size) for i in range(self.observation_space_size)}

    def next_action(self, state):
        if np.random.rand()< self.epsilon:
            return self.env.action_space.sample()
        else :
            print("lalalal" , f"{state[0]}{state[1]}")
            print("value : ",np.argmax(self.Qtable[f"{state[0]}{state[1]}"]))
            return np.argmax(self.Qtable[f"{state[0]}{state[1]}"])
        
    def learn(self, state, action, reward, next_state, done):
        if done:
            q_target = reward
        else:
            q_target =  reward + self.gamma * np.max(self.Qtable[f"{next_state[0]}{next_state[1]}"])

        self.Qtable[f"{state[0]}{state[1]}"][action] += self.lr*(q_target - self.Qtable[f"{state[0]}{state[1]}"][action])

        self.state_history.append(tuple(state))

    def train(self, episodes = 100):
        for episode in range(episodes):
            state = self.env.reset()
            score = 0

            while True:
                action = self.next_action(state)
                next_state, reward, done  =  self.env.step(action,self.state_history)
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
                
                time.sleep(0.2)
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
    env = MazeEnvSample(maze_size=maze_size)
    game  =  QlearningAgent(env)
    game.train()