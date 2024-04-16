import numpy as np
import cv2
import gymnasium

class MazeSolver(gymnasium.Env):
    def _init_(self, maze, target, rat=(0, 0)):
        self.og_rat = rat
        self.path = []
        self.rat = rat
        self.target = target
        self.tr = 0
        self.info = {}
        self.action_space = gymnasium.spaces.Discrete(4)
        self.maze = maze
        self.height, self.width = maze.shape
        self.observation_space = gymnasium.spaces.Box(low=0, high=1, shape=(self.height, self.width, 2), dtype=np.int16)

    def reset(self, **kwargs):
        self.rat = self.og_rat
        self.path = []
        self.tr = 0
        return (self._get_observation(),self.info)
    
    def step(self, action):
        prev = self.rat
        done = False
        reward = 0
        
        if action == 0:
            self.rat = (self.rat[0] - 1, self.rat[1])
        elif action == 1:
            self.rat = (self.rat[0], self.rat[1] + 1)
        elif action == 2:
            self.rat = (self.rat[0] + 1, self.rat[1])
        else:
            self.rat = (self.rat[0], self.rat[1] - 1)
            
        if self.rat[0] < 0 or self.rat[1] < 0 or self.rat[0] >= self.height or self.rat[1] >= self.width:
            reward += -0.8
            self.rat = prev 
            print("Hit a boundry")
        elif self.maze[self.rat[0]][self.rat[1]] == 0:
            reward += -0.8
            print("Hit obstacle")
            self.rat = prev
        elif self.rat in self.path:
            reward += -0.25
        elif self.maze[self.rat[0]][self.rat[1]] == 0:
            reward += -0.04
        elif self.rat == self.target:
            reward += 1.0
            print("target found")
            done = True
            
        self.tr += reward
        if self.tr < (-0.5 * self.height * self.width):
            print("Overtime")
            done = True
            
        self.path.append(self.rat)
        self.info = {"TimeLimit.truncated": False}
        return self._get_observation(), reward, done, False, self.info

    def _get_observation(self):
        rat_pos = np.zeros((self.height, self.width), dtype=np.int16)
        rat_pos[self.rat[0], self.rat[1]] = 1
        observation = np.dstack((self.maze, rat_pos))
        return observation.astype(np.int16)
    def render(self, mode='human'):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :, 2] = self.maze * 255  
        canvas[self.rat[0], self.rat[1], :] = [0, 255, 0]  
        canvas[self.target[0], self.target[1], :] = [255, 0, 0] 
        resized_canvas = cv2.resize(canvas, None, fx=50, fy=50, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Maze', resized_canvas)
        cv2.waitKey(10)

MazeSolver()

maze=np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
    [ 1.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
], dtype=np.int32)