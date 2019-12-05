from gym.spaces import Box, Discrete
import gym
import pygame
import random
import collections
import numpy as np
import os
import time
from PIL import Image
import datetime

class Catch(gym.Env):

    def __init__(self):
        pass
    
    def configure(self, render=False, fps=30):
        if not render:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.COLOR_WHITE = (255,255,255)
        self.COLOR_BLACK = (0,0,0)
        self.BLOCK_WIDTH = 98
        self.BLOCK_HEIGHT = 30
        self.GAME_WIDTH = 500
        self.GAME_HEIGHT = 500
        self.BALL_SPEED = 30
        self.render = render
        self.observation_space = Box(low=0.0, high=1.0, shape=(80, 80, 4))
        self.action_space = Discrete(3)
    
    def returnXpos(self):
        return np.random.uniform(47,447)

    def returnFrames(self):
        return np.moveaxis(np.array(self.frames), 0, -1)
    
    def fillEmptyFrames(self):
        self.frames = collections.deque(maxlen=4)
        tmpArray = np.zeros((80, 80))
        for frame in range(0, self.frames.maxlen):
            self.frames.append(tmpArray.copy())

    def reset(self):
        self.BASE_BLOCK_X = 200
        self.BASE_BLOCK_Y = 470
        self.UPPER_BLOCK_X = self.returnXpos()
        self.UPPER_BLOCK_Y = 20
        self.GAME_OVER = False
        self.REWARD = 0
        self.fillEmptyFrames()

        self.screen = pygame.display.set_mode((self.GAME_WIDTH,self.GAME_HEIGHT))
        pygame.display.set_caption('Stack')
        self.clock = pygame.time.Clock()
        self.rect1 = pygame.draw.rect(self.screen,self.COLOR_WHITE,pygame.Rect(self.BASE_BLOCK_X,self.BASE_BLOCK_Y,self.BLOCK_WIDTH,self.BLOCK_HEIGHT))
        pygame.display.flip()

        frame = np.array(Image.fromarray(pygame.surfarray.array2d(self.screen)).resize((80, 80))).astype('float')
        frame /= 255.0
        self.frames.append(frame.copy())
        return self.returnFrames()

    def step(self, action):

        if action == 1: # action = 1 move left
            self.BASE_BLOCK_X -= 30
            if self.BASE_BLOCK_X < 0:
                self.BASE_BLOCK_X = 0
        elif action == 2: # action = 2 move right
            self.BASE_BLOCK_X += 30
            if self.BASE_BLOCK_X > 499:
                self.BASE_BLOCK_X = 400
        else: # action = 0 do nothing
            pass

        self.UPPER_BLOCK_Y += self.BALL_SPEED   

        self.screen.fill(self.COLOR_BLACK)
        self.rect1 = pygame.draw.rect(self.screen,self.COLOR_WHITE,pygame.Rect(self.BASE_BLOCK_X,self.BASE_BLOCK_Y,self.BLOCK_WIDTH,self.BLOCK_HEIGHT))
        self.circle = pygame.draw.rect(self.screen,self.COLOR_WHITE,pygame.Rect(self.UPPER_BLOCK_X,self.UPPER_BLOCK_Y,15,15))
        pygame.display.flip()

        if self.rect1.colliderect(self.circle):
            self.REWARD = 1
            self.GAME_OVER = True
        
        if self.UPPER_BLOCK_Y > self.GAME_WIDTH:
            self.REWARD = -1
            self.GAME_OVER = True

        frame = np.array(Image.fromarray(pygame.surfarray.array2d(self.screen)).resize((80, 80))).astype('float')
        frame /= 255.0
        self.frames.append(frame.copy())

        if self.render:
            self.clock.tick(30)

        return self.returnFrames(), self.REWARD, self.GAME_OVER, {}

'''
if __name__ == "__main__":
    env = Catch()
    env.set_env_props(render=True, fps=40)
    env.reset()
'''