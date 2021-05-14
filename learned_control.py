
"""
This script allows a trained policy to control the simulator.
Usage:

"""

import sys
import argparse
import pyglet
import math
from pyglet import clock
import numpy as np
import gym
import gym_miniworld
import torch

from policy import DDPGActor
import os 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step(action):
    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))

    obs, reward, done, info = env.step(action)
    if env.is_render_depth:
        print("obs len", len(obs))
        for i in range(len(obs)):
            #print("obs[%d] shape: %s" % (i, obs[i].shape))
            print(obs[i])
    else:
        print("obs shape: ", obs.shape)
        print('min: %f, max: %f' % (np.amin(obs), np.amax(obs)))

    #if reward > 0:
    print('reward={:.2f}'.format(reward))

    if done:
        print('done!')
        obs = env.reset()

    env.render('pyglet', view=view_mode)

    return obs


#if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='MiniWorld-Hallway-v0')
parser.add_argument('--checkpoint', default='results/DDPG-best_reward.pth')
parser.add_argument('--no-time-limit', action='store_true', help='ignore time step limits')
parser.add_argument('--agent-view', action='store_true', help='show the agent view instead of the top view')
args = parser.parse_args()

env = gym.make(args.env_name)

if args.no_time_limit:
    env.max_episode_steps = math.inf

view_mode = 'agent' if args.agent_view else 'top'

state = env.reset()

# Create the display window
env.render('pyglet', view=view_mode) 

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_action = 1. 

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
}

actor = DDPGActor(**kwargs).to(device)
print("Loading checkpoint: ", args.checkpoint)
actor.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
actor.eval()
  

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    global state
    print("Symbol: ", symbol)
    if symbol == 32:
        step(env.actions.done)
        pyglet.app.exit()
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = actor(state).cpu().data.numpy().flatten()
            action = np.argmax(action)
            print("Action:", action)
            state = step(action)

@env.unwrapped.window.event
def on_key_release(symbol, modifiers):
    pass

@env.unwrapped.window.event
def on_draw():
    env.render('pyglet', view='top')

@env.unwrapped.window.event
def on_close():
    pyglet.app.exit()


# Enter main event loop
pyglet.app.run()
env.close() 