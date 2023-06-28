import gym
env = gym.make('procgen:procgen-heist-v0', start_level=0, num_levels=1)
# num_levels=0 - The number of unique levels that can be generated. Set to 0 to use unlimited levels.
# start_level=0 - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
obs = env.reset()
print(obs.shape)  # (64, 64, 3)