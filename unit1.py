import gymnasium as gym
from stable_baselines3 import PPO



# Create the environment, a stacked environment to speed up the training, having more diverse experiences for each step
#env = make_vec_env('LunarLander-v2', n_envs=16)

env = gym.make('LunarLander-v2',render_mode="human")

model = PPO.load("moon_env")
obs = env.reset()[0]
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated:
        obs = env.reset()[0]


