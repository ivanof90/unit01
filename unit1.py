import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy




# Create the environment, a stacked environment to speed up the training, having more diverse experiences for each step
#env = make_vec_env('LunarLander-v2', n_envs=16)

env = gym.make('LunarLander-v2',render_mode="human")


model = PPO.load("moon_env")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print("mean reward",mean_reward)
print("std reward",std_reward)
#mean reward 229.8375847608589
#std reward 93.37386161974413


obs = env.reset()[0]
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated:
        obs = env.reset()[0]


