import gymnasium as gym
from stable_baselines3 import PPO



# Create the environment, a stacked environment to speed up the training, having more diverse experiences for each step
#env = make_vec_env('LunarLander-v2', n_envs=16)

env = gym.make('LunarLander-v2',render_mode="human")
#env = gym.make('LunarLander-v2')

# Instantiate the agent
#model = PPO('MlpPolicy', env, verbose=1, n_steps = 1024,
#    batch_size = 64,
#    n_epochs = 4,
#    gamma = 0.999,
#    gae_lambda = 0.98,
#    ent_coef = 0.01)
#print("training the model")
#model.learn(total_timesteps=int(1e6))
#model.save("moon_env")
#print("training complete")
#del model # remove to demonstrate saving and loading
model = PPO.load("moon_env")
obs = env.reset()[0]
#env.render()
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated:
        obs = env.reset()[0]


