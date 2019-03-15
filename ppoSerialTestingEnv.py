import retro

from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

n_cpu = 12
env = SubprocVecEnv([lambda: retro.make('Airstriker-Genesis') for i in range(n_cpu)])

model = PPO2.load('ppo2_airstriker', env)


obs = env.reset()
while True:
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)
	env.render()
