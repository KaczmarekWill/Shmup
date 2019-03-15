import retro

from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

counter = 0
def callback(_locals,  _globals):
	global counter
	counter += 1
	if counter % 100 == 0:
		model.save('ppo2_airstriker')
	return True

n_cpu = 12
env = SubprocVecEnv([lambda: retro.make('Airstriker-Genesis') for i in range(n_cpu)])

model = PPO2(CnnLstmPolicy, env, verbose=1, tensorboard_log='./airstriker_ppo_cnnlstmpolicy')

model.learn(total_timesteps=100000000, callback=callback)

model.save('ppo2_airstriker')

obs = env.reset()
while True:
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)
	env.render()
