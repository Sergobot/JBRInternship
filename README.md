# RL-Playground
A small project of mine to explore RL algorithms out there in the wild.

As of now, only MountainCar and BipedalWalker environments are solved, but I plan on adding more in the future.

### TD3 vs BipedalWalker-v3

> Here's a nice [video demonstration](https://sergobot.me/bipedal_walker.mp4) of how TD3 model behaves. It is produced by recording every 50th episode during the training process.

![progress of TD3][td3]

### Double DQN vs MountainCar-v0

> Not-as-fancy [demonstration](https://sergobot.me/mountain_car.mp4) of DDQN algorithm in the MountainCar setting.

![progress of DDQN][ddqn]

### Notes

Even though the training process heavily depends on the CPU, as gym computations are happening there,
I made a good use of laptop's built-in GeForce MX150 to speed trainig up almost twice! Especially the
TD3 algorithm, as its implemetation features larger layers.

[td3]: td3.png "Progress of TD3"
[ddqn]: ddqn.png "Progress of DDQN"
