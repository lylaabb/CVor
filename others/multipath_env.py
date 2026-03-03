import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class MultiPathTransferEnv(gym.Env):
    def __init__(self, num_paths=3, max_bandwidth=10, max_delay=100, max_data_size=1000):
        super(MultiPathTransferEnv, self).__init__()
        self.num_paths = num_paths
        self.max_bandwidth = max_bandwidth
        self.max_delay = max_delay
        self.max_data_size = max_data_size
        self.observation_space = spaces.Box(low=0, high=max_bandwidth, shape=(num_paths*2 + 2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=max_bandwidth, shape=(num_paths,), dtype=np.float32)
        self.state = None
        self.max_steps = 100
        self.current_step = 0

        # 绘图初始化
        self.fig, self.axs = plt.subplots(num_paths, 1, figsize=(5, num_paths*2))
        plt.ion()  # 开启交互模式
        self.bars = []

    def reset(self):
        self.current_step = 0
        self.state = np.zeros((self.num_paths*2 + 2,), dtype=np.float32)
        self.state[-1] = np.random.randint(1, self.max_data_size+1)
        return self.state

    def step(self, action):
        self.current_step += 1
        self.state[-2] += np.sum(action)
        delays = np.random.randint(1, self.max_delay+1, size=self.num_paths)
        total_bandwidth = np.sum(action)
        total_delay = np.sum(delays)
        reward = total_bandwidth - total_delay
        done = self.current_step >= self.max_steps or self.state[-2] >= self.state[-1]
        return self.state, reward, done, {}

    def render(self, mode='human'):
        if not self.bars:  # 如果bars列表为空，即第一次渲染，创建bar对象
            plt.clf()  # 清除之前的图形
            for i in range(self.num_paths):
                self.axs[i].clear()
                available = self.max_bandwidth - self.state[i]
                used = self.state[i]
                self.bars.append(self.axs[i].barh(['Available', 'Used'], [available, used], color=['blue', 'red']))
                self.axs[i].set_xlim(0, self.max_bandwidth)
                self.axs[i].set_xlabel('Bandwidth')
                self.axs[i].set_title(f'Path {i+1}')
            plt.tight_layout()
        else:  # 更新现有的bar对象
            for i, bar_set in enumerate(self.bars):
                available = self.max_bandwidth - self.state[i]
                used = self.state[i]
                bar_set[0].set_width(available)
                bar_set[1].set_width(used)

        plt.draw()
        plt.pause(0.01)  # 暂停一段时间，让图形更新

    def close(self):
        plt.ioff()  # 关闭交互模式
        plt.show()  # 显示图形

# 测试环境
env = MultiPathTransferEnv()
observation = env.reset()
done = False
total_reward = 0

while not done:
    action = np.random.uniform(0, 10, size=(env.num_paths,))
    observation, reward, done, _ = env.step(action)
    env.render()
    total_reward += reward

env.close()
print("Total Reward:", total_reward)
