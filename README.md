# PPO-PyTorch with CVOR

### 项目更新 [2026]：

- **集成 CVOR 控制变量算子**：引入了 Control Variate Off-policy REINFORCE 算子，旨在通过降低策略梯度估计的方差来提升 PPO 在复杂环境下的采样效率。
- **算法融合**：合并了离散（Discrete）与连续（Continuous）动作空间的处理逻辑。
- **动态衰减**：针对连续动作空间增加了 `action_std` 的线性衰减，显著提升了训练稳定性。
- **独立学习率**：支持为 Actor 和 Critic 设置不同的学习率。
- **数据日志**：Episodes、Timesteps 和 Rewards 会自动保存至 `.csv` 文件。
- **可视化工具**：新增绘图脚本（plot_graph.py）及 GIF 生成工具（make_gif.py）。
- **Colab 支持**：提供 `PPO_colab.ipynb` 方便在 Google Colab 上一键运行。

#### [在 Google Colab 中打开 PPO_colab.ipynb](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb)

---

## 项目简介

本项目提供了一个基于 PyTorch 实现的极简 **近端策略优化（PPO）** 算法框架，并特别加入了 **CVOR (Control Variate)** 机制进行优化。该项目主要面向强化学习初学者，旨在通过清晰的代码结构展示 PPO 的核心原理。



为了保持训练过程的简洁与高效：
- **CVOR 优化**：在标准 PPO 的剪切目标函数基础上，利用控制变量算子修正优势估计，减少梯度更新的波动。
- **动作分布**：对于连续环境，采用多元正态分布（对角协方差矩阵）。其标准差（Standard Deviation）作为超参数进行**线性衰减**，而非可训练参数。
- **优势计算**：结合 CVOR 采用蒙特卡洛估计，未采用复杂的 GAE（Generalized Advantage Estimate），以便于快速上手和调试。
- **单线程实现**：采用单线程模型收集经验。

## 使用说明

- **训练新模型**：运行 `train.py`
- **测试预训练模型**：运行 `test.py`
- **绘制训练曲线**：运行 `plot_graph.py`
- **生成演示 GIF**：运行 `make_gif.py`
- **配置参数**：所有训练/测试/绘图相关的超参数均在各自对应的 `.py` 文件中进行配置。

#### 注意事项：
- 如果环境在 CPU 上运行（如 Box-2d 和 Roboschool），建议将 `device` 设置为 `cpu`。对于此类轻量级环境，频繁的 CPU-GPU 数据传输反而会降低训练速度。

---

## 训练结果展示

| PPO + CVOR 连续环境 (RoboschoolHalfCheetah-v1) | 奖励增长曲线 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_gifs/RoboschoolHalfCheetah-v1/PPO_RoboschoolHalfCheetah-v1_gif_0.gif) | ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_figs/RoboschoolHalfCheetah-v1/PPO_RoboschoolHalfCheetah-v1_fig_0.png) |

| PPO + CVOR 离散环境 (LunarLander-v2) | 奖励增长曲线 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_gifs/LunarLander-v2/PPO_LunarLander-v2_gif_0.gif) | ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_figs/LunarLander-v2/PPO_LunarLander-v2_fig_0.png) |

---

## 依赖环境
本项目在以下环境下通过测试：
- Python 3
- PyTorch
- NumPy
- gym (OpenAI)
- pandas & matplotlib (数据处理与绘图)
- Pillow (GIF 生成)

## 参考资料
- [PPO Paper: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
