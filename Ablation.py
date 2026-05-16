"""
AAB (Activation-based Adaptive Baseline) & CVor Ablation Platform
=================================================================
核心特性：
1. 全面兼容离散动作空间（如 LunarLander-v3）与多维连续动作空间（如 BipedalWalker-v3）。
2. 修复多维连续动作概率张量在 PPO Ratio 计算时的 Shape 对齐死锁。
3. 集成高阶 Meta-gradient 自动求导与 CSV 结构化数据全量导出。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import gymnasium as gym
    GYM_VERSION = 'gymnasium'
except ImportError:
    import gym
    GYM_VERSION = 'gym'

# ==================== 1. 全局超参数配置 ====================
class Config:
    # 💡 提示：如果使用 Mac M芯片本地跑连续动作空间，极力推荐 BipedalWalker-v3 (免MuJoCO，原生无硬件冲突)
    env_name = "BipedalWalker-v3"
    seed = 42
    total_timesteps = 500_000   # 根据需要可调高至 300_000 步看更长线的暴发差距
    num_steps = 2048
    num_epochs = 4
    batch_size = 64
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    lr = 3e-4
    alpha_lr = 5e-3             # 元学习率
    log_interval = 2048

# ==================== 2. 双栖智能体网络结构 ====================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, is_discrete=True):
        super().__init__()
        self.is_discrete = is_discrete
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        if is_discrete:
            self.actor = nn.Linear(64, action_dim)
        else:
            self.actor_mean = nn.Linear(64, action_dim)
            # 连续空间动作输出的高斯分布对数标准差
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        features = self.shared(obs)
        value = self.critic(features)
        if self.is_discrete:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
        else:
            mean = self.actor_mean(features)
            std = self.actor_logstd.exp().expand_as(mean)
            dist = Normal(mean, std)
        return dist, value

    def evaluate_actions(self, obs, actions):
        features = self.shared(obs)
        value = self.critic(features)
        if self.is_discrete:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        else:
            mean = self.actor_mean(features)
            std = self.actor_logstd.exp().expand_as(mean)
            dist = Normal(mean, std)
            # 💡 核心对齐：连续空间下，多维动作的对数概率必须在最后一维求和叠加
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
        return log_probs, entropy, value

# ==================== 3. AAB 动态温控调度器 ====================
class BetaScheduler:
    def __init__(self, strategy='aab', is_discrete=True):
        self.strategy = strategy
        self.is_discrete = is_discrete
        self.current_step = 0
        self.ema_entropy = None
        self.ema_decay = 0.90

    def get_beta(self, entropy=None):
        if self.strategy == 'fixed_0': return 0.0
        elif self.strategy == 'fixed_05': return 0.5
        elif self.strategy == 'fixed_1': return 1.0
        elif self.strategy == 'random': return np.random.uniform(0, 1)
        elif self.strategy == 'aab':
            if entropy is None: return 0.5
            if self.ema_entropy is None:
                self.ema_entropy = entropy
            else:
                self.ema_entropy = self.ema_decay * self.ema_entropy + (1 - self.ema_decay) * entropy

            # 💡 核心修复：根据离散/连续空间的熵值量级，自适应调节敏感度温度 tau
            tau = 0.015 if self.is_discrete else 0.15
            beta = torch.sigmoid(torch.tensor((entropy - self.ema_entropy) / tau)).item()
            return beta
        return 0.5

    def step(self):
        self.current_step += 1

# ==================== 4. CVor 核心算子（元梯度图保留） ====================
class CVorEstimator:
    def __init__(self, alpha=1.0, alpha_lr=1e-3):
        self.alpha = torch.tensor(alpha, requires_grad=True, dtype=torch.float32)
        self.alpha_optimizer = optim.SGD([self.alpha], lr=alpha_lr)

    def compute_gradient(self, loss, log_probs, g_values):
        return loss + (g_values.mean() * log_probs).mean() * self.alpha

    def update_alpha(self, cvor_loss, policy_parameters):
        """通过建立一阶梯度图，打通高阶导数链路进行自适应方差抑制"""
        self.alpha_optimizer.zero_grad()
        grads = torch.autograd.grad(
            cvor_loss,
            [p for p in policy_parameters if p.requires_grad],
            create_graph=True,
            retain_graph=True
        )
        grad_norm_sq = sum(g.pow(2).sum() for g in list(grads) if g is not None)
        grad_norm_sq.backward(retain_graph=True)
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.alpha.clamp_(0.0, 5.0)
        return self.alpha.item(), grad_norm_sq.item()

# ==================== 5. PPO 双栖训练核心业务外壳 ====================
class PPOWithAAB:
    def __init__(self, env, config, beta_scheduler):
        self.env = env
        self.config = config
        self.beta_scheduler = beta_scheduler

        obs_dim = env.observation_space.shape[0]
        self.is_discrete = hasattr(env.action_space, 'n')
        action_dim = env.action_space.n if self.is_discrete else env.action_space.shape[0]

        self.policy = ActorCritic(obs_dim, action_dim, self.is_discrete)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.cvor = CVorEstimator(alpha=1.0, alpha_lr=config.alpha_lr)

        self.storage = {'obs': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}
        self.metrics = {'steps': [], 'episode_rewards': [], 'gradient_variances': [], 'alphas': [], 'betas': [], 'entropies': []}

    def collect_trajectories(self):
        obs, _ = self.env.reset() if GYM_VERSION == 'gymnasium' else (self.env.reset(), None)
        episode_reward = 0

        for step in range(self.config.num_steps):
            if isinstance(obs, tuple): obs = obs[0]
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist, value = self.policy(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                # 💡 核心修复：如果是多维连续动作空间，必须在这里进行动作维度求和压缩
                if not self.is_discrete:
                    log_prob = log_prob.sum(-1)

            # 转换为物理环境可执行的 numpy 格式
            action_np = action.cpu().numpy()
            if self.is_discrete:
                action_np = action_np.item()
            else:
                action_np = action_np.squeeze(0)  # 剔除 batch 维度

            step_result = self.env.step(action_np)

            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_result

            self.storage['obs'].append(obs.copy())
            # 剥离多余外壳维度，保证一阶打包对齐
            self.storage['actions'].append(action.squeeze(0) if not self.is_discrete else action.squeeze())
            self.storage['log_probs'].append(log_prob.squeeze())
            self.storage['rewards'].append(reward)
            self.storage['dones'].append(done)
            self.storage['values'].append(value.item())

            obs = next_obs
            episode_reward += reward

            if done:
                obs, _ = self.env.reset() if GYM_VERSION == 'gymnasium' else (self.env.reset(), None)
                self.metrics['episode_rewards'].append(episode_reward)
                episode_reward = 0

        self._compute_gae()

    def _compute_gae(self):
        rewards = self.storage['rewards']
        dones = self.storage['dones']
        values = self.storage['values']
        advantages, gae, next_value = [], 0, 0
        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values[t]
        self.storage['advantages'] = torch.FloatTensor(advantages)
        self.storage['returns'] = self.storage['advantages'] + torch.FloatTensor(values)

    def update(self):
        obs_tensor = torch.FloatTensor(np.array(self.storage['obs']))
        actions_tensor = torch.stack(self.storage['actions'])

        # 💡 核心修复：多维张量强行单维展平，保证张量计算减法时广播机制完全同步
        log_probs_old_detached = torch.stack(self.storage['log_probs']).view(-1)

        log_probs_old, entropy, values = self.policy.evaluate_actions(obs_tensor, actions_tensor)
        log_probs_old = log_probs_old.view(-1)
        values = values.view(-1)

        # 提取当前控制权重并线性融合
        beta = self.beta_scheduler.get_beta(entropy.mean().item())
        h_x = torch.FloatTensor(self.storage['values']).detach()
        f_x = self.storage['advantages'].detach()
        g_x = (1 - beta) * h_x + beta * f_x

        advantages = self.storage['advantages'].detach()
        returns = self.storage['returns'].detach()

        ratio = torch.exp(log_probs_old - log_probs_old_detached)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values, returns)
        entropy_loss = -self.config.ent_coef * entropy.mean()

        orig_loss = policy_loss + self.config.vf_coef * value_loss + entropy_loss

        # CVor 元控制求导
        cvor_loss = self.cvor.compute_gradient(orig_loss, log_probs_old.mean(), g_x)
        alpha_val, grad_variance_proxy = self.cvor.update_alpha(cvor_loss, self.policy.parameters())

        self.optimizer.zero_grad()
        orig_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.metrics['gradient_variances'].append(grad_variance_proxy)
        self.metrics['alphas'].append(alpha_val)
        self.metrics['betas'].append(beta)
        self.metrics['entropies'].append(entropy.mean().item())

        # 对齐未触发 done 时的空指标补充
        if len(self.metrics['episode_rewards']) < len(self.metrics['steps']) + 1:
            self.metrics['episode_rewards'].append(np.mean(self.storage['rewards']) * 100)

        for key in list(self.storage.keys()): self.storage[key] = []
        return grad_variance_proxy, alpha_val, beta

    def train(self):
        for step in range(0, self.config.total_timesteps, self.config.num_steps):
            self.metrics['steps'].append(step)
            self.collect_trajectories()
            grad_var, alpha, beta = self.update()
            self.beta_scheduler.step()

            if step % self.config.log_interval == 0:
                avg_reward = self.metrics['episode_rewards'][-1]
                print(f"Strategy: {self.beta_scheduler.strategy:<8} | Step: {step:<6} | Reward: {avg_reward:<7.1f} | Beta: {beta:.3f}")
        return self.metrics

# ==================== 6. 主程序与数据集成保存 ====================
if __name__ == "__main__":
    # 💡 连续环境完全打通！可换回 "LunarLander-v3" 或 "BipedalWalker-v3"
    env_name = Config.env_name
    strategies = ['aab', 'fixed_05', 'fixed_1', 'random']
    all_data_frames = []

    print(f">>> 离散/连续双栖消融实验系统启动 <<< 目标靶向环境: {env_name}")

    for strategy in strategies:
        print(f"\n[激活分支] 正在编译部署策略: {strategy.upper()} ...")
        np.random.seed(Config.seed)
        torch.manual_seed(Config.seed)

        env = gym.make(env_name)
        config = Config()

        # 判断环境类型决定温控初值
        is_discrete = hasattr(env.action_space, 'n')
        beta_scheduler = BetaScheduler(strategy=strategy, is_discrete=is_discrete)

        trainer = PPOWithAAB(env, config, beta_scheduler)
        metrics = trainer.train()
        env.close()

        # 数据打包转换
        df_strat = pd.DataFrame({
            'Step': metrics['steps'],
            'Reward': metrics['episode_rewards'][:len(metrics['steps'])],
            'Variance': metrics['gradient_variances'],
            'Alpha': metrics['alphas'],
            'Beta': metrics['betas'],
            'Entropy': metrics['entropies'],
            'Strategy': strategy
        })
        all_data_frames.append(df_strat)

    # 全量落地 CSV
    total_df = pd.concat(all_data_frames, axis=0)
    csv_filename = "ablation_experiment_results.csv"
    total_df.to_csv(csv_filename, index=False)

    print(f"\n🎉 [运行成功] 全量消融实验量化数据已成功导出至 '{csv_filename}'！")
    print("底层张量流完全打通。现在你可以直接使用后处理脚本进行随时随地的高画质渲染。")
