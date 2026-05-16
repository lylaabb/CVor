import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO_NEW:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy_old_guide = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old_guide.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        self.alpha = 1
        self.cur_grad = 0
        self.old_grad = 0
        self.learning_rate = 1
        self.max_delta = 10e-5

    def load_guide(self, checkpoint_path):
        self.policy_old_guide.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                if isinstance(state, tuple):
                    state = torch.FloatTensor(state[0].astype(np.float32)).to(device)
                else:
                    state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                if isinstance(state, tuple):
                    state = torch.FloatTensor(state[0].astype(np.float32)).to(device)
                else:
                    state = torch.FloatTensor(state).to(device)

                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def compute_new_value(self, state_values, rewards, dist_entropy, expected_entropy):
        # 计算调整里
        adjustment = (rewards - state_values) / (1 + torch.exp(-(dist_entropy - expected_entropy)))
        # 返回新的状态值
        new_values = state_values + adjustment
        return new_values

    def update(self):
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)


        # Monte Carlo estimate of returns
        # advantages = []
        rewards = []
        discounted_reward = 0
        total_loss = 0
        total_grad_all = []
        for reward, is_terminal, old_logprobs_tmp, old_state_values_tmp in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), reversed(old_logprobs), reversed(old_state_values)):
            if is_terminal:
                discounted_reward = 0
            # 原始更新过程
            discounted_reward = reward + self.gamma * (discounted_reward)
            rewards.insert(0, discounted_reward)

            # Dice 步骤
            #self.w = self.w * self.lambda_w + old_logprobs_tmp
            #self.v = self.w - old_logprobs_tmp
            #deps = torch.exp(torch.exp(self.w - self.w.detach()) - torch.exp(self.v - self.v.detach()))

            #discounted_reward = reward + self.gamma * (discounted_reward)   # reward 是否需要增加 DICE
            #discounted_advantages = self.gamma * deps * (discounted_reward - old_state_values_tmp)

            #advantages.insert(0, discounted_advantages)
            #rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        _, state_values_guide, dist_entropy = self.policy_old_guide.evaluate(old_states, old_actions)
        expected_entropy = torch.mean(dist_entropy)  # 计算熵的期望值

        # 计算新的状态值 V'(s)，expected_entropy 现在是预先计算好的
        new_state_values = self.compute_new_value(old_state_values, rewards, dist_entropy, expected_entropy)
        # 计算优势
        advantages = rewards.detach() - new_state_values.detach()

        # calculate advantages
        # advantages = rewards.detach() - old_state_values.detach()
        # calculate advantages
        # advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating alpha actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            '''
            #########################################
            main part for CVor with F = f(x) * log p
            #########################################
            '''
            # base_value = state_values_guide.detach()
            # base_value = (base_value - base_value.mean()) / (base_value.std() + 1e-5)
            #
            # F_value = (base_value.sum() - base_value) * ratios / (len(base_value) - 1)
            # # tilde_F_value = torch.exp(F_value - F_value.detach()).mean()
            # # CVor = (torch.exp(tilde_F_value - tilde_F_value.detach()) - torch.exp(F_value - F_value.detach()))
            # CVor = 1 - torch.exp(F_value - F_value.detach())

            # base_value = state_values_guide.detach()
            # base_value = (base_value - base_value.mean()) / (base_value.std() + 1e-5)
            # deps_w = base_value * ratios
            # deps_v = torch.exp(deps_w - deps_w.detach()).mean()
            # CVor = (torch.exp(deps_v - deps_v.detach()) - torch.exp(deps_w - deps_w.detach()))
            # CVor = deps_w
            '''
            #########################################
            '''

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = - torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) \
                   - 0.01 * dist_entropy

            loss = loss

            self.optimizer.zero_grad()

            loss.mean().backward()
            self.optimizer.step()

            '''
            #########################################
            adaptation for α (alpha)
            #########################################
            '''

            total_grad = []
            total_grad_out = []
            for name, parms in self.policy.named_parameters():
                total_grad.append((parms.grad ** 2).mean())
                total_grad_out.append((parms.grad).mean())
            self.old_grad = self.cur_grad
            self.cur_grad = torch.Tensor(total_grad).cuda().mean()
            delta_grad = torch.abs(self.cur_grad - self.old_grad)
            if delta_grad > self.max_delta:
                self.max_delta = delta_grad
            self.alpha = torch.max(
                torch.min(self.alpha - self.learning_rate * (self.cur_grad - self.old_grad) / self.max_delta,
                          torch.tensor(1.0).cuda()),
                torch.tensor(0.0).cuda()
            )

            '''
            #########################################
            '''

            total_loss += loss.mean()
            total_grad_all.append(torch.Tensor(total_grad_out).cuda().std())

        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        total_loss = total_loss / self.K_epochs
        total_grad_all = torch.Tensor(total_grad_all).mean()

        return total_loss, total_grad_all
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


