import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import fire_env
from DDDQN import DuelingDQN, ReplayBuffer
import gym

GAMMA = 0.99
BATCH_SIZE = 64
LR = 5e-5
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9999
TARGET_UPDATE = 2000
BUFFER_SIZE = 20000
MIN_REPLAY_SIZE = 1000

# 环境初始化
env = fire_env.SimpleFireEnv(render_mode='human')
#env = gym.make('CartPole-v1', render_mode='none')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 网络和优化器
online_net = DuelingDQN(state_dim, action_dim)
target_net = DuelingDQN(state_dim, action_dim)
target_net.load_state_dict(online_net.state_dict())
optimizer = optim.Adam(online_net.parameters(), lr=LR)

buffer = ReplayBuffer(BUFFER_SIZE)
epsilon = EPSILON_START

episode_reward = 0
episode_num = 0
state = env.reset()

# 用于记录训练指标
episode_rewards = []          # 每个episode的总reward
episode_mean_losses = []      # 每个episode的平均loss
episode_mean_qs = []          # 每个episode的平均Q值

# 在一条episode中临时收集（多个step）的loss和Q值
step_losses = []
step_qs = []

global_step=0
while True:
    global_step+=1
    if isinstance(state, tuple):
        state = state[0]  # 处理新版gym返回的tuple

    # Epsilon-greedy 动作选择
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = online_net(state_tensor)
        action = torch.argmax(q_values).item()

    # 执行动作
    next_state, reward, done, truncated, _ = env.step(action)
    done = done or truncated

    # 存储经验
    buffer.push(state, action, reward, next_state, done)
    state = next_state
    episode_reward += reward

    # 训练步骤
    if len(buffer) > MIN_REPLAY_SIZE:
        # 采样batch
        states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        # 计算当前Q值
        current_q = online_net(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            next_actions = online_net(next_states).argmax(dim=1, keepdim=True)
            target_q = target_net(next_states).gather(1, next_actions)
            target_q = rewards.unsqueeze(1) + GAMMA * target_q * (~dones).unsqueeze(1)

        # 计算损失
        loss = F.smooth_l1_loss(current_q, target_q)

        # 优化网络
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.parameters(),max_norm=10.0)
        optimizer.step()

        # 记录当前 batch 的 loss 和 Q 值（这里的 Q 值取均值）
        step_losses.append(loss.item())
        step_qs.append(current_q.mean().item())

        # 衰减epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # 更新目标网络
    if global_step% TARGET_UPDATE == 0:
        target_net.load_state_dict(online_net.state_dict())

    # 重置环境
    if done:
        # 记录当前episode的汇总指标
        episode_rewards.append(episode_reward)
        if len(step_losses) > 0:
            mean_loss = np.mean(step_losses)
            mean_q = np.mean(step_qs)
        else:
            # 如果 buffer 还太小，没有训练，就记录 0 或者 None
            mean_loss = 0.0
            mean_q = 0.0
        episode_mean_losses.append(mean_loss)
        episode_mean_qs.append(mean_q)

        print(f"Episode: {episode_num}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}, "
              f"MeanLoss: {mean_loss:.4f}, MeanQ: {mean_q:.4f}")

        # 清空此 episode 的 step_losses/step_qs，进入下一个episode重新统计
        step_losses = []
        step_qs = []

        # 每 50 个 episode 绘图一次
        if (episode_num + 1) % 1000 == 0:
            plt.figure(figsize=(10, 6))
            # subplot1: reward
            plt.subplot(3, 1, 1)
            plt.plot(episode_rewards, label='Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()

            # subplot2: loss
            plt.subplot(3, 1, 2)
            plt.plot(episode_mean_losses, label='Mean Loss', color='red')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.legend()

            # subplot3: Q
            plt.subplot(3, 1, 3)
            plt.plot(episode_mean_qs, label='Mean Q', color='green')
            plt.xlabel('Episode')
            plt.ylabel('Q Value')
            plt.legend()

            plt.tight_layout()
            plt.show()  # 或者 plt.pause(0.001) 实时显示
            torch.save(online_net.state_dict(), f"dddqn_model_{episode_num + 1}.pth")

        # 重置环境、增加 episode_num
        state,_ = env.reset()
        episode_reward = 0
        episode_num += 1

        # 如果想要渲染人机可视化，可以保留（需注意实时绘图和渲染环境是否冲突）
        # if env.render_mode=="human":
        #     plt.pause(0.001)
