import gym
import torch
import torch.nn.functional as F
import random
from torch.autograd import Variable

# 定义Q网络
class QNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 训练Q网络
def train_q_network(env, q_network, optimizer, num_episodes, gamma=0.99, epsilon=0.1):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # 选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = q_network(Variable(torch.Tensor(state)))
                action = torch.argmax(q_values).item()

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 更新Q网络
            target_q_values = q_network(Variable(torch.Tensor(next_state)))
            target_q_value = reward + gamma * torch.max(target_q_values)
            loss = F.mse_loss(q_network(Variable(torch.Tensor(state)))[action], target_q_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        if episode % 100 == 0:
            print(f"Episode {episode} completed")

    return q_network


# 测试Q网络
def test_q_network(env, q_network, num_episodes):
    total_reward = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            q_values = q_network(Variable(torch.Tensor(state)))
            action = torch.argmax(q_values).item()
            state, reward, done, _ = env.step(action)
            total_reward += reward

    return total_reward / num_episodes


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    q_network = QNetwork(env.observation_space.n, 64, env.action_space.n)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.01)
    num_episodes = 1000

    q_network = train_q_network(env, q_network, optimizer, num_episodes)
    avg_reward = test_q_network(env, q_network, 10)
    print(f"Average reward: {avg_reward}")