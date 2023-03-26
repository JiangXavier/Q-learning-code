import argparse
import os
import shutil
from random import random, randint, sample
from heapq import *
#from src.inte import Inte
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from collections import deque, defaultdict

from src.inte import Inte


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args

def ini():
    f = open('data.txt')
    line = f.readline().strip()  # 读取第一行
    txt = []
    txt.append(line)
    while line:  # 直到读取完文件
        line = f.readline().strip()  # 读取一行文件，包括换行符
        txt.append(line)
    f.close()# 关闭文件
    txt.pop()
    #print(txt)
    return txt

def optimal_max(A , B , C , stack):
    constrain = [0] * len(A)
    ans = [0] * len(C)
    result = 0
    mutex = 1
    while mutex:
        if not stack:
            break
        value = heappop(stack)
        n = value[1]
        if C[n] < 0:
            continue
        for i in range(len(A)):
            if constrain[i] + A[i][n] <= B[i]:
                constrain[i] += A[i][n]
            else:
                mutex = 0
                break
        if mutex:
            ans[n] = 1
            result += C[n]
    return ans , result


def train(opt,A , B , C , stack , Q_table):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # 环境，要改
    env = Inte( A , B , C ,stack , Q_table)
    model = DeepQNetwork(len(C) , 1 , 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()


    #要改
    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        #state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    while epoch < opt.num_epochs: #迭代循环
        next_steps = env.get_next_states()  #得到下一个状态所需要的动作
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        # 根据随机值来选择策略
        random_action = u <= epsilon
        next_actions, next_states = next_steps
        if torch.cuda.is_available():
            pass
            #next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():
            pass
            #predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            pass
            #index = torch.argmax(predictions).item()

        next_state = next_states
        action = next_actions

        reward, done = env.step(action, render=True)
        #Q_table[int(state , 2)][action] = Q_table[int(state , 2)][action] + opt.learning_rate * [reward + opt.gamma * max(Q_table[mm][action] for mm in next_state) - Q_table[int(state , 2)][action]]

        if torch.cuda.is_available():
            pass
            #next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            state = env.reset()
            if torch.cuda.is_available():
                pass
                #state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        #state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        #next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            pass
            #state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            #next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,))
        writer.add_scalar('Train/Score', final_score, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/linear".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    Q_table = defaultdict()
    arr = ini()
    i = 0
    # 挨个读取数据
    while i < len(arr):
        n = int(arr[i])
        i += 1
        A = []
        B = []
        k = 0
        while (len(arr[i]) >= 3):
            zhong = []
            for ind, j in enumerate(arr[i].split()):
                if (k != 0) and (ind == len(arr[i].split()) - 1):
                    B.append(int(j))
                    break
                zhong.append(int(j))
            k = 1
            A.append(zhong)
            i += 1

            if (i >= len(arr)):
                break
        C = A.pop(0)
        print("A系数矩阵", A)
        print("B系数矩阵", B)
        print("目标函数系数矩阵", C)
        #用堆来排序保证每次取得是最大值
        stack = []
        heapify(stack)
        #记录已经访问过的数组
        visit = [0] * len(C)
        for i_num in range(len(C)):
            heappush(stack , (C[i_num] / (sum(A[j][i_num] for j in range(len(A))) / sum(B) * -1.0) , i_num ))
        #print中间结果
        train(opt , A , B , C , stack , Q_table)
        print(optimal_max(A,B,C,stack))
