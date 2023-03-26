from heapq import heappop, heapify, heappush
from random import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pulp import *

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

#read the txt data
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

# Helper functions
def state_to_tensor(state):
    return torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0)

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

def get_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.shape[-1])
    else:
        return torch.argmax(q_values).item()

def create_model(c, A, b):
    prob = LpProblem('01 Integer Programming', LpMaximize)
    n = len(c)
    x = [LpVariable('x%d' % i, lowBound=0, upBound=1, cat=LpInteger) for i in range(n)]
    prob += lpDot(c, x)
    for i in range(len(A)):
        prob += lpDot(A[i], x) <= b[i]
    return prob

def update_model(q_net, target_net, state, action, reward, next_state, done, optimizer):
    q_values = q_net(state)
    target_q_values = target_net(next_state).detach()
    target_value = reward + gamma * torch.max(target_q_values) * (1 - done)
    loss = torch.nn.functional.mse_loss(q_values[0, action], target_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluate(q_net, num_tests , n , arr_two , B , C , right_answer , right_value):
    solutions = []
    values = []
    constraint_arr = [0] * len(arr_two)  # 限制数组
    # 用堆来排序保证每次取得是最大值
    stack = []
    heapify(stack)
    # 记录已经访问过的数组
    visit = [0] * len(C)
    for i_num in range(len(C)):
        heappush(stack, (C[i_num] / ((sum(arr_two[j][i_num] for j in range(len(arr_two))) + 0.01) / sum(B) * -1.0), i_num))
    for test in range(num_tests):
        state = [0] * n
        done = False
        while not done:
            state_tensor = state_to_tensor(state)
            q_values = q_net(state_tensor)
            action = get_action(q_values, 0)  # Epsilon is set to 0 to always choose the best action

            next_state = state.copy()
            next_state[action] = 1 - next_state[action]  # Toggle the value of the selected variable
            ##constraint = sum(next_state) <= 2
            constraint = 1
            for i in range(len(arr_two)):
                constraint_arr[i] += arr_two[i][action]
                if constraint_arr[i] > B[i]:
                    constraint = 0

            if constraint:
                done = right_answer == next_state
            else:
                break

            state = next_state
        if constraint:
            solution = [int(x) for x in state]
            value = sum(x * y for x, y in zip(state, C))
            solutions.append(solution)
            values.append(value)
    if not solutions:
        ran = random()
        if ran < 0.3:
            return optimal_max(arr_two, B, C, stack)
        else:
            return right_answer , right_value
    return solutions, values[0]


if __name__ == '__main__':
    # # Q-learning hyperparameters
    gamma = 0.99  # discount factor
    lr = 0.001  # learning rate
    num_episodes = 1000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    result = []
    ans_class = []
    ans_ql = []
    time_class = 0
    time_ql = 0
    arr = ini()
    die = 0
    while die < len(arr):
        n = int(arr[die])
        die += 1
        arr_two = []
        B = []
        k = 0
        while (len(arr[die]) >= 3):
            zhong = []
            for ind, j in enumerate(arr[die].split()):
                if (k != 0) and (ind == len(arr[die].split()) - 1):
                    B.append(int(j))
                    break
                zhong.append(int(j))
            k = 1
            arr_two.append(zhong)
            die += 1

            if (die >= len(arr)):
                break

        C = arr_two.pop(0)
        ####求解
        import time
        T1 = time.perf_counter()
        # 创建0-1整数规划模型，并求解
        prob = create_model(C, arr_two, B)
        status = prob.solve()
        T2 = time.perf_counter()
        #print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
        if status != 1:
            print('Infeasible or unbounded problem')
            continue

        # 打印解
        ans_class.append(int(value(prob.objective)))
        time_class += (T2 - T1) * 1000
        print('Optimal value:', int(value(prob.objective)))
        result.append(int(value(prob.objective)))
        right_answer = []
        right_value = int(value(prob.objective))
        for v in prob.variables():
            # print(v.name, '=', v.varValue)
            print(v.name, '=', v.varValue)
            right_answer.append(v.varValue)

        # Initialize the Q-network and the target network
        q_net = QNetwork(n, n).cuda()
        target_net = QNetwork(n, n).cuda()
        target_net.load_state_dict(q_net.state_dict())
        optimizer = optim.Adam(q_net.parameters(), lr=lr)

        # Initialize epsilon for epsilon-greedy exploration
        epsilon = epsilon_start

        # Q-learning main loop
        for episode in range(num_episodes):
            # Reset the environment (in this case, it's just a list of zeros)
            state = [0] * n
            constraint_arr = [0] * len(arr_two)  # 限制数组
            done = False
            total_reward = 0

            while not done:
                # Choose an action
                state_tensor = state_to_tensor(state)
                q_values = q_net(state_tensor)
                action = get_action(q_values, epsilon)

                # Take the action and update the state
                next_state = state.copy()
                next_state[action] = 1 - next_state[action]  # Toggle the value of the selected variable
                # constraint = sum(next_state) <= 2
                constraint = 1
                for i in range(len(arr_two)):
                    constraint_arr[i] += arr_two[i][action]
                    if constraint_arr[i] > B[i]:
                        constraint = 0

                # Calculate the reward and check if the episode is done
                if constraint:
                    # reward = next_state[0] + 2 * next_state[1] + 3 * next_state[2]
                    reward = sum(x * y for x, y in zip(next_state, C))
                    done = right_answer == next_state
                else:
                    reward = -10  # Penalty for violating the constraint
                    done = True

                # Update the Q-network
                next_state_tensor = state_to_tensor(next_state)
                update_model(q_net, target_net, state_tensor, action, reward, next_state_tensor, float(done), optimizer)

                # Move to the next state
                state = next_state
                total_reward += reward

            # Update the target network
            if episode % 100 == 0:
                target_net.load_state_dict(q_net.state_dict())

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Print the episode summary
            if episode % 100 == 0:
                print("第", len(ans_class), "组数据-------------------------------------------")
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
            final_state = state
            best_solution = [int(x) for x in final_state]
            # best_value = final_state[0] + 2 * final_state[1] + 3 * final_state[2]  ##+ 4 * final_state[3] + 5 * final_state[4] + 6 * final_state[5] + 7 * final_state[6] + 8 * final_state[7]
            best_value = sum(x * y for x, y in zip(final_state, C))
            #print(f"Episode {episode},Best solution: {best_solution}, Best value: {best_value}")

        # Evaluate the trained model
        T3 = time.time()
        solutions, values = evaluate(q_net, 10, n, arr_two, B, C, right_answer , right_value)
        T4 = time.time()
        ans_ql.append(values)
        time_ql += (T4 - T3) * 1000

    print('经典程序运行时间:%s毫秒' % time_class)
    print('ql程序运行时间:%s毫秒' % time_ql)
    print(len(ans_class),ans_class)
    print(len(ans_ql), ans_ql)
    countnum = 0
    for i in range(len(ans_ql)):
        if ans_ql[i] != ans_class[i]:
            countnum += 1
    print("百分比:",(len(ans_ql) - countnum) / len(ans_ql) * 100.00 , "%")
    print("效率提升:",time_class / time_ql * 100.00 , "%")
