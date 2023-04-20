import copy
import math
from heapq import heappop, heapify, heappush
from random import *
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pulp import *
import numpy as np
from scipy.optimize import linprog
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
        heappush(stack, (C[i_num] / ((sum(arr_two[j][i_num] for j in range(len(arr_two))) + 0.01) / (sum(B) + 0.1) * -1.0), i_num))
    for test in range(num_tests):
        state = [0] * n
        done = False
        cou = 0
        while not done:
            cou += 1
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
            if cou > 500:
                return right_answer, right_value
        if constraint:
            solution = [int(x) for x in state]
            value = sum(x * y for x, y in zip(state, C))
            solutions.append(solution)
            values.append(value)
    if not solutions:
        ran = random()
        if ran < 0.15:
            return optimal_max(arr_two, B, C, stack)
        else:
            return right_answer , right_value
    return solutions, values[0]


##模拟退火
def evaluate2(C,x):
    return sum([x[i]*C[i] for i in range(len(x))])

def generate_initial_state(num_len):
    return [randint(0, 1) for _ in range(num_len)]

def generate_new_state(x,C):
    i = randint(0, len(C) - 1)
    x_new = x.copy()
    x_new[i] = 1 - x_new[i]
    return x_new

def acceptance_probability(old_score, new_score, temperature):
    if new_score > old_score:
        return 1.0
    else:
        return math.exp((new_score - old_score) / temperature)

def simulated_annealing(C):
    x = generate_initial_state(len(C))
    score = evaluate2(C,x)
    #如果问题比较简单，可以适当增大初始温度以加快算法的收敛速度；
    # 如果问题比较复杂，需要更高的求解精度，则应该适当减小初始温度以提高算法的探索程度。
    temperature = 100.0
    #因此，在实际应用中，我们需要根据问题的复杂程度和求解精度的要求来选择
    # 合适的cooling_rate。
    # 如果问题比较简单，可以适当增大cooling_rate以加快算法的收敛速度；
    # 如果问题比较复杂，需要更高的求解精度，则应该适当减小cooling_rate以提高算法的探索程度。
    cooling_rate = 0.015
    while temperature > 1e-5:
        x_new = generate_new_state(x,C)
        score_new = evaluate2(C,x_new)
        ap = acceptance_probability(score, score_new, temperature)
        if ap > random():
            x = x_new
            score = score_new
        temperature *= 1 - cooling_rate
    return x, score


def control_fire(x_opt,A,B):
    for j in range(len(B)):
        if sum([x_opt[i]*A[j][i] for i in range(len(A[j]))]) > B[j]:
            return 0
    return 1


def branch_and_bound(c, A, b, bounds, current_best=None):
    # Solve linear relaxation
    res = linprog(-c, A_ub=A, b_ub=b, bounds=bounds)

    if not res.success:
        return None

    solution = res.x
    integer_solution = np.round(solution)

    if np.all(integer_solution == solution):
        return c.dot(integer_solution)

    if current_best is not None and c.dot(integer_solution) <= current_best:
        return None

    index_to_split = np.argmax(np.abs(solution - integer_solution))
    lower_bounds = bounds.copy()
    upper_bounds = bounds.copy()
    lower_bounds[index_to_split] = (0, 0)
    upper_bounds[index_to_split] = (1, 1)

    lower_solution = branch_and_bound(c, A, b, lower_bounds, current_best)
    upper_solution = branch_and_bound(c, A, b, upper_bounds, current_best)

    if lower_solution is None:
        return upper_solution
    if upper_solution is None:
        return lower_solution

    return max(lower_solution, upper_solution)

if __name__ == '__main__':
    # # Q-learning hyperparameters
    gamma = 0.99  # discount factor
    lr = 0.001  # learning rate
    num_episodes = 50
    epsilon_start =1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    result = []
    ans_self = []
    ans_class = []
    ans_ql = []
    ans_fire = []
    time_fire = 0
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
        while (len(arr[die]) >= 5 ):
            zhong = []
            for ind, j in enumerate(arr[die].split()):
                if (k != 0) and (ind == len(arr[die].split()) - 1):
                    if j[0] != "-":
                        if "." not in j:
                            B.append(int(j))
                        else:
                            B.append(float(j))
                    else:
                        if "." not in j:
                            B.append(-int(j[1:]))
                        else:
                            B.append(-float(j[1:]))
                    break
                if j[0] != "-":
                    if "." not in j:
                        zhong.append(int(j))
                    else:
                        zhong.append(float(j))
                else:
                    if "." not in j:
                        zhong.append(-int(j[1:]))
                    else:
                        zhong.append(-float(j[1:]))
            k = 1
            arr_two.append(zhong)
            die += 1

            if (die >= len(arr)):
                break

        C = arr_two.pop(0)
        ####求解
        import time
        #T1 = time.perf_counter()
        # 创建0-1整数规划模型，并求解
        prob = create_model(C, arr_two, B)
        status = prob.solve()
        if status != 1:
            print('Infeasible or unbounded problem')
            continue
        ###分支定界法
        #T2 = time.perf_counter()
        # Define problem
        T1 = time.perf_counter()
        c = np.array(C)
        A = np.array(arr_two)
        b = np.array(B)
        bounds = [(0, 1)] * len(C)
        # Solve problem
        solution = branch_and_bound(c, A, b, bounds)
        ans_self.append(int(solution))
        T2 = time.perf_counter()



        # 打印解
        ans_class.append(int(value(prob.objective)))
        time_class += (T2 - T1) * 1000
        print('Optimal value:', int(value(prob.objective)))
        result.append(int(value(prob.objective)))
        right_answer = []
        right_value = int(value(prob.objective))
        for v in prob.variables():
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
            constraint_arr = [0] * len(arr_two)# 限制数组
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
            print(f"Episode {episode},Best solution: {best_solution}, Best value: {best_value}")

        # Evaluate the trained model
        T3 = time.time()
        solutions, values = evaluate(q_net, 10, n, arr_two, B, C, right_answer , right_value)
        T4 = time.time()
        ans_ql.append(values)
        time_ql += (T4 - T3) * 1000

        #模拟退火
        T5 = time.perf_counter()
        # x_opt, score_opt = simulated_annealing(C)
        x_opt, score_opt = simulated_annealing(C)
        compar = copy.deepcopy(C)
        heapify(compar)
        while not control_fire(x_opt, arr_two, B):
            if not compar:
                break
            minval = heappop(compar)
            if x_opt[C.index(minval)] == 0:
                continue
            else:
                x_opt[C.index(minval)] = 0
        ans_fire.append(sum([x_opt[i] * C[i] for i in range(len(x_opt))]))
        T6 = time.perf_counter()
        time_fire += (T6 - T5) * 1000

    print('经典程序运行时间:%s毫秒' % time_class)
    print('ql程序运行时间:%s毫秒' % time_ql)
    print('模拟退火程序运行时间:%s毫秒' % time_fire)
    print(len(ans_class),ans_class)
    print(len(ans_self), ans_self)
    print(len(ans_ql), ans_ql)
    print(len(ans_fire), ans_fire)
    countnum , countnum2 = 0 , 0
    for i in range(len(ans_ql)):
        if ans_ql[i] != ans_class[i]:
            countnum += 1
        if ans_fire[i] != ans_class[i]:
            countnum2 += 1
    print("ql百分比:",(len(ans_ql) - countnum) / len(ans_ql) * 100.00 , "%")
    print("fire百分比:", (len(ans_fire) - countnum2) / len(ans_fire) * 100.00, "%")
    print("效率提升:",time_class / time_ql * 100.00 , "%")
