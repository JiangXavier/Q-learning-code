import copy
import math
import random
from heapq import *

from pulp import *

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
def s():
    # 定义问题
    prob = LpProblem("MyProblem", LpMaximize)
    # 定义变量
    x = [LpVariable("x{}".format(wh), cat=LpBinary) for wh in range(n)]
    # 定义目标函数
    for t in range(arr_two[0]):
        prob += x[t] * arr_two[t]
    # 定义约束条件
    prob += x[0] + x[1] <= 1
    prob += x[2] + x[3] + x[4] <= 2
    prob += x[5] + x[6] + x[7] + x[8] + x[9] <= 3

    # 求解问题
    prob.solve()

    # 输出结果
    print("Status:", LpStatus[prob.status])
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    print("Objective =", value(prob.objective))
def create_model(c, A, b):
    prob = LpProblem('01 Integer Programming', LpMaximize)
    n = len(c)
    x = [LpVariable('x%d' % i, lowBound=0, upBound=1, cat=LpInteger) for i in range(n)]
    prob += lpDot(c, x)
    for i in range(len(A)):
        prob += lpDot(A[i], x) <= b[i]
    return prob
##模拟退火
def evaluate(C,x):
    return sum([x[i]*C[i] for i in range(len(x))])

def generate_initial_state(num_len):
    return [random.randint(0, 1) for _ in range(num_len)]

def generate_new_state(x,C):
    i = random.randint(0, len(C) - 1)
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
    score = evaluate(C,x)
    #如果问题比较简单，可以适当增大初始温度以加快算法的收敛速度；
    # 如果问题比较复杂，需要更高的求解精度，则应该适当减小初始温度以提高算法的探索程度。
    temperature = 100.0
    #因此，在实际应用中，我们需要根据问题的复杂程度和求解精度的要求来选择
    # 合适的cooling_rate。
    # 如果问题比较简单，可以适当增大cooling_rate以加快算法的收敛速度；
    # 如果问题比较复杂，需要更高的求解精度，则应该适当减小cooling_rate以提高算法的探索程度。
    cooling_rate = 0.005*len(C)
    while temperature > 1e-5:
        x_new = generate_new_state(x,C)
        score_new = evaluate(C,x_new)
        ap = acceptance_probability(score, score_new, temperature)
        if ap > random.random():
            x = x_new
            score = score_new
        temperature *= 1 - cooling_rate
    return x, score


def control_fire(x_opt,A,B):
    for j in range(len(B)):
        if sum([x_opt[i]*A[j][i] for i in range(len(A[j]))]) > B[j]:
            return 0
    return 1

if __name__ == '__main__':
    result = []
    result2 = []
    result3 = []
    arr = ini()
    die = 0
    timejingque , timemohu , timetan = 0 , 0 , 0
    while die < len(arr):
        n = int(arr[die])
        die += 1
        arr_two = []
        B = []
        k = 0
        while (len(arr[die]) >= 3):
            zhong = []
            for ind , j in enumerate(arr[die].split()):
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
        print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
        if status != 1:
            print('Infeasible or unbounded problem')
            continue

        T3 = time.perf_counter()
        # x_opt, score_opt = simulated_annealing(C)
        x_opt, score_opt = simulated_annealing(C)
        compar = copy.deepcopy(C)
        heapify(compar)
        while not control_fire(x_opt,arr_two,B):
            minval = heappop(compar)
            if x_opt[C.index(minval)] == 0:
                continue
            else:
                x_opt[C.index(minval)] = 0
        result2.append(sum([x_opt[i] * C[i] for i in range(len(x_opt))]))
        T4 = time.perf_counter()
        T5 = time.perf_counter()
        mmm = [1] * len(C)
        compar = copy.deepcopy(C)
        heapify(compar)
        while not control_fire(x_opt,arr_two,B):
            minval = heappop(compar)
            if mmm[C.index(minval)] == 0:
                continue
            else:
                mmm[C.index(minval)] = 0
        result3.append(sum([mmm[i] * C[i] for i in range(len(mmm))]))
        T6 = time.perf_counter()
        timejingque += (T2 - T1)
        timemohu += (T4 - T3)
        timetan += (T6 - T5)
        # 打印解
        print('Optimal value:', int(value(prob.objective)))
        result.append(int(value(prob.objective)))
        for v in prob.variables():
            print(v.name, '=', int(v.varValue))

        print("A系数矩阵",arr_two)
        print("B系数矩阵",B)
        print("目标函数系数矩阵",C)
    print("最终结果")
    print(result)
    print(result2)
    print(result3)
    print('精确程序运行时间:%s毫秒' % (timejingque * 1000))
    print('模糊程序运行时间:%s毫秒' % (timemohu * 1000))
    print('纯贪程序运行时间:%s毫秒' % (timetan * 1000))
    print(sum(1 if result[i] == result2[i] else 0 for i in range(len(result))))
    print(sum(1 if result[i] == result3[i] else 0 for i in range(len(result))))
    print(timejingque / timemohu)
