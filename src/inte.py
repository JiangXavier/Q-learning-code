from _heapq import heapify
from collections import defaultdict
from heapq import heappop, heappush

import numpy as np
from matplotlib import style
import torch
import random


style.use("ggplot")

class Inte:
    def __init__(self, A , B , C ,s , Q_table):
        self.A = A
        self.B = B
        self.C = C
        self.stack = s
        self.constrain_len = len(A)
        self.var_num = len(C)
        self.Q = Q_table
        self.state = ["0"] * len(C)
        self.score = 0

    def reset(self):
        self.score = 0
        self.constrain = [0] * self.constrain_len
        #选择
        self.option = 0
        return "".join(self.state)

    def get_next_states(self):
        heapify(self.stack)
        for i_num in range(len(self.C)):
            heappush(self.stack, (self.C[i_num] / (sum(self.A[j][i_num] for j in range(len(self.A))) / sum(self.B) * -1.0), i_num))
        val = heappop(self.stack)
        return val[1] , int("".join(self.state) , 2)
    #执行一个动作并返回新的状态、奖励、是否结束和调试信息
    def step(self, action, render=True, video=None):
        constrain = [0] * len(self.A)
        mutex = 1
        while mutex:
            if not self.stack:
                break
            value = heappop(self.stack)
            select_var = value[1]
            if self.C[select_var] < 0:
                continue
            for i in range(len(self.A)):
                if constrain[i] + self.A[i][select_var] <= self.B[i]:
                    constrain[i] += self.A[i][select_var]
                else:
                    mutex = 0
                    break
            if mutex:
                self.state[select_var] = "1"
                self.score += self.C[select_var]
                if self.Q.get(int("".join(self.state) , 2)):
                    self.Q[int("".join(self.state), 2)][select_var] += self.C[select_var]
                else:
                    self.Q[int("".join(self.state), 2)] = defaultdict()
                    self.Q[int("".join(self.state), 2)][select_var] = 0
                    self.Q[int("".join(self.state), 2)][select_var] += self.C[select_var]
        return self.C[select_var] , mutex
