from random import *
###随机次数
num = int(input())
file = open('data.txt', 'w')
for i in range(num):
    ##value当前变量个数
    value = randint(3,10)
    ##target_arr目标函数系数矩阵
    target_arr = []
    for p in range(value):
        target_arr.append(randint(-50,100))
    ##con约束不等式的个数
    con_arr = []
    for q in range(randint(int(value/2) + 1,10)):
        one_arr = []
        for j in range(value + 1):
            if j == value:
                ##约束右值
                one_arr.append(randint(50,100))
                break
            ran_dom = random()
            if ran_dom >= 0.2:
                one_arr.append(randint(1, 25))
            else:
                one_arr.append(0)
        con_arr.append(one_arr)
    print(value)
    print(target_arr)
    print(con_arr)

    file.write(str(value))
    file.write("\n")
    for val in target_arr:
        file.write(str(val))
        file.write(" ")
    file.write("\n")
    for array in con_arr:
        for val in array:
            file.write(str(val))
            file.write(" ")
        file.write("\n")
file.close()