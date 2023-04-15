def inint():
    txt = []
    with open('hello.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            txt.append(line.strip().split(","))  # strip()方法用于去除行末的换行符
    return txt

def r(A,B,C):
    # 打开文件，如果文件不存在则创建
    with open('data.txt', 'a') as f:
        # 写入数据
        f.write(str(len(C)) + "\n")
        for i in range(len(C)):
            f.write(str(C[i]) + " ")
        f.write("\n")
        for i in range(len(A[0])):
            f.write(str(A[0][i]) + " ")
        f.write(str(B[0]))
        f.write("\n")
        # for i in range(len(C)):
        #     f.write("1" + " ")
        # f.write(str(2 * len(C)) + " ")
        # f.write("\n")

if __name__ == '__main__':
    a = inint()
    n = len(a)
    i = 0
    while i < len(a):
        if a[i][0][0] == "k":
            i += 1
            continue
        A, z , C , B = [], [], [], []
        n = 0
        while not a[i][0][0] == "-":
            if a[i][0][0] == "n":
                n = int(a[i][0][2:])
                i += 1
                continue
            elif a[i][0][0] == "c":
                B.append(int(a[i][0][2:]))
                i += 1
                continue
            elif a[i][0][0] == "t" or a[i][0][0] == "z":
                i += 1
                continue
            C.append(int(a[i][1]))
            z.append(int(a[i][2]))
            i += 1
        A.append(z)
        i += 2
        print(A)
        print(B)
        print(C)
        r(A,B,C)