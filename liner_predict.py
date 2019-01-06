import math
import numpy as np
def data_read(filename,k):
    """读入数据，当读训练数据时k取0，返回两个矩阵，测试数据返回一个矩阵"""
    ft = open(filename)
    lx = []
    ly = []
    read = ft.readlines()
    for line in read:
        te = list(line.rstrip('\n').split(','))
        if te[0] == 'M':#将M、F、I替换为三维向量
            te[0] = '1'
            te.insert(0, '0')
            te.insert(0, '0')
        elif te[0] == 'F':
            te[0] = '0'
            te.insert(0, '1')
            te.insert(0, '0')
        else:
            te[0] = '0'
            te.insert(0, '0')
            te.insert(0, '1')
        te_int = [float(x) for x in te]
        if k == 0:
            ly.append([te_int[-1]])
            te_int.pop()
        lx.append(te_int)
        mat_x = np.mat(lx)
        mat_y = np.mat(ly)
        if k == 0:
            return mat_x, mat_y
        else:
            return mat_x

def olsb(train_file,test_file,k):
    """最小二乘法，局部加权"""
    mat_x,mat_y = data_read(train_file,0)#读入数据
    mat_a = data_read(test_file,1)
    ans = []
    length = len(mat_a)
    for i in range(length):
        weight = np.mat(np.eye(len(mat_x)))#权重矩阵
        for j in range(length):
            diff = mat_a[i, :]-mat_x[j,:]#计算当前测试点和所有训练点的距离
            weight[j, j] = math.exp(diff * diff.T / (-2.0 * k ** 2))#权重矩阵赋值
            mat_t = mat_x.T * (weight * mat_x)
            if np.linalg.det(mat_t) == 0.0:#如果没有逆矩阵，求广义逆矩阵
                mat_b = np.linalg.pinv(mat_t) * (mat_x.T * (weight * mat_y))
            else:
                mat_b = mat_t.I * (mat_x.T * (weight * mat_y))
            ans.append(float(mat_a[i, :] * mat_b))
    return ans

def data_write(filename,ans):#结果写回
    f1 = open(filename, 'w')
    for i in range(len(ans)):
        f1.write(str(ans[i]) + '\n')
    f1.close()
k = 0.27
train_file = r"D:\train.txt"
test_file = r"D:\test.txt"
ans_file = r"D:\ans.txt"
ans = olsb(train_file,test_file,k)
data_write(ans_file,ans)