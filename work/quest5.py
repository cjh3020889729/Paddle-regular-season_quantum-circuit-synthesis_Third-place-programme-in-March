import numpy as np
import tqdm

import paddle
from paddle import nn
from paddle_quantum.utils import dagger, gate_fidelity, von_neumann_entropy, purity
from paddle_quantum.circuit import UAnsatz

from visualdl import LogWriter

from read_data import read_data

# 第五问
num_qubits = 4  # 量子数
# 超参数设置
depth = 8
theta_size = [depth, num_qubits-1, 4]  # 未知——自定义--弱
ITR = 100       # 设置迭代次数
LR = 0.5        # 设置学习速率
SEED = 2        # 固定随机数种子

# 数据集索引id--通过read_data得到所需的数据集
ques2_pathid = 5

print('-- 开始构建模块电路 --')
# 构建第四题的电路模块
def ques5_theta(theta):
    # 初始化电路然后添加量子门
    cir = UAnsatz(num_qubits)

    cir.real_block_layer(theta, depth)
    
    # 返回参数化矩阵
    return cir.U

print('-- 开始构建优化网络 --')
class Optimization_ex5(nn.Layer):
    paddle.seed(SEED)
    def __init__(self, shape, param_attr=nn.initializer.Uniform(
        low=0., high=2*np.pi), dtype='float64'):
        super(Optimization_ex5, self).__init__()
        
        # 初始化一个长度为 theta_size的可学习参数列表
        self.theta = self.create_parameter(
        shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

        self.sample_u = paddle.to_tensor(read_data(ques2_pathid))

    # 定义损失函数和前向传播机制
    def forward(self):
        v = ques5_theta(self.theta)

        v = paddle.trace(paddle.matmul(self.sample_u, paddle.transpose(v, [1, 0])))
        loss = - paddle.abs(paddle.real(v)[0][0]) / 16.

        return loss


# vdl记录训练过程
record_writer = LogWriter(logdir='work/record_logs/quest5_logs', file_name='vdlrecords.quest5_log.log')
loss_list = []
parameter_list = []

myLayer = Optimization_ex5(theta_size)

# 一般来说，我们利用Adam优化器来获得相对好的收敛
# 当然你可以改成SGD或者是RMS prop.
optimizer = paddle.optimizer.Adam(
learning_rate = LR, parameters = myLayer.parameters())    

print('-- 开始优化theta参数 --')
# 优化循环
for itr in tqdm.tqdm(range(ITR)):
    
    # 前向传播计算损失函数
    loss = myLayer()[0]
    
    # 在动态图机制下，反向传播极小化损失函数
    loss.backward()
    optimizer.minimize(loss)
    optimizer.clear_gradients()
    
    # 记录学习曲线
    record_writer.add_scalar(tag="train/loss", value=loss.numpy()[0], step=itr+1)
    loss_list.append(loss.numpy()[0])
    parameter_list.append(myLayer.parameters()[0].numpy())
    
print('-- 优化完成 --')   
print('损失函数的最小值是: ', loss_list[-1])


# 验证结果
print('-- 开始验证theta参数 --')
data1 = ques5_theta(paddle.to_tensor(parameter_list[-1])).numpy()
data2 = read_data(ques2_pathid)
val = gate_fidelity(data1, data2)
print('F: ', val)
if val<0.75:
    val = 0
elif val < ((252)/400.0):
    val = 0
else:
    val = 34*(val - (252)/400.)

print('score: ', val)


'''
保存参数
'''
def __add_real_block(theta, position, f):
    assert len(theta) == 4, 'the length of theta is not right'
    assert 0 <= position[0] < num_qubits and 0 <= position[1] < num_qubits, 'position is out of range'

    k_size = 0
    
    k_size += 2
    f.write("R {0} {1}\n".format(int(position[0]), theta[0]))
    f.write("R {0} {1}\n".format(int(position[1]), theta[1]))
    
    k_size += 8
    f.write("C {0} {1}\n".format(int(position[0]), int(position[1])))
    
    k_size += 2
    f.write("R {0} {1}\n".format(int(position[0]), theta[2]))
    f.write("R {0} {1}\n".format(int(position[1]), theta[3]))

    return k_size

def __add_real_layer(theta, position, f):
    assert theta.shape[1] == 4 and theta.shape[0] == (position[1] - position[0] + 1) / 2,\
        'the shape of theta is not right'
    
    k_size = 0

    for i in range(position[0], position[1], 2):
        k_size += __add_real_block(theta[int((i - position[0]) / 2)], [i, i + 1], f)
    
    return k_size

def real_block_layer(theta, depth, f):
    assert num_qubits > 1, 'you need at least 2 qubits'
    assert len(theta.shape) == 3, 'The dimension of theta is not right'
    _depth, m, block = theta.shape
    assert depth > 0, 'depth must be greater than zero'
    assert _depth == depth, 'the depth of parameters has a mismatch'
    assert m == num_qubits - 1 and block == 4, 'The shape of theta is not right'
    
    k_size = 0

    if num_qubits % 2 == 0:
        for i in range(depth):
            k_size += __add_real_layer(theta[i][:int(num_qubits / 2)], [0, num_qubits - 1], f)
            k_size += __add_real_layer(theta[i][int(num_qubits / 2):], [1, num_qubits - 2], f) if num_qubits > 2 else None
    else:
        for i in range(depth):
            k_size += __add_real_layer(theta[i][:int((num_qubits - 1) / 2)], [0, num_qubits - 2], f)
            k_size += __add_real_layer(theta[i][int((num_qubits - 1) / 2):], [1, num_qubits - 1], f)
        
    return k_size


# 保存第五问答案
def save_ques5(theta):
    print('-- 开始保存第五问答案 --')
    with open('Anwser/Question_5_Answer.txt', 'w') as f:
        k_size = 0
        k_size = real_block_layer(theta, depth, f)
        print('k_size = ', k_size)

    print("Hasing Saved!")

save_ques5(parameter_list[-1])
