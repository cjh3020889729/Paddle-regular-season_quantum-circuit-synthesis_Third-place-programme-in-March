import numpy as np
import tqdm

import paddle
from paddle import nn
from paddle_quantum.utils import dagger, gate_fidelity, von_neumann_entropy, purity
from paddle_quantum.circuit import UAnsatz

from visualdl import LogWriter

from read_data import read_data


# 第三问
num_qubits = 3  # 量子数
# 超参数设置
theta_size = 5 
ITR = 100       # 设置迭代次数
LR = 0.08        # 设置学习速率
SEED = 2        # 固定随机数种子

# 数据集索引id--通过read_data得到所需的数据集
ques2_pathid = 3

print('-- 开始构建模块电路 --')
# 构建第三题的电路模块
def ques3_theta(theta):
    # 初始化电路然后添加量子门
    cir = UAnsatz(num_qubits)
    cir.ry(theta[0], 0)
    cir.ry(theta[1], 1)
    cir.ry(theta[2], 2)
    cir.cnot([0, 1])
    cir.cnot([1, 2])
    cir.ry(theta[3], 0)
    cir.ry(theta[4], 2)
    
    # 返回参数化矩阵
    return cir.U

print('-- 开始构建优化网络 --')
class Optimization_ex3(nn.Layer):
    paddle.seed(SEED)
    def __init__(self, shape, param_attr=nn.initializer.Uniform(
        low=-2.*np.pi, high=2.*np.pi), dtype='float64'):
        super(Optimization_ex3, self).__init__()
        
        # 初始化一个长度为 theta_size的可学习参数列表
        self.theta = self.create_parameter(
        shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

        self.sample_u = paddle.to_tensor(read_data(ques2_pathid))

    # 定义损失函数和前向传播机制
    def forward(self):
        v = ques3_theta(self.theta)

        v = paddle.trace(paddle.matmul(self.sample_u, paddle.transpose(v, [1, 0])))
        loss = 1. - paddle.abs(paddle.real(v)[0][0]) / 8.

        return loss


# vdl记录训练过程
record_writer = LogWriter(logdir='work/record_logs/quest3_logs', file_name='vdlrecords.quest3_log.log')
loss_list = []
parameter_list = []


myLayer = Optimization_ex3([theta_size])

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
data1 = ques3_theta(paddle.to_tensor(parameter_list[-1])).numpy()
data2 = read_data(ques2_pathid)
val = gate_fidelity(data1, data2)
print(val)

# 保存第三问答案
def save_ques3(theta):
    print('-- 开始保存第三问答案 --')
    with open('Anwser/Question_3_Answer.txt', 'w') as f:
        f.write("R 0 {0}\n".format(theta[0]))
        f.write("R 1 {0}\n".format(theta[1]))
        f.write("R 2 {0}\n".format(theta[2]))
        f.write("C 0 1\n")
        f.write("C 1 2\n")
        f.write("R 0 {0}\n".format(theta[3]))
        f.write("R 2 {0}".format(theta[4]))
    print("Hasing Saved!")

save_ques3(parameter_list[-1])