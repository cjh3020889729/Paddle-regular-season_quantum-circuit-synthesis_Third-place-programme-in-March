import numpy as np
import tqdm

import paddle
from paddle import nn
from paddle_quantum.utils import dagger, gate_fidelity, von_neumann_entropy, purity
from paddle_quantum.circuit import UAnsatz

from visualdl import LogWriter

from read_data import read_data


# 第二问
num_qubits = 2  # 量子数
# 超参数设置
theta_size = 4 
ITR = 100       # 设置迭代次数
LR = 0.5        # 设置学习速率
SEED = 2        # 固定随机数种子

# 数据集索引id--通过read_data得到所需的数据集
ques2_pathid = 2   # quest2的数据集

print('-- 开始构建模块电路 --')
# 构建第二题的电路模块
def ques2_theta(theta):
    # 初始化电路然后添加量子门
    cir = UAnsatz(num_qubits)
    cir.ry(theta[0], 0)
    cir.ry(theta[1], 1)
    cir.cnot([0, 1])
    cir.ry(theta[2], 0)
    cir.ry(theta[3], 1)
    
    # 返回参数化矩阵
    return cir.U

print('-- 开始构建优化网络 --')
class Optimization_ex2(nn.Layer):
    paddle.seed(SEED)
    def __init__(self, shape, param_attr=nn.initializer.Uniform(
        low=-2.*np.pi, high=2.*np.pi), dtype='float64'):
        super(Optimization_ex2, self).__init__()
        
        # 初始化一个长度为 theta_size的可学习参数列表
        self.theta = self.create_parameter(
        shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

        self.sample_u = paddle.to_tensor(read_data(ques2_pathid))

    # 定义损失函数和前向传播机制
    def forward(self):
        v = ques2_theta(self.theta)

        v = paddle.trace(paddle.matmul(self.sample_u, paddle.transpose(v, [1, 0])))
        loss = - paddle.log(paddle.abs(paddle.real(v)[0][0]) / 4.)

        return loss


# vdl记录训练过程
record_writer = LogWriter(logdir='work/record_logs/quest2_logs', file_name='vdlrecords.quest2_log.log')
loss_list = []
parameter_list = []
    
myLayer = Optimization_ex2([theta_size])

optimizer = paddle.optimizer.Adam(
learning_rate = LR, parameters = myLayer.parameters())    

print('-- 开始优化theta参数 --')
# 优化循环
for itr in tqdm.tqdm(range(ITR)):
    
    # 前向传播计算损失函数
    loss = myLayer()[0]
    
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
data1 = ques2_theta(paddle.to_tensor(parameter_list[-1])).numpy()

data2 = read_data(ques2_pathid)
val = gate_fidelity(data1, data2)
print('-- 验证得分: ',val, '--')

# 保存第二问答案
def save_ques2(theta):
    print('-- 开始保存第二问答案 --')
    with open('Anwser/Question_2_Answer.txt', 'w') as f:
        f.write("R 0 {0}\n".format(theta[0]))
        f.write("R 1 {0}\n".format(theta[1]))
        f.write("C 0 1\n")
        f.write("R 0 {0}\n".format(theta[2]))
        f.write("R 0 {0}".format(theta[3]))
    print("-- 完成保存! --")

save_ques2(parameter_list[-1])