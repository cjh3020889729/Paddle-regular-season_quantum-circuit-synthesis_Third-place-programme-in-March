'''
    第一问的解答思路代码实现
'''
import numpy as np
import tqdm

import paddle
from paddle import nn
from paddle_quantum.utils import dagger, gate_fidelity, von_neumann_entropy, purity
from paddle_quantum.circuit import UAnsatz

from visualdl import LogWriter


num_qubits = 1  # 量子数
# 超参数设置
theta_size = 1 
ITR = 100       # 设置迭代次数
LR = 0.05        # 设置学习速率
SEED = 2        # 固定随机数种子

print('-- 开始构建模块电路 --')
# 构建第一题的电路模块
def ques1_theta(theta):
    # UAnsatz初始化电路
    cir = UAnsatz(num_qubits)

    # 添加y量子门
    cir.ry(theta[0], 0)

    # 返回参数化矩阵
    return cir.U

print('-- 开始构建优化网络 --')
# 构建搜索优化网络
class Optimization_ex1(nn.Layer):
    paddle.seed(SEED)  # 设置全局随机数种子
    def __init__(self, shape, param_attr=nn.initializer.Uniform(
        low=-2.*np.pi, high=2.**np.pi), dtype='float64'):
        '''
            shape: 超参theta的形状，要与量子数和门数对应
            param_attr：所需要的模型参数的初始化方式
        '''
        super(Optimization_ex1, self).__init__()
        
        # 初始化一个长度为 theta_size的可学习参数列表
        self.theta = self.create_parameter(
        shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        
        # 加载目标矩阵
        self.sample_u = paddle.to_tensor(1. / np.sqrt([2.]) * np.asarray([[1, -1], [1, 1]]))

    # 定义损失函数和前向传播机制
    def forward(self):

        v = ques1_theta(self.theta)  # 返回网络矩阵

        v = paddle.trace(paddle.matmul(self.sample_u, paddle.transpose(v, [1, 0])))  # 计算所需要的值

        loss = 1. - paddle.abs(paddle.real(v)[0][0]) / 2.  # paddle.real(v)[0][0]为对应的实数部分

        return loss

# vdl记录训练过程
record_writer = LogWriter(logdir='work/record_logs/quest1_logs', file_name='vdlrecords.quest1_log.log')
loss_list = []      # 记录损失过程
parameter_list = [] # 参数保存--theta参数

myLayer = Optimization_ex1([theta_size])  # 实例优化网络

# 利用Adam优化器来获得相对好的收敛
# 可以改成RMS
optimizers = paddle.optimizer.Adam(learning_rate = LR, parameters = myLayer.parameters())    

# 优化循环
print('-- 开始优化theta参数 --')
for itr in tqdm.tqdm(range(ITR)):
    
    # 前向传播计算损失函数
    loss = myLayer()[0]  # 获取[loss]->loss
    
    loss.backward()
    optimizers.minimize(loss)  # 最小化对应的loss，直接使用step会报错，因为内部参数存在complex
    optimizers.clear_grad()
    
    # 记录学习曲线
    record_writer.add_scalar(tag="train/loss", value=loss.numpy()[0], step=itr+1)
    loss_list.append(loss.numpy()[0])
    parameter_list.append(myLayer.parameters()[0].numpy())
print('-- 优化完成 --')
print('-- 损失函数的最小值是: ', loss_list[-1], '--')

# 验证结果
print('-- 开始验证theta参数 --')
data1 = ques1_theta(paddle.to_tensor(parameter_list[-1])).numpy()
data2 = 1. / np.sqrt([2.]) * np.asarray([[1, -1], [1, 1]])
val = gate_fidelity(data1, data2)
print('-- 验证得分: ',val, '--')

# 保存第一问答案
def save_ques1(theta):
    print('-- 开始保存第一问答案 --')
    with open('Anwser/Question_1_Answer.txt', 'w') as f:
        f.write("R 0 {0}".format(theta[0]))
    print("-- 完成保存! --")

# 保存答案
save_ques1(parameter_list[-1])