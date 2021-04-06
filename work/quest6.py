import numpy as np
import tqdm

import paddle
from paddle import nn
from paddle_quantum.utils import dagger, gate_fidelity, von_neumann_entropy, purity
from paddle_quantum.circuit import UAnsatz

from visualdl import LogWriter

from read_data import read_data

# 保存文件以支持提交格式
with open('Anwser/Question_6_Answer.txt', 'w') as f:
    f.write('\n')

print('未能完成第六问！后期跟进学习')