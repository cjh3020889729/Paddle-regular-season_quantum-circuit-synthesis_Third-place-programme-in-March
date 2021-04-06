import numpy as np
import tqdm

import paddle
from paddle import nn
from paddle_quantum.utils import dagger, gate_fidelity, von_neumann_entropy, purity
from paddle_quantum.circuit import UAnsatz

from visualdl import LogWriter

# 赛题问题数据
ques2_path = '飞桨常规赛：量子电路合成/Question_2_Unitary.txt'
ques3_path = '飞桨常规赛：量子电路合成/Question_3_Unitary.txt'
ques4_path = '飞桨常规赛：量子电路合成/Question_4_Unitary.txt'
ques5_path = '飞桨常规赛：量子电路合成/Question_5_Unitary.txt'
ques6_path = '飞桨常规赛：量子电路合成/Question_6_Unitary.txt'

# 赛题数据集
data_paths = [0, 0, ques2_path, ques3_path, ques4_path, ques5_path, ques6_path]

def read_data(set_id):
    '''
        读取赛题数据
        set_id: [2, 3, 4, 5, 6]
    '''
    assert set_id >= 2, \
        "please enter a truth set_id(Must more than 2)."

    import os
    paths = data_paths[set_id]
    assert os.path.isfile(paths), \
        "please enter a real path(Now path is not exist)."
    
    data = np.loadtxt(paths)
    data = np.complex128(data)

    return data
