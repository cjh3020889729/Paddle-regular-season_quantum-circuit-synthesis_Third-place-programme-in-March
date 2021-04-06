# 飞桨常规赛：量子电路合成 3月第3名方案

飞桨常规赛：量子电路合成 3月第3名方案--量桨实现版！
# 赛题分析简要说明

**目的：构建指定量子的量子电路求解最优门的$\theta$值**

## 项目结构
	-|data
		-|data71784
			-|飞桨常规赛：量子电路合成.zip
	-|work
		-|record_logs
		-|read_data.py
		-|quest1.py
		-|quest2.py
		-|quest3.py
		-|quest4.py
		-|quest5.py
		-|quest6.py
		-|requirements.txt
	-|logs_show
		-|train_loss1.png
		-|train_loss2.png
		-|train_loss3.png
		-|train_loss4.png
		-|train_loss5.png
	-README.MD
	-all_works.ipynb

## 结构说明
	-|data目录下为本赛题的数据
	-|work目录下的questx.py为解答赛题的解答程序，其下的record_logs为vdl日志文件
	-|logs_show目录下的图片为对应问题的优化Loss记录曲线图片
	-|all_works.ipynb为本项目的可执行方案

## 使用方式
（务必下载对应的依赖，并按照ipynb内容进行项目的运行！）

A：在AI Studio上上传以上各目录和文件，并将ipynb导入AI Studio，即可根据ipynb内容[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/1620929?channelType=0&channel=0)

## 赛题模型搭建的依赖(可在cpu上运行)


* `paddlepaddle-gpu==2.0.1`

* `openfermion==0.11.0`

* `openfermionpyscf==0.4`

* `paddle-quantum` ：专注与量子计算--[量桨入门](https://qml.baidu.com/quick-start/overview.html)

* `numpy`

* `visualdl`

* `os`

* `tqdm`
   
## 赛题解答核心

利用量桨的优势，结合paddle的反向传播优化机制快速搭建量子电路优化。
    
**模型构建思路：**

   	        1. 设置量子数目，根据量子数目构建量子电路模块--量子数目决定量子电路的输入端个数

            2. 【2-3可交换顺序】设置theta参数形状，从而确定电路过程中y门需要优化的参数$\theta$

            3. 上一步之前可以先配置量子电路网络后再确定其中theta的参数形状

            4. 构建优化网络部分，将theta的形状传入网络中，从而生成可优化参数--通过paddle的优势进行优化【这一步之前需要设计优化损失，自定义——但要根据反向优化的最小值求解为基础设计损失函数，避免优化错误】

            5. 创建优化器Adam、学习率等

            6. 迭代优化，记录theta参数与loss曲线

            7. 根据问题公式评估得分

            8. 保存训练后所需的theta值
            
**损失函数构建思路：**

	利用paddle的最小化优化方法，与优化参数矩阵与目标矩阵的最大相似结合:
    
    
    											创建目标函数: loss = 1 - 相似矩阵求解的值  or  loss = - 相似矩阵求解的值


**问题求解说明：**

1. 问题1：无直接的数据集比对，对给定量子电路，所以只需对应搭建y门电路即可
2. 问题2，3：对给定的电路进行theta优化，然后与直接的数据集中的数据进行比对，得到得分
3. 问题4、5：均参考paddle-quantum的内置若纠错网络结构进行构建电路，实现简单网络求解
   
                说明：
                在问题4中，对弱纠缠网络进行展开成y门进行逐一搭建
                在问题5中，对弱纠缠网络的组件网络进行展开——其源码可参考quantum的real_block_layer

以上网络求解不超过30秒

4. 问题6：简单用单一的弱纠缠网络与强纠缠网络搭建求解电路暂未能解决问题，所以需要未来做一点其它的尝试进行求解（当前未完成）
