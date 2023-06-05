
from sklearn import datasets, decomposition, manifold

import datetime
from VSG.ELM import ELM
import numpy as np

from VSG import loadData

import sys
from numpy.random import rand
import random
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w", encoding='utf-8')  #

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("..\Data\log.txt")  # 控制台输出信息保存到Data文件夹下

"""
    Isomap使用方法
    class sklearn.manifold.Isomap(n_neighbors=5,n_components=2,eigen_solver=’auto’,tol=0,max_iter=300,path_method=’auto’,neighbors_algorithm=’auto’)
    n_neighbors:近邻参数k
    n_components:指定低维的维数
    eigen_solver:指定求解特征值的算法
	’auto’:由算法自动选取
	‘arpack’:Arpack分解算法
	‘dense’:使用一个直接求解特征值的算法（如LAPACK）
    tol:求解特征算法的收敛阙值
    max_iter
    path_method:指定寻找最短路径的算法
	’auto’:由算法自动选取
	‘FW’:使用Floyd_Warshall算法
	‘D’:使用Dijkstra算法
    neighbors_algorithm:指定计算最近邻的算法
	’ball_tree’:使用BallTree算法
	‘kd_tree’:使用KDTree算法
	‘brute’:使用暴力搜索法


"""




class IsomapVSG(object):
    """参数汇总一下：
        算平均距离时，除数的倍数，越大生成的插值点越多

    说明：
        因为最后要使用MTD判断，平均距离哪里的2倍时，生成700+条插值只有28条符合，那么久只能通过改变倍数的量级，来扩大生成的数量

    """

    def __init__(self, data, k):
        """初始化"""
        self.k = k  # 表示除以平均距离的倍数
        self.data = data
        self.nNeighbors = 10
        self.nComponents = 2
        self.eigenSolver = 'auto'
        self.tol = 0
        self.maxIter = 300
        self.pathMethod = 'D'
        self.neighborsAlgorithm = 'auto'
        self.x_transformed = None
        self.x = None

        self.low = None  # MTD的下界
        self.up = None  # MTD的上界


    def excuteIsomap(self):
        """降维"""
        x = self.data[:, :-1].astype(float)
        self.x = x
        embedding = manifold.Isomap(n_neighbors=self.nNeighbors, n_components=self.nComponents, eigen_solver=self.eigenSolver
                                    , tol=self.tol, max_iter=self.maxIter, path_method=self.pathMethod, neighbors_algorithm=self.neighborsAlgorithm)
        x_transformed = embedding.fit_transform(x)
        self.x_transformed = x_transformed

    def generateVirtualSamples(self, attribute_num):
        """先训练一个ELM模型"""
        """使用原始小样本集合训练ELM，目的是在生成虚拟X后，能够得到虚拟Y"""
        elm1 = ELM(self.data, attribute_num, 64, 1, attribute_num)
        elm1.train()
        """进行降维，映射到2维空间中"""
        self.excuteIsomap()

        """训练第二个ELM"""
        """使用映射到2维空间后的x，与原始维度X训练，目的是得到从2维空间中的插值到原始高维空间的映射函数，
                输入是2维度，输出是原始维度，用的是原始样本对应的数据
            注意：
                1， 论文中的图5显示，从隐层到输出层之间并不是全连接的网络，训练的关系，这里就用全连接网络来代替了
                    """

        data_elm2 = np.hstack((self.x_transformed,self.x))
        elm2 = ELM(data_elm2, 2, 2*attribute_num, attribute_num, 2)  # 文中给的elm2的图显示，要求隐层数量是输出层的2倍，而且隐层到输出层之间也不是全连接的。
        elm2.train()

        """生成隐层插值
        其中平均距离的计算按照自己理解
        """
        count_num = 0
        dist_sum = 0.0
        node_num = len(self.x_transformed)
        """计算平均距离"""
        for node1_index in range(0, node_num):
            for node2_index in range(node1_index+1, node_num):
                node1 = self.x_transformed[node1_index]
                node2 = self.x_transformed[node2_index]
                dist_sum += np.sqrt(((node1[0]-node2[0])**2) + ((node1[1]-node2[1])**2))
                count_num += 1
        avg_dist = dist_sum / (count_num*self.k)   # 平均距离计算的有点远，这里将除数放大两倍

        """产生插值"""
        generated_node = []
        for node1_index in range(0, node_num):
            for node2_index in range(node1_index+1, node_num):
                '''
                先计算两个点之间的距离是平均距离的几倍，这个倍数要向下取整，然后产生（倍数-1）个插值
                再利用ELM2映射得到的原始X
                '''
                node1 = list(self.x_transformed[node1_index])
                node2 = list(self.x_transformed[node2_index])
                tmp_dist = np.sqrt(((node1[0]-node2[0])**2) + ((node1[1]-node2[1])**2))
                times = tmp_dist / avg_dist
                x_len = node2[0] - node1[0]
                generate_num = np.floor(times)
                k = (node2[1]-node1[1]) / (node2[0]-node1[0])  # 斜率
                key = 0
                while key < generate_num:  # 是num倍，所以生成num-1个插值
                    node1[0] = node1[0] + (x_len/generate_num)
                    node1[1] = node1[1] + k * (x_len/generate_num)
                    generated_node.append([node1[0],node1[1]])
                    key += 1

        generated_nodes = np.array(generated_node)
        """使用ELM2 进行反向特征映射"""
        generated_samples_x = elm2.predict(generated_nodes)

        """通过第一个ELM，得到输出Y"""
        generated_samples_y = elm1.predict(generated_samples_x)
        generated_samples = np.hstack((generated_samples_x, generated_samples_y))

        """先不进行MTD判断，看看效果怎么样"""
        """使用MTD判断是否符合范围要求"""
        self.MTD()
        virtual_samples_index = []  # 符合要求的样本的索引
        is_fit_metrix = (generated_samples_x > self.low) & (generated_samples_x < self.up)
        for i in range(len(is_fit_metrix)):
            if all(is_fit_metrix[i]):
                virtual_samples_index.append(i)

        return generated_samples_x[virtual_samples_index]

    def MTD(self):
        """论文中文字的介绍的算法那实现不出域范围扩展的效果
            因此换成直接用论文中图片的计算方法。
        """
        data = self.data[:, 0:-1]
        maximum = np.max(data, axis=0)
        minimum = np.min(data, axis=0)
        CL = np.mean(data, axis=0)
        Uset = (maximum + minimum) / 2
        Nu = np.sum(data >= CL, axis=0)
        Nl = np.sum(data <= CL, axis=0)
        skewL = Nl / (Nu + Nl)
        skewU = Nu / (Nu + Nl)
        s2x = sum((data - CL) ** 2) / len(data)  # 方差
        self.low = CL + (CL-minimum)/((1/Nu)-1)
        self.up = CL + (maximum - CL)/(1-(1/Nl))



