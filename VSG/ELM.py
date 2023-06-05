from scipy import linalg
import numpy as np


class ELM(object):
    def __init__(self, trainDataSet, NumberofInputNeurons, NumberofHiddenNeurons, NumberofOutputNeurons, attributesNum):
        self.trainDataSet = trainDataSet
        self.inputWeight = np.random.rand(NumberofInputNeurons, NumberofHiddenNeurons) * 2 - 1  # 随机生成输入层到隐层的权重
        self.numOfInputNeurons = NumberofInputNeurons  # 初始化参数
        self.numOfHiddenNeurons = NumberofHiddenNeurons
        self.numOfOutputNeurons = NumberofOutputNeurons
        self.attributesNum = attributesNum
        self.NumberofHiddenNeurons = NumberofHiddenNeurons
        self.attribute_max = None
        self.attribute_min = None
        self.attribute_max_x = np.max(trainDataSet[:, :attributesNum], axis=0)
        self.attribute_min_x = np.min(trainDataSet[:, :attributesNum], axis=0)
        self.attribute_max_y = np.max(trainDataSet[:, attributesNum:], axis=0)
        self.attribute_min_y = np.min(trainDataSet[:, attributesNum:], axis=0)
        self.bias = None

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def train(self):  # 训练ELM模型
        x = self.dataStandardize(self.trainDataSet[:, :self.attributesNum], 'x')
        y = self.dataStandardize(self.trainDataSet[:, self.attributesNum:], 'y')

        self.bias = np.random.rand(1, self.NumberofHiddenNeurons)
        # 求隐层输出
        t = np.dot(x, self.inputWeight)
        HiddenLayerOuputs = self.sigmoid(np.dot(x, self.inputWeight) + self.bias)
        tmp_a = linalg.pinv(HiddenLayerOuputs)
        self.outputWeight = np.dot(linalg.pinv(HiddenLayerOuputs), y)

    def predict(self, test_x):  # 预测模型
        x = self.dataStandardize(test_x, 'x')
        HiddenLayerOuputs = self.sigmoid(np.dot(x, self.inputWeight) + self.bias)
        predict_y = np.dot(HiddenLayerOuputs, self.outputWeight)
        return self.dataUnstandardize(predict_y, 'y')

    def dataStandardize(self, data, label):  # 归一化
        ans = None
        if label == 'x':  # 判断要归一化的是x还是y
            data_Subtract_min = data - self.attribute_min_x
            data_max_sub_min = self.attribute_max_x - self.attribute_min_x
            ans = data_Subtract_min / (data_max_sub_min+0.001)
        else:
            data_Subtract_min = data - self.attribute_min_y
            data_max_sub_min = self.attribute_max_y - self.attribute_min_y
            ans = data_Subtract_min / (data_max_sub_min+0.001)
        return ans

    def dataUnstandardize(self, data, label):  # 反向归一化
        ans = None
        if label == 'x':
            ans = data * (self.attribute_max_x - self.attribute_min_x) + self.attribute_min_x
        else:
            ans = data * (self.attribute_max_y - self.attribute_min_y) + self.attribute_min_y
        return ans


# import loadData
#
# if __name__ == '__main__':
#     np.random.seed(1)
#     data = loadData.readMLCC().astype(float)
#     elm = ELM(data, 12, 6, 1, 12)
#     elm.train()
#     print(elm.predict(data[:, :12]))
