import sklearn.ensemble
import numpy as np
from sklearn.neural_network import MLPClassifier

from vanilla_lime.utils import load_data
from vanilla_lime.lime_tabular import LimeTabularExplainer
import torch

from vanilla_lime.utils import jaccard_distance
from vanilla_lime.utils import Fisi
import datetime
from time import *


np.random.seed(1)

begin_time = time()


# 加载数据集
train, test, train_labels, test_labels, feature_names = load_data('exp', 'uci', 'EG')

# 训练模型
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, train_labels)
acc = sklearn.metrics.accuracy_score(test_labels, rf.predict(test))   # 分类准确率
print('模型精度: ', acc)


explainer = LimeTabularExplainer(train,
                                 feature_names=feature_names,
                                 discretize_continuous=True,
                                 verbose=False)
# # 1、创建一个解释器对象
# 解释测试集中的每一条样本
final_R2 = []      # 存储测试集中每一条样本十次解释R2的平均值
final_MAE = []     # 存储测试集中每一条样本十次解释MAE的平均值
final_jd = []      # 存储测试集中每一条样本十次解释jaccard系数的平均值
final_fisi = []    # 存储测试     集中每一条样本十次解释fisi系数的平均值
for x in range(test.shape[0]):
    print('第 %d 条样本解释结果' % x)
    feature_single = []
    values_single = []
    R2_single = []
    MAE_single = []
    for i in range(1):
        # 2、调用解释函数对单个测试实例进行解释
        # print(x, test[x])
        exp = explainer.explain_instance(test[x],
                                         blackbox=rf,
                                         predict_fn=rf.predict_proba,
                                         train=train,
                                         labels_train=train_labels,
                                         num_features=5)

        # 打印、可视化解释结果
        exp_list = exp.as_list(label=1)
        # print(exp_list)
        # fig = exp.as_pyplot_figure(label=1)
        # fig.show()

        features = []
        values = []
        for i, j in exp_list:
            features.append(i)
            values.append(j)
        feature_single.append(features)
        values_single.append(values)

        # print('R2的值为：', exp.score)
        R2_single.append(exp.score)
        # print('mae的值为：', exp.mae)
        MAE_single.append(exp.mae)


    # 计算R2系数
    final_R2.append(np.asarray(R2_single).mean())
    # 计算MAE系数
    final_MAE.append(np.asarray(MAE_single).mean())

    # 计算jaccard距离
    sim = jaccard_distance(feature_single)
    # plt.matshow(sim)
    # plt.colorbar()
    # plt.show()
    final_jd.append(np.asarray(sim).mean())

    # 计算FISI系数
    fisi = Fisi(feature_single, values_single)
    final_fisi.append(np.asarray(fisi).mean())


# 计算R2系数
R2 = np.asarray(final_R2).mean()
print('测试集R2系数的平均值：', R2)

# 计算MAE系数
MAE = np.asarray(final_MAE).mean()
# print('测试集MAE系数的平均值：', MAE)

# 计算jaccard距离
JD = np.asarray(final_jd).mean()
# print('测试集jaccard系数的平均值：', JD)

# 计算FISI系数
FISI = np.asarray(final_fisi).mean()
print('测试集FISI系数的平均值：', FISI)

# 将结果写入results.txt文件
# with open('../results/parameter', 'a', encoding='utf-8') as f:
#     # 获取当前时间
#     now_time = str(datetime.datetime.now())
#     f.write('Parkinsons数据集VAE-LIME解释结果\n')
#     f.write(str(noise)+":"+str(R2)+"\n")
#     f.write('\n测试集MAE系数的平均值：' + str(MAE))
#     f.write('\n测试集JACCARD系数的平均值：' + str(JD))
#     f.write('\n测试集FISI系数的平均值：' + str(FISI))
#
#     f.close()

end_time = time()
run_time = end_time - begin_time
print('run_time:', run_time)





