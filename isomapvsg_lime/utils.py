import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
import sklearn.datasets
import torch
from sklearn import preprocessing

np.random.seed(1)


# 稳定性指标版本1
# def Fisi(usecases):
#     length = len(usecases[0])
#     sim = []
#     for i in usecases:
#         i_sim = []
#         for j in usecases:
#             j_sim =[]
#             for k in range(length):
#                 if(i[k] == j[k]):
#                     j_sim.append(1)
#                 else:
#                     j_sim.append(0)
#             i_sim.append(j_sim)
#         sim.append(i_sim)
#
#     return sim


# 稳定性指标版本2（考虑解释翻转）
def Fisi(usecases, values):
    length = len(usecases[0])
    sim = []
    for i, j in zip(usecases, values):
        i_sim = []
        for m, n in zip(usecases, values):
            j_sim =[]
            for k in range(length):
                if((i[k] == m[k]) and (np.sign(j[k])==np.sign(n[k]))):
                    j_sim.append(1)
                else:
                    j_sim.append(0)
            # print(j_sim)
            i_sim.append(j_sim)
        sim.append(i_sim)

    return sim


# JD相似性计算
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def jaccard_distance(usecase):
    sim = []
    for l in usecase:
        i_sim = []
        for j in usecase:
            i_sim.append(1-jaccard_similarity(l, j))
        sim.append(i_sim)
    return sim


# 解释质量
def equality_recall(arr1, arr2):
    length = len(arr1)
    count = 0
    for i in arr2:
        if i in arr1:
            count = count+1
    return count/length





# PCA降维可视化  是所有数据一起降维还是分开降维？
def pca(data_row_path, data_generate_path, sava_name):

    data = pd.read_csv(data_row_path)
    data_row = np.array(data.iloc[:, :-1])
    target = np.array(data.iloc[:, -1])

    # data_generate = np.loadtxt('vae_lime_generated2_data.csv', dtype=np.float32, delimiter=",", skiprows=1)
    data_generate = pd.read_csv(data_generate_path)
    data_generate = np.array(data_generate)

    pca = PCA(n_components=2)
    pca.fit(data_row)

    data_row_pca = pca.fit_transform(data_row)
    data_generate_pca = pca.fit_transform(data_generate)

    # print(data_pca, data_pca.shape)

    # print(data_pca[:, 0])
    # print(data_pca[:, 1])

    plt.title('PCA for EG')

    plt.scatter(data_row_pca[:, 0], data_row_pca[:, 1], marker='o', color='red', label='Original EG')
    plt.scatter(data_generate_pca[:, 0], data_generate_pca[:, 1], marker='o', color='blue', label='Generated EG')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend(loc='best')
    plt.savefig(sava_name)

    plt.show()


# 加载数据
def load_data(use, source, name):
    if use == 'vae':               # 数据生成
        if source == 'sklearn':    # sklearn自带的数据集
            if name == 'diabets':
                sk_data = sklearn.datasets.load_diabetes()
            elif name == 'iris':
                sk_data = sklearn.datasets.load_iris()
            elif name == 'bc':
                sk_data = sklearn.datasets.load_breast_cancer()

            # 数据归一化
            mm = preprocessing.MinMaxScaler()
            mm_data = mm.fit_transform(sk_data.data)

            train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(mm_data,
                                                                                              sk_data.target,
                                                                                              train_size=0.80)
            train = torch.from_numpy(train).float()
            test = torch.from_numpy(test).float()

            return train, test

        else:                                 # uci数据集
            if name == 'wine':
                uci_data = pd.read_csv('../data/winequality-white.csv', delimiter=';')
            elif name == 'parkinsons':
                uci_data = pd.read_csv('../data/parkinsons.csv')
            elif name == 'waveform':
                uci_data = pd.read_csv('../data/waveform.csv')
            elif name == 'EG':
                uci_data = pd.read_csv('../data/ElectricalGrid.csv')
            elif name == 'hepatitis':
                uci_data = pd.read_csv('../data/hepatitis.csv')
            elif name == 'liver':
                uci_data = pd.read_csv('../data/liver_patient.csv')
            elif name == 'bank':
                uci_data = pd.read_csv('../data/bank_process.csv')

            data = np.array(uci_data.iloc[:, :-1])
            target = np.array(uci_data.iloc[:, -1])

            # 数据归一化
            mm = preprocessing.MinMaxScaler()
            mm_data = mm.fit_transform(data)

            X_train, X_test, y_train, y_test = train_test_split(mm_data,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=True,
                                                                shuffle=True)
            train = torch.from_numpy(X_train).float()
            test = torch.from_numpy(X_test).float()

            return train, test

    else:                                   # lime解释
        if source == 'sklearn':
            if name == 'diabets':
                sk_data = sklearn.datasets.load_diabetes()
            elif name == 'iris':
                sk_data = sklearn.datasets.load_iris()
            elif name == 'bc':
                sk_data = sklearn.datasets.load_breast_cancer()

            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(sk_data.data,
                                                                                        sk_data.target,
                                                                                        train_size=0.80)

            return X_train, X_test, y_train, y_test, sk_data.feature_names

        else:
            if name == 'wine':
                uci_data = pd.read_csv('../data/winequality-white.csv', delimiter=';')
            elif name == 'parkinsons':
                uci_data = pd.read_csv('../data/parkinsons.csv')
            elif name == 'waveform':
                uci_data = pd.read_csv('../data/waveform.csv')
            elif name == 'EG':
                uci_data = pd.read_csv('../data/ElectricalGrid.csv')
            elif name == 'hepatitis':
                uci_data = pd.read_csv('../data/hepatitis.csv')
            elif name == 'liver':
                uci_data = pd.read_csv('../data/liver_patient.csv')
            elif name == 'bank':
                uci_data = pd.read_csv('../data/bank_process.csv')

            feature_names = uci_data.columns[0:-1]
            # print(len(feature_names))
            # print(feature_names)

            data = np.array(uci_data.iloc[:, :-1])
            target = np.array(uci_data.iloc[:, -1])
            # print(data)
            # print(data.shape)

            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=True,
                                                                shuffle=True)

            return X_train, X_test, y_train, y_test, feature_names


# train, test, labels_train, labels_test, feature_names = load_data('exp', 'uci', 'liver')

# list1 = np.array([1, 2, 3, 4, 5])
# print(list1)
# list2 = np.array([2, 5, 6, 3, 7])
# recall = equality_recall(list1, list2)
# print(recall)
