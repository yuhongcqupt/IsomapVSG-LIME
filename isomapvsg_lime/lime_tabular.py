"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import copy
from functools import partial
import json
import warnings

import numpy as np
import scipy as sp
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import check_random_state
from sklearn.neighbors import KNeighborsClassifier
import torch
from sklearn.cluster import AgglomerativeClustering
from imblearn.over_sampling import SMOTE

from vanilla_lime.discretize import QuartileDiscretizer
from vanilla_lime.discretize import DecileDiscretizer
from vanilla_lime.discretize import EntropyDiscretizer
from vanilla_lime.discretize import BaseDiscretizer
from vanilla_lime.discretize import StatsDiscretizer
from vanilla_lime import lime_base
from vanilla_lime import explanation

from VSG.IsomapVSG import IsomapVSG



class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""

    def __init__(self, feature_names, feature_values, scaled_row,
                 categorical_features, discretized_feature_names=None,
                 feature_indexes=None):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            scaled_row: scaled row
            categorical_features: list of categorical features ids (ints)
            feature_indexes: optional feature indexes used in the sparse case
        """
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_indexes = feature_indexes
        self.scaled_row = scaled_row

        if sp.sparse.issparse(scaled_row):
            self.all_categorical = False
        else:
            self.all_categorical = len(categorical_features) == len(scaled_row)

        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """
        names = self.exp_feature_names
        if self.discretized_feature_names is not None:
            names = self.discretized_feature_names
        return [(names[x[0]], x[1]) for x in exp]

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                show_table=True,
                                show_all=False):
        """Shows the current example in a table format.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             show_table: if False, don't show table visualization.
             show_all: if True, show zero-weighted features in the table.
        """
        if not show_table:
            return ''
        weights = [0] * len(self.feature_names)
        for x in exp:
            weights[x[0]] = x[1]
        if self.feature_indexes is not None:
            # Sparse case: only display the non-zero values and importances
            fnames = [self.exp_feature_names[i] for i in self.feature_indexes]
            fweights = [weights[i] for i in self.feature_indexes]
            if show_all:
                out_list = list(zip(fnames,
                                    self.feature_values,
                                    fweights))
            else:
                out_dict = dict(map(lambda x: (x[0], (x[1], x[2], x[3])),
                                zip(self.feature_indexes,
                                    fnames,
                                    self.feature_values,
                                    fweights)))
                out_list = [out_dict.get(x[0], (str(x[0]), 0.0, 0.0)) for x in exp]
        else:
            out_list = list(zip(self.exp_feature_names,
                                self.feature_values,
                                weights))
            if not show_all:
                out_list = [out_list[x[0]] for x in exp]
        ret = u'''
            %s.show_raw_tabular(%s, %d, %s);
        ''' % (exp_object_name, json.dumps(out_list, ensure_ascii=False), label, div_name)
        return ret



"""
LimeTabularExplainer：表格数据解释器类
解释步骤：
    1、可解释数据表示（数据离散化）
    2、数据生成（用VAE生成数据；数据离散化；数据二进制化）
    3、数据选择
    4、数据平衡
    5、数据加权，获取标签信息
    6、训练可解释模型
函数：
    1、explain_instance：核心函数，最终返回待解释实例的解释结果
    2、data_generate：数据生成
    3、DataSelection：数据选择
    4、DataBalancing：数据平衡

"""
class LimeTabularExplainer(object):
    """

    """

    def __init__(self,
                 training_data,                    # 训练集
                 mode="classification",            # 分类或者回归任务
                 training_labels=None,             # 训练集的标签
                 feature_names=None,               # 特征名称列表
                 categorical_features=None,        # 分类特征索引列表（下标）
                 categorical_names=None,           # 分类特征的名称列表
                 kernel_width=None,                # 指数核的核宽，默认为sqrt（列数）* 0.75
                 kernel=None,                      # 核
                 verbose=False,                    # 打印线性模型的局部预测值
                 class_names=None,                 # 类别名称列表（可能的标签值）
                 feature_selection='auto',         # 特征选择方法
                 discretize_continuous=True,       # 离散化连续变量
                 discretizer='quartile',           # 离散器
                 sample_around_instance=False,     # 以待解释实例为中心采样
                 random_state=None,                # 随机状态
                 training_data_stats=None):        # 训练集的状态，只在 discretize_continuous为True时有意义
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True
                and data is not sparse. Options are 'quartile', 'decile',
                'entropy' or a BaseDiscretizer instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            training_data_stats: a dict object having the details of training data
                statistics. If None, training data information will be used, only matters
                if discretize_continuous is True. Must have the following keys:
                means", "mins", "maxs", "stds", "feature_values",
                "feature_frequencies"
        """
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance
        self.training_data_stats = training_data_stats

        # 检查训练集状态，如果训练数据的状态不为空的话，就调用validate_training_data_stats函数
        if self.training_data_stats:
            self.validate_training_data_stats(self.training_data_stats)

        if categorical_features is None:
            categorical_features = []

        # 如果特征名为空，那么就用’1‘，’2‘.....代替特证名
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        # 创建离散器对象
        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(training_data):
            # Set the discretizer if training data stats are provided
            if self.training_data_stats:      # 用训练集的状态离散
                discretizer = StatsDiscretizer(training_data, self.categorical_features,
                                               self.feature_names, labels=training_labels,
                                               data_stats=self.training_data_stats,
                                               random_state=self.random_state)

            if discretizer == 'quartile':      # 四分位数离散
                self.discretizer = QuartileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif discretizer == 'decile':      # 十分位数离散
                self.discretizer = DecileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif discretizer == 'entropy':     # 熵离散
                self.discretizer = EntropyDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')

            # 离散化过后，所有特征都变为了类别特征，所以要改变categorical_features的值。将其设置为所有特征的索引下标
            self.categorical_features = list(range(training_data.shape[1]))

            # 如果training_data_stats没提供，离散化训练集
            if(self.training_data_stats is None):
                discretized_training_data = self.discretizer.discretize(training_data)

        # 组建核函数，默认为RBF核函数
        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.feature_selection = feature_selection

        # lime_base类的作用--学习一个局部线性稀疏模型
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

        self.class_names = class_names

        # Though set has no role to play if training data stats are provided
        # 如果training_data_stats不为空，那么这一步不起作用
        # 计算训练集数据的均值、方差、标准差（类别特征的均值设置为0，标准差为1）
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)      # 计算训练集的均值和标准差
        self.feature_values = {}
        self.feature_frequencies = {}

        # 统计每一列特征的取值以及每个值出现的频率,如果离散器不为空，统计的是离散化的训练集的状态，即values是整数
        # 如果需要随机生成数据，则需要用到values和frequencies
        for feature in self.categorical_features:
            if training_data_stats is None:
                if self.discretizer is not None:
                    column = discretized_training_data[:, feature]
                else:
                    column = training_data[:, feature]

                feature_count = collections.Counter(column)   # counter统计词频
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            else:
                values = training_data_stats["feature_values"][feature]
                frequencies = training_data_stats["feature_frequencies"][feature]

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))

            self.scaler.mean_[feature] = 0  # 均值
            self.scaler.scale_[feature] = 1  # 标准差



    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    @staticmethod
    def validate_training_data_stats(training_data_stats):   # 检查/验证训练集状态："means", "mins", "maxs", "stds"等等
        """
            Method to validate the structure of training data stats
        """
        stat_keys = list(training_data_stats.keys())
        valid_stat_keys = ["means", "mins", "maxs", "stds", "feature_values", "feature_frequencies"]
        missing_keys = list(set(valid_stat_keys) - set(stat_keys))
        if len(missing_keys) > 0:
            raise Exception("Missing keys in training_data_stats. Details: %s" % (missing_keys))



    def explain_instance(self,
                         data_row,           # 待解释数据
                         blackbox,
                         predict_fn,         # 预测模型
                         train,              # 用于训练KNN模型
                         labels_train,       # 用于训练KNN模型
                         labels=(1,),
                         top_labels=None,    # 需要解释的top标签
                         num_features=10,    # 解释中含有的最大特征数量
                         num_samples=1000,
                         distance_metric='euclidean',
                         model_regressor=None):
        """
        函数功能：
            为某个待解释实例的预测结果生成解释

        参数：
            data_row：待解释实例--一维 numpy 数组
            predict_fn：预测函数--训练好的分类器/回归器
            data_generator：数据生成模型
            labels：指定需要解释的标签索引--对于多标签任务会有多个预测值
            top_labels：如果不为None，忽略labels，并为有最高预测概率的 K 个标签生成解释，K是参数值。
            num_features：解释中出现的最大特征数目--如线性模型中自变量的个数
            num_samples：学习线性模型的领域大小--需要生成的数据数目
            distance_metric：距离函数--用于计算生成的数据与待解释实例的距离，默认为欧氏距离。
            model_regressor：解释中使用sklearn回归器，默认为岭回归器（见 LimeBase.py）。

        返回值：
            一一个带有相应解释的explanation对象（见explanation.py）
        """

        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()

        # 数据生成
        data, binary_data = self.data_generate(data_row,
                                               blackbox,
                                               num_samples,
                                               train,
                                               labels_train)

        # np.savetxt('../generated_data/vaelime_liver.csv', data)
        # np.save('../generated_data/vaelime_bank.npy', data)

        if sp.sparse.issparse(binary_data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = binary_data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (binary_data - self.scaler.mean_) / self.scaler.scale_

        # 计算距离--有的是二进制数据
        distances = sklearn.metrics.pairwise_distances(
                    scaled_data,
                    scaled_data[0].reshape(1, -1),
                    metric=distance_metric
        ).ravel()

        # 获取采样数据的黑盒预测标签值
        yss = predict_fn(data)
        # print('lime_tabular405行yss：', yss.shape)
        # print('403行待解释实例的分类/预测结果为：', yss[0])

        # 判断是分类任务还是回归任务
        # 对于分类任务，模型一个类别元组列表
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:  # 如果分类名为None，那么将其设置为0、1、2......
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):    # 判断每一个样本被分为每一个类别的概率的和是否和1相近
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # 对于回归任务，输出应该是预测的一维数组
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            # 最大值、最小值的作用
            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery， 每个标签占一行
            yss = yss[:, np.newaxis]

        # 如果特征名称为空，那么将其设置为0、1、2.......
        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)   # 将值转换为两位小数形式
            feature_indexes = None

        # 设置类别特征名
        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]

            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'

        categorical_features = self.categorical_features

        # 设置离散化的特征名
        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(binary_data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)  # 待解释实例离散化
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                    discretized_instance[f])]

        # 创建一个TableDomainMapper对象
        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)

        # 创建一个Explanation对象
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)

        # 选择要解释的标签
        # 分类任务
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]   # predict_proba用于在html页面显示原始分类结果
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]   # 按预测概率值从小到大排序后的索引，然后取最大的top索引
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()                # 从大到小排序
            # print('vae_lime_tabular522行labelsret_exp.top_labels的值为：', ret_exp.top_labels)
        # 回归任务
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        # 产生解释，返回截距、局部解释（特征以及特征的权值）、R2、局部预测值
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score,
             ret_exp.mae,
             ret_exp.local_pred,
             ret_exp.used_features) = self.base.explain_instance_with_data(scaled_data,
                                                                           yss,
                                                                           distances,
                                                                           label,
                                                                           num_features,
                                                                           model_regressor=model_regressor,
                                                                           feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    # 样本生成
    def data_generate(self,
                      data_row,
                      blackbox,
                      num_samples,
                      train,
                      labels_train):
        """
        函数功能：
            为解释模型生成训练数据，数据格式包括原始数据格式(与data_row数据格式一致)、离散化格式、二进制值格式

        参数：
             data_row：待解释实例--一维 numpy 数组
             num_samples：学习线性模型的领域大小--需要生成的数据数目
             data_generator：数据生成模型

        返回值：
            一个元组(data, binary_data),其中;
            data：与data_row格式相同的数据
            binary_data：二进制值格式数据，用于训练解释模型
        """
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
        else:
            num_cols = data_row.shape[0]

        categorical_features = range(num_cols)


        # 没有离散器
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]
            data = self.random_state.normal(
                0, 1, num_samples * num_cols).reshape(
                num_samples, num_cols)
            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples,
                                                 data_row.shape[1]),
                                                dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]))
            categorical_features = self.categorical_features
            first_row = data_row

        # 离散化待解释实例
        else:
            first_row = self.discretizer.discretize(data_row)

        # 1、用knn从训练集中找出待解释实例的10个样本
        # 训练KNN
        KNN = KNeighborsClassifier()
        KNN.fit(train, labels_train)

        # 找出data_row的10个邻居的索引
        k_10 = KNN.kneighbors(data_row.reshape(1, -1), 11, False)
        k = k_10[0]
        # print(k)

        # 定义一个一维数组，便于后面进行数组拼接
        k_samples = np.zeros(shape=(num_cols, ))

        # 找出训练集中data_row的邻居数据
        for i in k:
            k_samples = np.vstack((k_samples, train[i]))
        data_knn = k_samples[1:]
        # print(data_knn.shape)
        # print(labels_train[k].shape)

        data_knn = np.hstack([data_knn, labels_train[k].reshape(-1, 1)])
        # print(data_knn)
        #
        # exit(0)

        # 样本生成
        seed = 1
        # torch.manual_seed(seed)  # 为CPU设置随机种子
        # torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        # torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        np.random.seed(seed)
        for times in range(1, 100):
            isoMapVSG = IsomapVSG(data_knn.astype(np.float64), times)
            virtual_data = isoMapVSG.generateVirtualSamples(data_knn.shape[1]-1)  # 传入参数，属性个数
            if len(virtual_data) >= num_samples:
                break

        data = virtual_data

        # 将离散形特征取整
        if(self.categorical_features is not None):
            data[:, categorical_features] = np.around(data[:, categorical_features]).astype('int')

        # np.savetxt('../generate_data/wine_generated_data.csv', data, delimiter=',')



        # # 数据选择
        data = self.DataSelection(data_row, blackbox, data, 100)

        # 数据平衡
        # data = self.DataBalancing(blackbox, representive_samples)

        # 将生成的数据进行离散化
        data_discretized = self.discretizer.discretize(data)

        # 将离散化的样本转换成二进制值形式
        binary_data = np.zeros((data.shape[0], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                binary_data[i][j] = (data_discretized[i][j] == first_row[j]).astype(int)

        # 返回生成的样本和二进制值形式的样本
        data[0] = data_row
        binary_data[0] = np.ones(num_cols)

        # np.savetxt('../generate_data/hepatitis_lime_generated2_data.csv', data, delimiter=',')
        return data, binary_data

    def DataSelection(self, instance2explain, blackbox, dense_samples, tau):
        """
        This function accept generated compact data and select representative samples
        as candidate set for the instance2explain. In this way, created data points in
        the previous phase that are outlier are removed from sample set. This helps the
        interpretable model to rely on the samples in the close locality for explanation.

        instance2explain:待解释实例
        blackbox:黑盒模型
        dense_samples:数据
        tau:要选择的每类代表数据的数量
        """

        n_clusters = 2  # Number of clusters
        groups = list()  # Groups of data per class label
        preds = blackbox.predict(dense_samples)  # 获取标签信息
        labels = np.unique(preds)  # 去除数组中的重复数字，并进行排序之后输出
        # print("preds:", preds)
        # print(preds.shape)

        i = 0
        for l in labels:
            # Appending instance2explain to each class of data
            # print(dense_samples)
            # print(np.where(preds == l))
            # print(dense_samples[np.where(preds == l), :].shape)
            # print(dense_samples[np.where(preds == l), :])
            # print(np.squeeze(dense_samples[np.where(preds == l), :], axis=0).shape)
            groups.append(
                np.r_[instance2explain.reshape(1, -1), np.squeeze(dense_samples[np.where(preds == l), :], axis=0)])

            # print(groups)

            # Iterative data selection
            while True:
                clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(groups[i])
                # Collecting data points belong to the cluster of instance2explain
                indices = np.where(clustering.labels_ == clustering.labels_[0])
                c_instance2explain = np.squeeze(groups[i][indices, :], axis=0)
                # Checking the termination condition
                if c_instance2explain.shape[0] <= tau:
                    break
                else:
                    groups[i] = c_instance2explain

            i = i + 1
        # Merging the representative samples of every class
        representative_samples = np.concatenate([np.array(groups[j]) for j in range(len(groups))])
        return representative_samples

    def DataBalancing(self, blacbox, representative_samples):
        """
        The aim of this function is to handle potential class imbalance problem
        in the representative sample set. Having a balanced data set is necessary
        for creating a fair interpretable model. The output of this step is the
        final training data for the interpretable model.
        """

        # Applying SMOTE oversampling
        oversampler = SMOTE(random_state=42)
        os_samples, os_labels = oversampler.fit_resample(representative_samples,
                                                         blacbox.predict(representative_samples))
        # discrete_indices = dataset['discrete_indices']
        # balanced_samples[:, discrete_indices] = np.around(balanced_samples[:, discrete_indices])  # 四舍五入
        return os_samples











class RecurrentTabularExplainer(LimeTabularExplainer):
    """
    An explainer for keras-style recurrent neural networks, where the
    input shape is (n_samples, n_timesteps, n_features). This class
    just extends the LimeTabularExplainer class and reshapes the training
    data and feature names such that they become something like

    (val1_t1, val1_t2, val1_t3, ..., val2_t1, ..., valn_tn)

    Each of the methods that take data reshape it appropriately,
    so you can pass in the training/testing data exactly as you
    would to the recurrent neural network.

    """

    def __init__(self, training_data, mode="classification",
                 training_labels=None, feature_names=None,
                 categorical_features=None, categorical_names=None,
                 kernel_width=None, kernel=None, verbose=False, class_names=None,
                 feature_selection='auto', discretize_continuous=True,
                 discretizer='quartile', random_state=None):
        """
        Args:
            training_data: numpy 3d array with shape
                (n_samples, n_timesteps, n_features)
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True. Options
                are 'quartile', 'decile', 'entropy' or a BaseDiscretizer
                instance.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """

        # Reshape X
        n_samples, n_timesteps, n_features = training_data.shape
        training_data = np.transpose(training_data, axes=(0, 2, 1)).reshape(
                n_samples, n_timesteps * n_features)
        self.n_timesteps = n_timesteps
        self.n_features = n_features

        # Update the feature names
        feature_names = ['{}_t-{}'.format(n, n_timesteps - (i + 1))
                         for n in feature_names for i in range(n_timesteps)]

        # Send off the the super class to do its magic.
        super(RecurrentTabularExplainer, self).__init__(
                training_data,
                mode=mode,
                training_labels=training_labels,
                feature_names=feature_names,
                categorical_features=categorical_features,
                categorical_names=categorical_names,
                kernel_width=kernel_width,
                kernel=kernel,
                verbose=verbose,
                class_names=class_names,
                feature_selection=feature_selection,
                discretize_continuous=discretize_continuous,
                discretizer=discretizer,
                random_state=random_state)

    def _make_predict_proba(self, func):
        """
        The predict_proba method will expect 3d arrays, but we are reshaping
        them to 2D so that LIME works correctly. This wraps the function
        you give in explain_instance to first reshape the data to have
        the shape the the keras-style network expects.
        """

        def predict_proba(X):
            n_samples = X.shape[0]
            new_shape = (n_samples, self.n_features, self.n_timesteps)
            X = np.transpose(X.reshape(new_shape), axes=(0, 2, 1))
            return func(X)

        return predict_proba

    def explain_instance(self, data_row, classifier_fn, labels=(1,),
                         top_labels=None, num_features=10, num_samples=5000,
                         distance_metric='euclidean', model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 2d numpy array, corresponding to a row
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities. For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        # Flatten input so that the normal explainer can handle it
        data_row = data_row.T.reshape(self.n_timesteps * self.n_features)

        # Wrap the classifier to reshape input
        classifier_fn = self._make_predict_proba(classifier_fn)
        return super(RecurrentTabularExplainer, self).explain_instance(
            data_row, classifier_fn,
            labels=labels,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=num_samples,
            distance_metric=distance_metric,
            model_regressor=model_regressor)
