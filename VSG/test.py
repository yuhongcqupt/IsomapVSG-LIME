from VSG import ELM
import pandas as pd
import numpy as np

from VSG.IsomapVSG import IsomapVSG

np.random.seed(1)

# 加载数据
data = pd.read_csv('../data/parkinsons.csv')
# 包含标签的数据
data = np.array(data.iloc[0:20, :])

# print(data)

# 样本生成
# 因为要指定除以最小长度的倍数，这个值是不确定的，因此就在这里加一层循环吧
for times in range(1, 100):
    isoMapVSG = IsomapVSG(data.astype(np.float64), times)
    virtual_data = isoMapVSG.generateVirtualSamples(22)  # 传入参数，属性个数
    if len(virtual_data) >= 1000:
        break

# print(virtual_data.shape)
virtual_samples = virtual_data[:1000]

# 保存数据
np.savetxt('generate_data.csv', virtual_samples)
