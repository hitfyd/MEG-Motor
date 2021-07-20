from glob import glob

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

import CNN
import util

# 设置随机数种子
util.setup_seed(1000)

BATCH_SIZE = 1024
EPOCHS = 2000  # 总共训练批次

model = CNN.CNN(channels=248, points=203, classes=4, spatial_sources=128)

data_path = 'D:/HCP_epochs/'
npzFiles = glob(data_path+'*10.npz')[:20]
data, labels = None, None
# 读取剩余npz
for file in npzFiles:
    npz = np.load(file)
    if len(npz['data']) > 0:
        if data is None:
            data, labels = npz['data'], npz['labels']
        else:
            data = np.concatenate((data, npz['data']))
            labels = np.concatenate((labels, npz['labels']))

# 'leftHand': 0, 'rightFoot': 1, 'rightHand': 2, 'leftFoot': 3
# 合并手脚或者左右
# for i in range(len(labels)):
#     if labels[i] == 0 or labels[i] == 2:
#         labels[i] = 0
#     else:
#         labels[i] = 1

skf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)
for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    util.run(model, train_data, train_labels, test_data, test_labels, BATCH_SIZE, EPOCHS)
    # 只测试一次，不进行完整的交叉验证
    break
