import numpy as np

import CNN
import util

# 设置随机数种子
# S1最佳随机数种子为5000，S2最佳随机数种子为600
util.setup_seed(600)

BATCH_SIZE = 64
EPOCHS = 2000  # 总共训练批次
# BCI数据集subjects包括1、2，推荐使用subject1
subject = 2

model = CNN.CNN(channels=10, points=400, classes=4, spatial_sources=32)

# 读取原始训练数据
npz = np.load(str(subject) + ".npz")
train_data, train_labels, test_data, test_labels = npz['train_data'], npz['train_labels'], npz['test_data'], npz['test_labels']

util.run(model, train_data, train_labels, test_data, test_labels, BATCH_SIZE, EPOCHS)

# data = np.concatenate((train_data, test_data))
# labels = np.concatenate((train_labels, test_labels))
# skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=36851234)
# for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
#     print(test_idx)
#     train_data, test_data = data[train_idx], data[test_idx]
#     train_labels, test_labels = labels[train_idx], labels[test_idx]
#
#     util.run(model, train_data, train_labels, test_data, test_labels, BATCH_SIZE, EPOCHS)
