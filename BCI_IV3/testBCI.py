import numpy as np

import CNN
import util

# 设置随机数种子
# S1最佳随机数种子为4000，S2最佳随机数种子为1300
# 先VAR后LF，S1：1000，S2:1000
util.setup_seed(1000)

BATCH_SIZE = 64
EPOCHS = 2000  # 总共训练批次
# BCI数据集subjects包括1、2，推荐使用subject1
subject = 2
util.set_subjects_info(subject)

model = CNN.CNN(channels=10, points=240, classes=4, spatial_sources=32, dropout_factor=0.75)

# 读取原始训练数据
npz = np.load(str(subject) + ".npz")
train_data, train_labels, test_data, test_labels = npz['train_data'], npz['train_labels'], npz['test_data'], npz['test_labels']

util.run(model, train_data, train_labels, test_data, test_labels, BATCH_SIZE, EPOCHS)
