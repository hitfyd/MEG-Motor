from glob import glob

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

import CNN
import util

# 设置随机数种子
util.setup_seed(1000)

BATCH_SIZE = 1024
EPOCHS = 5000  # 总共训练批次，由于60人数据集共包含38400个样本，训练时间较长

model = CNN.CNN(channels=248, points=203, classes=4, spatial_sources=128)

# 测试不同受试者规模下模型的性能，按照受试者编号依次选择10、20、30、40、50、60名受试者组成数据集，第61名受试者始终未使用
for number_subjects in [10, 20, 30, 40, 50, 60]:
    data_path = 'D:/HCP_epochs/'
    npzFiles = glob(data_path+'*.npz')[:number_subjects*2]  # 每名受试者参与了2次session
    util.set_subjects_info(npzFiles)    # 设置受试者信息
    data, labels = None, None
    # 读取npz
    for file in npzFiles:
        npz = np.load(file)
        if len(npz['data']) > 0:
            if data is None:
                data, labels = npz['data'], npz['labels']
            else:
                data = np.concatenate((data, npz['data']))
                labels = np.concatenate((labels, npz['labels']))

    # 'leftHand': 0, 'rightFoot': 1, 'rightHand': 2, 'leftFoot': 3
    # 合并手脚或者左右标签
    for i in range(len(labels)):
        if labels[i] == 0 or labels[i] == 2:
            labels[i] = 0
        else:
            labels[i] = 1

    # 使用交叉验证函数完成训练集、测试集随机划分，比例3:1
    skf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1)
    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        train_data, test_data = data[train_idx], data[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]
        util.run(model, train_data, train_labels, test_data, test_labels, BATCH_SIZE, EPOCHS)
        # 只测试一次，不进行完整的交叉验证
        break
