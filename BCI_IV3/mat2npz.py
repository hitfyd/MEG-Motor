import numpy as np
import scipy.io as scio

S1_data_path = "D:/BCI Competition IV Datasets 3/S1.mat"
S1_label_path = "D:/BCI Competition IV Datasets 3/TrueLabelsS1.mat"
S2_data_path = "D:/BCI Competition IV Datasets 3/S2.mat"
S2_label_path = "D:/BCI Competition IV Datasets 3/TrueLabelsS2.mat"


def scale_type(X, intrvl):
    """Perform scaling based on pre-stimulus baseline"""
    X0 = X[:, :, :intrvl]
    X0 = X0.reshape([X.shape[0], -1])
    X -= X0.mean(-1)[:, None, None]
    X /= X0.std(-1)[:, None, None]
    # X = X[:, :, intrvl:]
    return X


def mat2npy(subject):
    """
    :param subject:可选1、2
    """
    # 提取训练集
    if subject == 1:
        data_mat = scio.loadmat(S1_data_path)
        label_mat = scio.loadmat(S1_label_path)
        test_labels = label_mat.get('TrueLabelsS1')[0]
    elif subject == 2:
        data_mat = scio.loadmat(S2_data_path)
        label_mat = scio.loadmat(S2_label_path)
        test_labels = label_mat.get('TrueLabelsS2')[0]
    else:
        print('subject error!')
        return
    training_data = data_mat.get('training_data')[0]
    data, labels = [], []
    for i in range(training_data.size):
        label_data = training_data[i]
        for j in range(label_data.shape[0]):
            data.append(label_data[j])
            labels.append(i)
    # 提取测试集
    test_data = data_mat.get('test_data')

    assert test_data.shape[0] == test_labels.size

    for j in range(test_data.shape[0]):
        test_labels[j] -= 1  # 将标签从1、2、3、4调整为0、1、2、3

    # 数据类型转换到float32，与torch默认类型一致
    data = np.array(data, dtype=np.float32)
    data = data.transpose([0, 2, 1])  # 转换维度，与HCP数据集一致
    labels = np.array(labels, dtype=np.longlong)
    test_data = test_data.astype(np.float32)
    test_data = test_data.transpose([0, 2, 1])  # 转换维度，与HCP数据集一致
    test_labels = test_labels.astype(np.longlong)

    data = scale_type(data, 160)
    test_data = scale_type(test_data, 160)

    np.savez(str(subject) + ".npz", train_data=data, train_labels=labels, test_data=test_data, test_labels=test_labels)


mat2npy(subject=1)
mat2npy(subject=2)
