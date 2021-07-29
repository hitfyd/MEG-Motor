import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
criterion = nn.CrossEntropyLoss()
global_seed = -1
global_data_subjects = -1

def setup_seed(seed):
    global global_seed
    global_seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_subjects_info(subjects):
    global global_data_subjects
    global_data_subjects = subjects


def Z_score(X, intrvl):
    """Perform scaling based on pre-stimulus baseline"""
    X0 = X[:, :, :intrvl]
    X0 = X0.reshape([X.shape[0], -1])
    X -= X0.mean(-1)[:, None, None]
    X /= X0.std(-1)[:, None, None]
    X = X[:, :, intrvl:]
    return X


# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def train(model, device, train_loader, optimizer, epoch, classes=4):
    # 创建一个空矩阵存储混淆矩阵
    conf_matrix = torch.zeros(classes, classes)
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx + 1) % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f})]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       1. * batch_idx / len(train_loader), loss.item()))
        conf_matrix = confusion_matrix(pred, labels=target, conf_matrix=conf_matrix)
    accuracy = 100. * correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('Training Dataset\tEpoch：{}\tAccuracy: [{}/{} ({:.6f})]\tAverage Loss: {:.6f}'.format(
        epoch, correct, len(train_loader.dataset), 1. * correct / len(train_loader.dataset), loss))
    # print(conf_matrix)
    return accuracy, train_loss, conf_matrix


def test(model, device, test_loader, classes=4):
    # 创建一个空矩阵存储混淆矩阵
    conf_matrix = torch.zeros(classes, classes)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model(data)
            test_loss += criterion(output, target).item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
            conf_matrix = confusion_matrix(pred, labels=target, conf_matrix=conf_matrix)

    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('Test Dataset\tAccuracy: {}/{} ({:.6f}%)\tAverage loss: {:.6f}'.format(
        correct, len(test_loader.dataset), test_accuracy, test_loss))
    # print(conf_matrix)
    return test_accuracy, test_loss, conf_matrix


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def run(model, train_data, train_labels, test_data, test_labels, batch_size=256, epochs=200, classes=4, optimizer_type='ASGD', learn_rate=3e-3):
    model.to(DEVICE)

    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learn_rate, amsgrad=True)
    elif optimizer_type == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=learn_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    # 训练集
    train_data = torch.from_numpy(train_data)
    train_labels = torch.from_numpy(train_labels)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)

    # 测试集
    test_data = torch.from_numpy(test_data)
    test_labels = torch.from_numpy(test_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    test_accuracy_list, test_loss_list, train_accuracy_list, train_loss_list = [], [], [], []
    model.zero_grad()
    model.apply(weight_reset)
    best_test_accuracy, best_test_conf_matrix, best_state_dict = 0, [], {}
    for epoch in range(epochs):
        train_accuracy, train_loss, train_conf_matrix = train(model, DEVICE, train_loader, optimizer, epoch, classes)
        test_accuracy, test_loss, test_conf_matrix = test(model, DEVICE, test_loader, classes)

        if test_accuracy >= best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_conf_matrix = test_conf_matrix
            best_state_dict = model.state_dict()

        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_loss)
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss)

    print(best_test_accuracy, max(test_accuracy_list),
          [(i, test_loss_list[i]) for i, x in enumerate(test_accuracy_list) if x == max(test_accuracy_list)],
          min(test_loss_list), [i for i, x in enumerate(test_loss_list) if x == min(test_loss_list)])

    current_time = time.strftime("%Y%m%d%H%M%S")
    torch.save(best_state_dict, 'record/{}_checkpoint.pt'.format(current_time))
    record_file = "record/{}_record.txt".format(current_time)
    with open(record_file, "a+") as f:
        f.write('seed: {}\n'.format(global_seed))
        f.write('data subjects: {}\n'.format(global_data_subjects))
        f.write('batch_size: {}\n'.format(batch_size))
        f.write('train_epochs: {}\n'.format(epochs))
        f.write('model setup: {}\n'.format(model.get_model_info()))
        f.write('optimizer: {}\n'.format(optimizer))
        f.write('best_accuracy: {:.6f}%\n'.format(best_test_accuracy))
        f.write('best_test_conf_matrix: {}\n'.format(best_test_conf_matrix.data))
        f.write('{}\n'.format(train_accuracy_list))
        f.write('{}\n'.format(train_loss_list))
        f.write('{}\n'.format(test_accuracy_list))
        f.write('{}\n'.format(test_loss_list))
