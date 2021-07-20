import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    # 输入数据大小为 batch * channels * points
    def __init__(self, channels=204, points=100, classes=2, spatial_sources=32, pool_factor=2, dropout_factor=0.75,
                 active_func=nn.ELU()):
        super().__init__()
        self.channels, self.times, self.classes = channels, points, classes
        self.spatial_sources, self.pool_factor, self.dropout_factor = spatial_sources, pool_factor, dropout_factor
        # 激活函数
        self.active_func = active_func
        # 空间滤波器 1
        self.Spatial = nn.Linear(channels, spatial_sources)
        self.Spatial2 = nn.Linear(spatial_sources, spatial_sources)
        # 时间滤波器 2
        self.Temporal_VAR = nn.Conv1d(spatial_sources, spatial_sources, (7,), padding=(3,))  # 输入通道数32，输出通道数32，核的大小(7, 32)，填充(3, 0), 通道数等于潜在空间源数
        self.Temporal_LF = nn.Conv2d(spatial_sources, spatial_sources, (1, 7), padding=(0, 3), groups=spatial_sources)
        # print(self.Temporal_VAR.weight.shape, self.Temporal_VAR.bias.shape)
        # 池化层
        self.pool = nn.MaxPool2d((1, pool_factor), (1, pool_factor))  # 时间维度上池化因子为2，步长为2
        # 脱落层
        self.dropout = nn.Dropout(p=dropout_factor)
        # 输出层 3
        self.FC = nn.Linear(spatial_sources * int(points / 2), 1024)
        self.output = nn.Linear(1024, classes)

    def forward(self, x):
        batch_size = x.size(0)  # 输入数据大小为 batch * channels * points
        # print(x.shape, batch_size)
        # x = torch.squeeze(x)

        x = torch.transpose(x, 1, 2)  # 交换数据维度 ->batch * points * channels
        # print(x.shape)
        x = self.Spatial(x)  # 空间滤波器 ->batch * points * spatial_sources
        # print(x.shape)
        # x = self.Spatial2(x)

        x = torch.transpose(x, 1, 2)  # 交换数据维度 ->batch * spatial_sources * points
        # print(x.shape)
        x = torch.unsqueeze(x, -2)   # 增加数据维度 ->batch * spatial_sources * 1 * points
        # print(x.shape)
        x = self.Temporal_LF(x)    # 时间滤波器 ->batch * spatial_sources * 1 * points
        x = self.active_func(x)
        # print(x.shape)

        x = torch.squeeze(x)       # 降低数据维度 ->batch * spatial_sources * points
        x = self.Temporal_VAR(x)  # 时间滤波器 ->batch * spatial_sources * points
        x = self.active_func(x)
        # print(x.shape)

        x = torch.unsqueeze(x, -3)   # 增加数据维度 ->batch * 1 * spatial_sources * points
        x = self.pool(x)    # 时间维度上池化 ->batch * 1 * spatial_sources * (points / pool_factor)
        # print(x.shape)

        x = x.view(batch_size, -1)  # 扁平化 ->batch * (1 * spatial_sources * (points / pool_factor))
        # print(x.shape)

        x = self.dropout(x)     # 随机脱落 ->batch * (1 * spatial_sources * (points / pool_factor))
        # print(x.shape)

        x = self.FC(x)
        x = self.dropout(x)
        x = self.output(x)
        # print(x.shape, x)
        x = F.log_softmax(x, dim=1)  # 计算log(softmax(x))
        return x
