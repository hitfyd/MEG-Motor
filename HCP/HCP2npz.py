import mne
import numpy as np
from glob import glob
from multiprocessing import Pool

import util

# 操作数据的目录读取动作，数据目录：HCP_raw_paths，tfrecord存储目录：savepath
data_path = '/media/hit/1/HCP/'
fileName = glob(data_path+'*')
raw_session10 = '/unprocessed/MEG/10-Motort/4D/c,rfDC'
raw_session11 = '/unprocessed/MEG/11-Motort/4D/c,rfDC'
HCP_session10_raw_paths = [file + raw_session10 for file in fileName]
HCP_session11_raw_paths = [file + raw_session11 for file in fileName]
save_path = '/media/hit/1/HCP_epochs/'


def preprocess(path):
    subject = path[17:23]   # 获取受试者编号
    session = path[40:42]   # 获取session编号

    # 读取数据
    raw = mne.io.read_raw_bti(path, head_shape_fname=None, preload=True)    # preload=True一定要为true，否则会很慢
    events = mne.find_events(raw, min_duration=0.001, output='onset')
    events = mne.pick_events(events, include=[22, 134, 70, 38])
    # 'leftHand': 22, 'rightHand': 70, 'leftFoot': 38, 'rightFoot': 134
    # 修改事件编号
    events = mne.merge_events(events, [22], 0)
    events = mne.merge_events(events, [134], 1)
    events = mne.merge_events(events, [70], 2)
    events = mne.merge_events(events, [38], 3)

    picks = mne.pick_types(raw.info, meg='mag')

    # 滤波
    fmin = 1.
    fmax = 45.
    raw = raw.filter(l_freq=fmin, h_freq=fmax)
    # 提取Epoch，分割区间为[-0.2s， 0.8s]，降采样到八分之一
    epochs = mne.epochs.Epochs(raw, events, tmin=-.2, tmax=.8, decim=8, detrend=1, picks=picks, preload=True)
    del raw
    data = epochs.get_data()

    data = util.z_score_standardization(data, 51)   # 降采样后采样率为254(=2034.5/8)Hz，前0.2s为51个采样点
    print(data.shape)
    # labels必须对应上0 1 2 3依次，要不训练就会出线loss为0的情况
    labels = epochs.events[:, 2]

    # 转换数据类型与torch默认类型一致
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.longlong)
    np.savez(save_path + subject + '_session' + session + '.npz', data=data, labels=labels)
    del epochs, data, labels


if __name__ == '__main__':
    # for path in HCP_session10_raw_paths:
    #     preprocess(path)
    # for path in HCP_session11_raw_paths:
    #     preprocess(path)

    # 合并两个session的文件存储路径
    HCP_session10_raw_paths.extend(HCP_session11_raw_paths)
    # 多线程处理
    pool = Pool(10)
    pool.map(preprocess, HCP_session10_raw_paths)
    pool.close()
    pool.join()
