import mne
import numpy as np
from glob import glob
from multiprocessing import Pool

# 操作数据的目录读取动作，数据目录：HCP_raw_paths，tfrecord存储目录：savepath
import util

data_path = '/media/hit/1/HCP/'
fileName = glob(data_path+'*')
raw_session10 = '/unprocessed/MEG/10-Motort/4D/c,rfDC'
raw_session11 = '/unprocessed/MEG/11-Motort/4D/c,rfDC'
HCP_session10_raw_paths = [file + raw_session10 for file in fileName]
HCP_session11_raw_paths = [file + raw_session11 for file in fileName]
save_path = '/media/hit/1/HCP_epochs/'


def preprocess(path):
    # preload=True一定要为true，否则会很慢
    # t subject number and session number
    subject = path[17:23]
    session = path[40:42]
    raw = mne.io.read_raw_bti(path, head_shape_fname=None, preload=True)
    events = mne.find_events(raw, min_duration=0.001, output='onset')
    events = mne.pick_events(events, include=[22, 134, 70, 38])
    # 'leftHand': 22, 'rightHand': 70, 'leftFoot': 38, 'rightFoot': 134
    events = mne.merge_events(events, [22], 0)
    events = mne.merge_events(events, [134], 1)
    events = mne.merge_events(events, [70], 2)
    events = mne.merge_events(events, [38], 3)

    picks = mne.pick_types(raw.info, meg='mag')
    fmin = 1.
    fmax = 45.
    raw = raw.filter(l_freq=fmin, h_freq=fmax)
    # tmin tmax decim共同决定了epochs.shape(X,Y,Z).在camcan数据集中采样率1kHz，所以-.3,.5,8.正好对应1000,该数据集采样率不规整，所以不能精准到100.计算：sfreq：2034.5/decim6*（tmax-tmin)0.6
    epochs = mne.epochs.Epochs(raw, events, tmin=-.2, tmax=.8, decim=8, detrend=1,
                               picks=picks,
                               preload=True)
    del raw
    data = epochs.get_data()

    data = util.Z_score(data, 51)# 降采样后采样率为254Hz，前0.2s为51个采样点
    print(data.shape)
    # labels必须对应上0 1 2 3依次，要不训练就会出线loss为0的情况
    labels = epochs.events[:, 2]

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.longlong)
    np.savez(save_path + subject + '_session' + session + '.npz', data=data, labels=labels)
    del epochs, data, labels


if __name__ == '__main__':
    # for path in HCP_session10_raw_paths:
    #     preprocess(path)
    # for path in HCP_session11_raw_paths:
    #     preprocess(path)

    HCP_session10_raw_paths.extend(HCP_session11_raw_paths)
    # 多线程处理
    pool = Pool(10)
    pool.map(preprocess, HCP_session10_raw_paths)
    pool.close()
    pool.join()
