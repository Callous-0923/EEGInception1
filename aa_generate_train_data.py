import os
import pickle
import numpy as np
import scipy.io as scio
from scipy import signal
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing



def bandpass_filter(data, srate):
    fs = srate
    # 陷波滤波器
    Q = 35.0  # Quality factor
    f1 = 50
    b0, a0 = signal.iirnotch(f1, Q, fs)
    data = signal.filtfilt(b0, a0, data, axis=-1)
    # 巴特沃斯滤波
    order = 4
    f_1, f_2 = 1, 40
    wn = [f_1 * 2 / fs, f_2 * 2 / fs]
    b1, a1 = signal.butter(order, wn, 'bandpass')
    data = signal.filtfilt(b1, a1, data, axis=-1)
    return data


# 参数设置
srate = 2048   # 采样率
n_samples = 1*128
down_sample = 128
flash_num = 6
downsample_rate = srate//down_sample
# 选择通道
all_channels = ['Fp1', 'AF3', 'F7',  'F3',  'FC1', 'FC5', 'T7',  'C3',  'CP1', 'CP5',
                'P7',  'P3',  'Pz',  'PO3', 'O1',  'Oz',  'O2',  'PO4', 'P4',  'P8',
                'CP6', 'CP2', 'C4',  'T8',  'FC6', 'FC2', 'F4',  'F8',  'AF4', 'Fp2',
                'Fz',  'Cz',  'MA1', 'MA2']
reference_channels = ['T7', 'T8']
reference_ch_id = []
for channel in reference_channels:
    reference_ch_id.append(all_channels.index(channel))
# 读取数据
indir = r"D:\files\datasets\erp"
subs = os.listdir(indir)
for sub in subs:
    print(sub)
    sessions = os.listdir(os.path.join(indir, sub))
    ses_eeg = {'data':[], 'label':[], 'com':[]}
    for session in sessions:
        print(session)
        epochs = os.listdir(os.path.join(indir, sub, session))
        for epoch in epochs:
            print(epoch)
            eeg = scio.loadmat(os.path.join(indir, sub, session, epoch))
            data = eeg['data']
            events = eeg['events']
            target = np.squeeze(eeg['target'])
            stimuli = np.squeeze(eeg['stimuli'])
            # 重参考
            data = data - np.mean(data[reference_ch_id,:],0)
            # 选择通道
            data = data[:32, :]
            # 滤波
            data = bandpass_filter(data, srate)
            # 下采样至128Hz
            data = data[:, ::downsample_rate]
            # 提取trials
            n_trials = events.shape[0]
            repeat_num = 20
            # events增加一列
            events = np.concatenate((events, np.zeros((events.shape[0], 1))), axis=1)
            # 取秒的小数部分到最后一列，倒数第二列只保留秒的整数部分,最后全部取整
            events[:, -1] = 1000000 * (events[:, -2] - np.floor(events[:, -2]))  # 避免取整时毫秒级被忽略
            events[:, -2] = np.floor(events[:, -2])
            events = events.astype(int)
            t0 = datetime(events[0, 0], events[0, 1], events[0, 2], events[0, 3], events[0, 4], events[0, 5], events[0, 6])
            trials_data = np.zeros((repeat_num, flash_num, data.shape[0], n_samples))
            trials_label = np.zeros((repeat_num, flash_num)).astype(int)
            for repeat_num_i in range(repeat_num):
                for flash_i in range(flash_num):
                    i = repeat_num_i*flash_num + flash_i
                    flash_id = stimuli[i]
                    if flash_id == target:
                        trials_label[repeat_num_i, flash_id - 1] = 1
                    else:
                        trials_label[repeat_num_i, flash_id - 1] = 0
                    tn = datetime(events[i, 0], events[i, 1], events[i, 2], events[i, 3], events[i, 4], events[i, 5], events[i, 6])
                    pos = round((tn - t0).total_seconds()*down_sample + 0.4*down_sample)
                    # 分割数据
                    a_trial_data = data[:, pos:pos+n_samples]
                    # 标准化
                    a_trial_data = preprocessing.scale(a_trial_data, axis=1)

                    # print(f"[DEBUG] 单个试次数据形状 - 通道×时间点: {a_trial_data.shape}")  # 新增

                    # 集合
                    trials_data[repeat_num_i, flash_id - 1, :, :] = a_trial_data
            ses_eeg['data'].append(trials_data)
            ses_eeg['label'].append(trials_label)
            ses_eeg['com'].append(target+100)

            # print(f"[DEBUG] 当前session数据形状 - 重复次数×刺激数×通道×时间点: {trials_data.shape}")  # 新增

            # 输出目标与非目标数
            data = trials_data.reshape(-1, data.shape[0], n_samples)
            label = trials_label.reshape(-1)
            target_index = np.where(label==1)[0]
            nontarget_index = np.where(label==0)[0]
            n_target = len(target_index)
            n_nontarget = len(nontarget_index)
            # print(n_target, n_nontarget)
            # 画图
            tar = np.mean(np.mean(data[target_index,:,:], 0), 0)
            nontar = np.mean(np.mean(data[nontarget_index,:,:], 0), 0)
            plt.plot(tar)
            plt.plot(nontar)
            plt.show()
    data_by_flash = np.array(ses_eeg['data'])
    label_by_flash = np.array(ses_eeg['label'])
    com_trig = np.array(ses_eeg['com'])

    print(f"[DEBUG] 最终数据集形状 - 总试次数×刺激数×通道×时间点: {data_by_flash.shape}")  # 新增
    print(f"[DEBUG] 最终标签集形状 - 总试次数×刺激数×通道×时间点: {label_by_flash.shape}")

    eeg = {'data': data_by_flash, 'label':label_by_flash, 'com':com_trig}
    # print(f"[DEBUG] 输出数据 - {eeg.shape}")
    # 输出路径
    outdir = os.path.join(os.path.dirname(__file__), 'PreData')
    os.makedirs(outdir, exist_ok=True)  # 创建输出数据文件夹
    with open(os.path.join(outdir, '%s.pkl'%sub), 'wb') as f1:
        pickle.dump(eeg, f1)


