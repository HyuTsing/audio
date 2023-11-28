import torchaudio
import os
import torch
from data_prepare import transform_asf,transforms
import numpy as np
file_dir = "F:\speech_commands_v0.02"
def read_files(folder_path):
    file_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(root)
            file_dict[file_path] = folder_name
    return file_dict


transform = transforms.MFCC(
    sample_rate=16000,  # 音频的采样率
    n_mfcc=46,  # 要提取的MFCC系数数量
    melkwargs={'n_fft': 512, 'hop_length': 100, 'n_mels': 68}  # Mel滤波器参数
)
# 示例用法

file_dict = read_files(file_dir)

for file_path, folder_name in file_dict.items():
    print("File:", file_path)
    print("Folder:", folder_name)
    print("-------------------")
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    filename=name+'.npy'
    new_folder_path = os.path.join('F:\MFCC', folder_name)
    new_file=os.path.join('F:\MFCC',folder_name,filename)

    os.makedirs(new_folder_path, exist_ok=True)
    print(new_folder_path)
    waveform,sr=torchaudio.load(file_path)
    m = waveform.abs().max()
    # print(m)
    if m > 0:
        waveform /= m
    a = len(waveform[0])

    b =(16000-a)if ((16000 - a))>0 else 0
    c = torch.zeros(1, b)
    waveform1 = torch.cat((waveform, c), dim=1)
    # print(waveform1.shape)
    mfcc = transform(waveform1)
    # 打印MFCC特征的形状--------------------
    print("MFCC shape:", mfcc.shape)
    mfcc=mfcc.reshape(1,46,161)
    np.save(new_file, mfcc)

    # waveform1 = waveform1.squeeze(dim=0)
    # waveform, _ = transform_asf(waveform1, thr=0.01, alpha=1.5)
    # print(waveform,waveform.shape)
    # t = np.linspace(0, 1, 16000)
    # plt.plot(t,waveform)
    # plt.show()
    # print(waveform.shape)
    # waveform = waveform.reshape(1, 16000)
    # torchaudio.save(new_file, waveform, sr)

