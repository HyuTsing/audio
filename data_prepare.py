import os
from typing import Callable, Tuple, Dict, Optional, Any
from pathlib import Path

import librosa
import torch
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
import matplotlib.pyplot as plt

from torch.hub import download_url_to_file
from torchaudio.datasets.utils import (

    _extract_tar
)
from torchvision.datasets.utils import verify_str_arg
import numpy as np
from random import choice
import torchaudio
import torchaudio.transforms as transforms
import torch

sr = 16000


# 加载音频数据（假设waveform是一个Tensor表示的音频波形）

# 定义编码转换器
def asf_encoding(waveform, alpha, thr):
    asf = torch.zeros_like(waveform)
    base = waveform[0]
    asf_count = 0
    for j in range(len(waveform)):
        if waveform[j] >= base + thr:
            asf[j] = 1
            # base = waveform[j] * rate
            base += thr
            thr *= alpha
            asf_count += 1
        if waveform[j] <= base - thr:
            asf[j] = -1
            # base = waveform[j] / rate
            base -= thr
            thr *= alpha
            asf_count += 1
        if j >= 1 and asf[j] == 0 and thr >= 0.01:
            thr = thr / alpha
    sparse = 1 - asf_count / len(waveform)
    return asf, sparse


transform_asf = asf_encoding
# 定义MFCC转换器
transform = transforms.MFCC(
    sample_rate=sr,  # 音频的采样率
    n_mfcc=46,  # 要提取的MFCC系数数量
    melkwargs={'n_fft': 512, 'hop_length': 100, 'n_mels': 68}  # Mel滤波器参数
)

# 将音频波形转换为MFCC特

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
_CHECKSUMS = {
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz":
        "3cd23799cb2bbdec517f1cc028f8d43c",
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz":
        "6b74f3901214cb2c2934e98196829835",
}
VAL_RECORD = "validation_list.txt"
TEST_RECORD = "testing_list.txt"
TRAIN_RECORD = "training_list.txt"


def load_speechcommands_item(relpath: str, path: str) -> tuple[Any, str]:
    filepath = os.path.join(path, relpath)
    label, filename = os.path.split(relpath)
    speaker, _ = os.path.splitext(filename)
    # print(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    # waveform, sample_rate = torchaudio.load(filepath)
    #
    # return waveform, sample_rate, label, speaker_id, utterance_number
    mfcc_features = np.load(filepath)

    return mfcc_features, label


class SPEECHCOMMANDS(Dataset):
    def __init__(self,
                 label_dict: Dict,
                 root: str,
                 silence_cnt: Optional[int] = 0,
                 silence_size: Optional[int] = 16000,
                 transform: Optional[Callable] = None,
                 url: Optional[str] = URL,
                 split: Optional[str] = "train",
                 folder_in_archive: Optional[str] = FOLDER_IN_ARCHIVE,
                 download: Optional[bool] = False) -> None:
        '''
        :param label_dict: 标签与类别的对应字典
        :type label_dict: Dict
        :param root: 数据集的根目录
        :type root: str
        :param silence_cnt: Silence数据的数量
        :type silence_cnt: int, optional
        :param silence_size: Silence数据的尺寸
        :type silence_size: int, optional
        :param transform: A function/transform that takes in a raw audio
        :type transform: Callable, optional
        :param url: 数据集版本，默认为v0.02
        :type url: str, optional
        :param split: 数据集划分，可以是 ``"train", "test", "val"``，默认为 ``"train"``
        :type split: str, optional
        :param folder_in_archive: 解压后的目录名称，默认为 ``"SpeechCommands"``
        :type folder_in_archive: str, optional
        :param download: 是否下载数据，默认为False
        :type download: bool, optional

        SpeechCommands语音数据集，出自 `Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition <https://arxiv.org/abs/1804.03209>`_，根据给出的测试集与验证集列表进行了划分，包含v0.01与v0.02两个版本。

        数据集包含三大类单词的音频：

        #. 指令单词，共10个，"Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go". 对于v0.02，还额外增加了5个："Forward", "Backward", "Follow", "Learn", "Visual".

        #. 0~9的数字，共10个："One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine".

        #. 辅助词，可以视为干扰词，共10个："Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", "Wow".

        v0.01版本包含共计30类，64,727个音频片段，v0.02版本包含共计35类，105,829个音频片段。更详细的介绍参见前述论文，以及数据集的README。

        代码实现基于torchaudio并扩充了功能，同时也参考了 `原论文的实现 <https://github.com/romainzimmer/s2net/blob/b073f755e70966ef133bbcd4a8f0343354f5edcd/data.py>`_。
        '''

        self.split = verify_str_arg(split, "split", ("train", "val", "test"))
        self.label_dict = label_dict
        self.transform = transform
        self.silence_cnt = silence_cnt
        self.silence_size = silence_size

        if silence_cnt < 0:
            raise ValueError(f"Invalid silence_cnt parameter: {silence_cnt}")
        if silence_size <= 0:
            raise ValueError(f"Invalid silence_size parameter: {silence_size}")

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)
        self._path = self._path.replace("/", os.path.sep)
        self._path = self._path.replace("\\", os.path.sep)
        self.noise_list = sorted(str(p) for p in Path(self._path).glob('_background_noise_/*.wav'))
        print(self._path)
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, root, hash_prefix=checksum)
                    _extract_tar(archive, self._path)
        elif not os.path.isdir(self._path):
            print(self._path)
            raise FileNotFoundError("Audio data not found. Please specify \"download=True\" and try again.")

        if self.split == "train":
            record = os.path.join(self._path, TRAIN_RECORD)
            print(record)
            if os.path.exists(record):
                with open(record, 'r') as f:
                    self._walker = list([line.rstrip('\n') for line in f])
                # print(self._walker)
            else:
                print("No training list, generating...")
                walker = sorted(str(p) for p in Path(self._path).glob('*/*.wav'))
                walker = filter(lambda w: HASH_DIVIDER in w and EXCEPT_FOLDER not in w, walker)
                walker = map(lambda w: os.path.relpath(w, self._path), walker)

                walker = set(walker)

                val_record = os.path.join(self._path, VAL_RECORD)
                with open(val_record, 'r') as f:
                    val_walker = set([line.rstrip('\n') for line in f])

                test_record = os.path.join(self._path, TEST_RECORD)
                with open(test_record, 'r') as f:
                    test_walker = set([line.rstrip('\n') for line in f])

                walker = walker - val_walker - test_walker
                self._walker = list(walker)

                with open(record, 'w') as f:
                    f.write('\n'.join(self._walker))

                print("Training list generated!")

            labels = [self.label_dict.get(os.path.split(relpath)[0]) for relpath in self._walker]
            # print(labels)
            label_weights = 1. / np.unique(labels, return_counts=True)[1]
            # print(label_weights)
            if self.silence_cnt == 0:
                label_weights /= np.sum(label_weights)
                self.weights = torch.DoubleTensor([label_weights[label] for label in labels])
                # print(self.weights)
            else:
                silence_weight = 1. / self.silence_cnt
                total_weight = np.sum(label_weights) + silence_weight
                label_weights /= total_weight
                self.weights = torch.DoubleTensor(
                    [label_weights[label] for label in labels] + [silence_weight / total_weight] * self.silence_cnt)

        else:
            if self.split == "val":
                record = os.path.join(self._path, VAL_RECORD)
            else:
                record = os.path.join(self._path, TEST_RECORD)
            with open(record, 'r') as f:
                self._walker = list([line.rstrip('\n') for line in f])

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        if n < len(self._walker):
            fileid = self._walker[n]
            # waveform, sample_rate, label, speaker_id, utterance_number = load_speechcommands_item(fileid, self._path)
            mfcc_features, label = load_speechcommands_item(fileid, self._path)
            # print(waveform.shape)
        else:
            # Silence data are randomly and dynamically generated from noise data

            # Load random noise
            noisepath = os.path.join(self._path, choice(self.noise_list))
            waveform, sample_rate = torchaudio.load(noisepath)

            # Random crop
            offset = np.random.randint(waveform.shape[1] - self.silence_size)
            waveform = waveform[:, offset:offset + self.silence_size]
            label = "_silence_"

        # m = waveform.abs().max()
        # # print(m)
        # if m > 0:
        #     waveform /= m
        # # print(waveform.shape)
        # a = len(waveform[0])
        #
        # b = 16000 - a
        # c = torch.zeros(1, b)
        # waveform1 = torch.cat((waveform, c), dim=1)
        # print(waveform1.shape)
        # mfcc = transform(waveform1)
        # 打印MFCC特征的形状
        # print("MFCC shape:", mfcc.shape)
        # waveform1=waveform1.squeeze(dim=0)
        # waveform, _ = transform_asf(waveform1, thr=0.01, alpha=1.5)
        # print(waveform,waveform.shape)
        # t = np.linspace(0, 1, 16000)
        # plt.plot(t,waveform)
        # plt.show()
        # print(waveform.shape)
        # waveform=waveform1.reshape(1,16000)

        # waveform = mfcc.reshape(1, 1, 46, 161)
        waveform = mfcc_features.reshape(1, 46, 161)

        label = self.label_dict.get(label)
        return waveform, label

    def __len__(self) -> int:
        return len(self._walker) + self.silence_cnt
