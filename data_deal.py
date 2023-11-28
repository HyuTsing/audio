from matplotlib import pyplot as plt
from torch.utils.data.dataloader import default_collate

import data_prepare
import os
from typing import Callable, Tuple, Dict, Optional
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

dict={'backward':0,'bed':1,'bird':2,'cat':3,'dog':4,'down':5,'eight':6,'five':7,'follow':8,'forward':9,'four':10,
      'go':11,'happy':12,'house':13,'learn':14,'left':15,'marvin':16,'nine':17,'no':18,'off':19,'on':20,
      'one':21,'right':22,'seven':23,'sheila':24,'six':25,'stop':26,'three':27,'tree':28,'two':29,'up':30,
      'visual':31,'wow':32,'yes':33,'zero':34,'slient':35}


train_dataset=data_prepare.SPEECHCOMMANDS(label_dict=dict,root='F:/',download=False,folder_in_archive='MFCC')
test_dataset=data_prepare.SPEECHCOMMANDS(label_dict=dict,root='F:/',split='test',download=False,folder_in_archive='MFCC')
# train_dataset=data_prepare.SPEECHCOMMANDS(label_dict=dict,root='/mnt/disk2/WangShuai/speak/dataset/v2',download=True)
# test_dataset=data_prepare.SPEECHCOMMANDS(label_dict=dict,root='/mnt/disk2/WangShuai/speak/dataset/v2',split='test',download=True)

print(len(train_dataset))
print(len(test_dataset))

wave,lab=train_dataset[3]
print(wave,lab)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
fig, axes = plt.subplots(2, 2)
data = pd.Series(wave[0,:], index=time)
data.plot.bar(ax=axes[1, 1], color='b', alpha=0.5)
data.plot.barh(ax=axes[0, 1], color='k', alpha=0.5)
plt.show()
'''
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
'''
for data in train_loader:
    wave,lab=data
    print(wave.shape)
    print(lab.shape)

'''