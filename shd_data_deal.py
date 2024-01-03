import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils import data
torch.manual_seed(2)

train_X = np.load('F:/PycharmProjects/shdtest/data/trainX_4ms.npy')
train_y = np.load('F:/PycharmProjects/shdtest/data/trainY_4ms.npy').astype(float)

test_X = np.load('F:/PycharmProjects/shdtest/data/testX_4ms.npy')
test_y = np.load('F:/PycharmProjects/shdtest/data/testY_4ms.npy').astype(float)

print('dataset shape: ', train_X.shape)
print('dataset shape: ', test_X.shape)

batch_size = 16

tensor_trainX = torch.Tensor(train_X)  # transform to torch tensor
tensor_trainY = torch.Tensor(train_y)
train_dataset = data.TensorDataset(tensor_trainX, tensor_trainY)
shd_train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
tensor_testX = torch.Tensor(test_X)  # transform to torch tensor
tensor_testY = torch.Tensor(test_y)
test_dataset = data.TensorDataset(tensor_testX, tensor_testY)
shd_test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
