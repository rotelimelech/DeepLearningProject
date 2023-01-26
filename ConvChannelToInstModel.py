import torch
import torch.nn as nn
from torchvision import models, transforms

DATA_TIME_LEN = 201
DATA_FREQ_LEN = 52

class ConvChannelToInstModel(torch.nn.Module):
    def __init__(self, n_instruments, n_notes):
        super(ConvChannelToInstModel, self).__init__()

        self.n_instruments = n_instruments
        self.n_notes = n_notes

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=n_instruments,
            kernel_size=(3, 3))
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=n_instruments, out_channels=n_instruments,
            kernel_size=(3, 3),
            groups=n_instruments)
        self.relu2 = nn.ReLU(inplace=True)

        size_after_convs = (DATA_TIME_LEN-4) * (DATA_FREQ_LEN-4)

        self.fcs = [nn.Linear(in_features=size_after_convs, out_features=n_notes) 
            for _ in range(n_instruments)]

        self.sig = nn.Sigmoid()
        

    def forward(self, data):
        batch_size = data.shape[0]
        data = self.conv1(data)
        data = self.relu1(data)

        data = self.conv2(data)
        data = self.relu2(data)

        data = data.reshape(batch_size, self.n_instruments, -1)

        output = torch.zeros(batch_size, self.n_instruments, self.n_notes)
        for b in range(batch_size):
            for i in range(self.n_instruments):
                output[b,i,:] = self.fcs[i](data[b,i,:])

        output = self.sig(output)
        return output

