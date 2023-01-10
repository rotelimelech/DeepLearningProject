import torch
import torch.nn as nn

#from MusicNet import MusicNet
from MusicNetManyhotNotes import MusicNet
from torchaudio import transforms
from torchvision import models

BATCH_SIZE = 10

"""
Train the model using MusicNet dataset
"""

def main():
    dataset = MusicNet(
        './MusicNet',
        # groups='train', # something broken in loading only one group
        transform=transforms.Spectrogram()
    )

    train_loader =  torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # The other model is broken :(
    model = models.resnet18(weights='DEFAULT')
    model = nn.Sequential(
        model,
        # to the correct output size
        nn.Linear(1000, dataset.n_instruments),
        # manyhot 
        nn.Sigmoid()
        )

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.001)

    for batch_data, instrument, note in train_loader:
        # resnet excepts three channels of input
        batch_multi_channel = batch_data.repeat(1,3,1,1)
        
        # train
        output = model.forward(batch_multi_channel)
        loss = loss_fn(output, instrument.float())
        opt.zero_grad()
        loss.backward()
        opt.step()

    # might need to add 
    # to make resnet use only one channel


if __name__ == '__main__':
    main()