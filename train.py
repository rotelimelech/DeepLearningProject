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

def get_test_score(test_loader, model):
    correct = 0
    with torch.no_grad():
        for batch_id, (batch_data, instrument, note) in enumerate(train_loader):
            batch_multi_channel = batch_data.repeat(1,3,1,1)
            output = model(batch_multi_channel)
            prediction = output.argmax(dim=1, keepdim=True)
            instrument = instrument.argmax(dim=1, keepdim=True)
            correct += prediction.eq(instrument.view_as(prediction)).sum().item()

    n_test_samples = len(test_loader.dataset)
    return correct / n_test_samples

def main():
    train_dataset = MusicNet(
        '.\\MusicNet',
        # metadata_path='./MusicNet/all_metadata_processed_150123.csv',
        metadata_path='./MusicNet/all_local_metadata_150123.csv',
        indexes_paths='./MusicNet/inst_and_note_index_150123.json',
        load_group='train',
        transform=transforms.Spectrogram()
    )
    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = MusicNet(
        '.\\MusicNet',
        # metadata_path='./MusicNet/all_metadata_processed_150123.csv',
        metadata_path='./MusicNet/all_local_metadata_150123.csv',
        indexes_paths='./MusicNet/inst_and_note_index_150123.json',
        load_group='test',
        transform=transforms.Spectrogram()
    )
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # The other model is broken :(
    model = models.resnet18(weights='DEFAULT')
    model = nn.Sequential(
        model,
        # to the correct output size
        nn.Linear(1000, train_dataset.n_instruments),
        # manyhot 
        nn.Sigmoid()
        )

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 1

    model.train()
    for epoch_id in range(num_epochs):
        for batch_id, (batch_data, instrument, note) in enumerate(train_loader):
            # resnet excepts three channels of input
            batch_multi_channel = batch_data.repeat(1,3,1,1)
            
            # train
            output = model.forward(batch_multi_channel)
            loss = loss_fn(output, instrument.float())
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'epoch - {epoch_id}, batch - {batch_id}, loss - {loss}')

    # checking performence 
    test_accuracy = get_test_score(test_loader, model)
    print(f'test accuracy is {test_accuracy:.2f}%')
    # might need to add 
    # to make resnet use only one channel


if __name__ == '__main__':
    main()