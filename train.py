import torch
import torch.nn as nn

#from MusicNet import MusicNet
from MusicNetManyhotNotes import MusicNet
from torchaudio import transforms
from torchvision import models
from os.path import join as path_join
from datetime import datetime

BATCH_SIZE = 10
N_EPOCHS = 1

# Enums for type of target to detect
TARGET_INST = 0
TARGET_NOTES = 1


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


def save_model(model, folder, name):
    now = datetime.now().strftime('%m%d_%H%M')
    torch.save(model, path_join(folder, f'{name}_{now}'))


def train_single_layer(base_model, train_loader, loss_fn, target):
    """
    Adds a single linear layer to `base_model` and trains it to detect
    the instruments present in the sample.
    """
    if target == TARGET_INST:
        target_features = train_loader.dataset.n_instruments
    else:
        target_features = train_loader.dataset.n_notes

    base_features = list(base_model.children())[-1].out_features
    model = nn.Sequential(
        base_model,
        # to the correct output size
        nn.Linear(base_features, target_features),
        # manyhot 
        nn.Sigmoid()
        )
    opt = torch.optim.SGD(model.parameters(), lr=0.001)

    model.train()
    for epoch_id in range(N_EPOCHS):
        for batch_id, (batch_data, instrument, note) in enumerate(train_loader):
            # resnet excepts three channels of input
            batch_multi_channel = batch_data.repeat(1,3,1,1)
            
            # train
            output = model.forward(batch_multi_channel)

            if target == TARGET_INST:
                loss = loss_fn(output, instrument.float())
            else:
                loss = loss_fn(output, notes.float())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if batch_id == 100:
                break
            print(f'epoch - {epoch_id}, batch - {batch_id}, loss - {loss}')
    return model

def main():
    train_dataset = MusicNet(
        '.\\MusicNet',
        # metadata_path='./MusicNet/all_metadata_processed_150123.csv',
        metadata_path='./MusicNet/all_local_metadata_150123.csv',
        indexes_paths='./MusicNet/inst_and_note_index_150123.json',
        load_group='train',
        transform=transforms.Spectrogram()
    )
    train_loader =  torch.utils.data.DataLoader(train_dataset, 
        batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = MusicNet(
        '.\\MusicNet',
        # metadata_path='./MusicNet/all_metadata_processed_150123.csv',
        metadata_path='./MusicNet/all_local_metadata_150123.csv',
        indexes_paths='./MusicNet/inst_and_note_index_150123.json',
        load_group='test',
        transform=transforms.Spectrogram()
    )
    test_loader =  torch.utils.data.DataLoader(test_dataset, 
        batch_size=BATCH_SIZE, shuffle=True)

    # Training a single layer do detect instruments
    base_model = models.resnet18(weights='DEFAULT')
    loss_fn = torch.nn.MSELoss()

    inst_detect_model = train_single_layer(
        base_model, train_loader, loss_fn, TARGET_INST)
    save_model(inst_detect_model, 'trained_models', 
        f'instrument_detect_single_layer_{N_EPOCHS}_epoch')

    note_detect_model = train_single_layer(
        base_model, train_loader, loss_fn, TARGET_INST)
    save_model(note_detect_model, 'trained_models', 
        f'note_detect_single_layer_{N_EPOCHS}_epoch')

if __name__ == '__main__':
    main()