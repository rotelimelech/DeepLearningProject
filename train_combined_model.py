import torch
import torch.nn as nn
import numpy as np
#from MusicNet import MusicNet
from MusicNetManyhotNotes import MusicNet
from utils import get_dataset_loaders
from torchaudio import transforms
from torchvision import models
from os.path import join as path_join
from datetime import datetime
from model import InstNoteIdentifier

BATCH_SIZE = 10
N_EPOCHS = 1
# just because my computer is slow :) 
BATCHES_PER_EPOCH = 200

# Enums for type of target to detect
TARGET_INST = 0
TARGET_NOTES = 1

def get_test_score(test_loader, model, target):
    perfect_match = 0
    identified_notes = 0
    missidentified_notes = 0

    with torch.no_grad():
        for batch_id, (batch_data, instrument, note) in enumerate(test_loader):
            batch_multi_channel = batch_data.repeat(1,3,1,1)
            output = model(batch_multi_channel)
            prediction = output.argmax(dim=1, keepdim=True)

            if target == TARGET_INST:
                labels = instrument.argmax(dim=1, keepdim=True)
            else:
                labels = note.argmax(dim=1, keepdim=True)

            perfect_match += (prediction == labels).all()
            identified_notes += (prediction == labels).sum().item()
            missidentified_notes += 
            prediction.eq(labels.view_as(prediction)).sum().item()

    n_test_samples = len(test_loader.dataset)
    return correct / SAMPLES_PER_BATCH


def train_with_warmup(model, train_loader, warmup_batches=100, 
    init_lr=1e-5, final_lr=0.001):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    
    warmup_lr_pace = (final_lr - init_lr) / warmup_batches
    for epoch_id in range(N_EPOCHS):
        for batch_id, (batch_data, instrument, note) in enumerate(train_loader):
            # resnet excepts three channels of input
            batch_multi_channel = batch_data.repeat(1,3,1,1)
            
            # Right now only one instrument per sample
            # so this hotpatch kinda makes it work
            note_per_inst = np.zeros((len(instrument), len(note)))
            instrument = np.argmax(instrument)
            note_per_inst[instrument,:] = note
            note_per_inst = note_per_inst.reshape([-1, 1])

            # train
            output = model.forward(batch_multi_channel)
            loss = loss_fn(output, note_per_inst.float())
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            # warmup
            if epoch_id == 0 and batch_id < warmup_batches:
                opt.lr += warmup_lr_pace

            print(f'epoch - {epoch_id}, batch - {batch_id}, loss - {loss}')

            if batch_id == BATCHES_PER_EPOCH:
                break

    return model


def main():
train_loader, test_loader = get_dataset_loaders()

base_model = models.resnet18(weights='DEFAULT')

pretrained_notes_model = torch.load(path_join(
    'trained_models',
    'note_detect_single_layer_1_epoch_0117_1245'
    ))
pretrained_notes_fc = list(pretrained_notes_model.children())[-2]

pretrained_inst_model = torch.load(path_join(
    'trained_models',
    'instrument_detect_single_layer_1_epoch_0117_1302'
    ))
pretrained_inst_fc = list(pretrained_inst_model.children())[-2]

combined_model = InstNoteIdentifier(
        base_model,
        pretrained_notes_fc, 
        pretrained_inst_fc,
        pretrained_notes_fc.out_features * pretrained_inst_fc.out_features
    )

    train_with_warmup(combined_model, train_loader)

if __name__ == '__main__':
    main()