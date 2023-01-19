import torch
import torch.nn as nn

#from MusicNet import MusicNet
from utils import get_test_score, get_dataset_loaders, iou_loss
from MusicNetManyhotNotes import MusicNet
from torchaudio import transforms
from torchvision import models
from os.path import join as path_join
from datetime import datetime
from torch.optim.lr_scheduler import StepLR 
from torchmetrics import JaccardIndex # IoU loss
from ConvAxisModel import ConvAxisModel


BATCH_SIZE = 10
N_EPOCHS = 20
SAMPLES_PER_BATCH = 200

# Enums for type of target to detect
TARGET_INST = 0
TARGET_NOTES = 1


def save_model(model, folder, name):
    now = datetime.now().strftime('%m%d_%H%M')
    torch.save(model, path_join(folder, f'{name}_{now}'))


def train(model, train_loader, target, lr=0.001,
        scheduler_step_size=1, scheduler_gamma=0.5):

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=scheduler_step_size, 
        gamma=scheduler_gamma)
    #torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch_id in range(N_EPOCHS):
        for batch_id, (batch_data, instrument, note) in enumerate(train_loader):
            # resnet excepts three channels of input            
            # train
            output = model(batch_data)[0][0]

            if target == 'instrument':
                # loss = iou_loss(output, instrument.float())
                loss = loss_fn(output, instrument.float())

            else:
                # loss = iou_loss(output, note.float())
                loss = loss_fn(output, note.float())

            opt.zero_grad()
            loss.backward()
            opt.step()

            # if batch_id == 200:
                # break
            print(f'epoch - {epoch_id}, batch - {batch_id}, loss - {loss}')

        save_model(model, 'trained_models', f'{target}_iou_loss_ep_{epoch_id}')
        scheduler.step()
    return model

def main():
    train_loader, test_loader = get_dataset_loaders()

    # Training a single layer do detect instruments
    n_notes = train_loader.dataset.n_notes
    note_detect_model = ConvAxisModel(11, 83)
    note_detect_model = train(note_detect_model, train_loader, 'notes', lr=1e-5)

    save_model(note_detect_model, 'trained_models', 
        f'note_detect_single_layer_{N_EPOCHS}_epoch')
    note_test_score = get_test_score(test_loader, note_detect_model, TARGET_NOTES)
    print(f'test score for notes detection - {note_test_score}')

    inst_detect_model = train_single_layer(
        base_model, train_loader, TARGET_INST)
    save_model(inst_detect_model, 'trained_models', 
        f'instrument_detect_single_layer_{N_EPOCHS}_epoch')
    inst_test_score = get_test_score(test_loader, inst_detect_model, TARGET_INST)
    print(f'Perfect identification - {inst_test_score}')
    print(f'Instruments identified correctly - {inst_test_score}')



if __name__ == '__main__':
    main()
