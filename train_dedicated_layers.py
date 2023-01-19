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

BATCH_SIZE = 10
N_EPOCHS = 20
SAMPLES_PER_BATCH = 200

# Enums for type of target to detect
TARGET_INST = 0
TARGET_NOTES = 1


def save_model(model, folder, name):
    now = datetime.now().strftime('%m%d_%H%M')
    torch.save(model, path_join(folder, f'{name}_{now}'))


def train_single_layer(base_model, train_loader, target, lr=0.001,
        scheduler_step_size=1, scheduler_gamma=0.5):
    """
    Adds a single linear layer to `base_model` and trains it to detect
    the instruments present in the sample.
    """
    if target == TARGET_INST:
        target_str = 'instrument'
        target_features = train_loader.dataset.n_instruments
    else:
        target_str = 'note'
        target_features = train_loader.dataset.n_notes

    base_features = list(base_model.children())[-1].out_features
    model = nn.Sequential(
        base_model,
        # to the correct output size
        nn.Linear(base_features, target_features),
        # manyhot 
        nn.Sigmoid()
        )
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=scheduler_step_size, 
        gamma=scheduler_gamma)
    # loss_fn = JaccardIndex(task="multiclass", num_classes=target_features)
    #torch.nn.BCEWithLogitsLoss()
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch_id in range(N_EPOCHS):
        for batch_id, (batch_data, instrument, note) in enumerate(train_loader):
            # resnet excepts three channels of input
            batch_multi_channel = batch_data.repeat(1,3,1,1)
            
            # train
            output = model.forward(batch_multi_channel)

            if target == TARGET_INST:
                loss = iou_loss(output, instrument.float())
            else:
                loss = iou_loss(output, note.float())

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
    base_model = models.resnet18(weights='DEFAULT')
    base_model = base_model.requires_grad_(False)

    note_detect_model = train_single_layer(
        base_model, train_loader, TARGET_NOTES, lr=1e-2)
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
