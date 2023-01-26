import torch
import torch.nn as nn

#from MusicNet import MusicNet
from utils import *
from torch.optim.lr_scheduler import StepLR 

BATCH_SIZE = 10

def get_model(base_model, target_features):
    base_features = list(base_model.children())[-1].out_features
    return nn.Sequential(
        base_model,
        # to the correct output size
        nn.Linear(base_features, target_features),
        # manyhot 
        nn.Sigmoid()
        )

def train_single_layer(model, train_loader, val_loader, target, loss_fn, device,
        predictor, predictor_params, save_name_prefix, lr=0.001, scheduler_step_size=1, 
        scheduler_gamma=0.5, n_epochs=5):
    if target == 'instrument':
        target_features = train_loader.dataset.n_instruments
    else:
        target_features = train_loader.dataset.n_notes

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=scheduler_step_size, gamma=scheduler_gamma)

    losses = []
    model.train()
    for epoch_id in range(n_epochs):
        for batch_id, (batch_data, instrument, note) in enumerate(train_loader):
            # resnet excepts three channels of input
            batch_multi_channel = batch_data.repeat(1,3,1,1).to(device)
            
            # train
            output = model.forward(batch_multi_channel).cpu()

            if target == 'instrument':
                loss = loss_fn(output, instrument.float())
            else:
                loss = loss_fn(output, note.float())

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss)

            if batch_id % 10_000 == 0 and batch_id != 0:
                save_model(f'{save_name_prefix}_{target}_epoch_{epoch_id}_batch_{batch_id}')

            if batch_id % 200 == 0 and batch_id != 0:
                print(f'epoch - {epoch_id}, batch - {batch_id}, loss - {loss}')

        print(get_test_score(val_loader, model, target))
        scheduler.step()
    return model, losses