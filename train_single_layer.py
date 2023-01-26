import torch
import torch.nn as nn

#from MusicNet import MusicNet
import utils
from MusicNetManyhotNotes import MusicNet
from torchaudio import transforms
from torchvision import models
from os.path import join as path_join
from datetime import datetime
from torch.optim.lr_scheduler import StepLR 
# from dedicated_layers_model import get_model, train_single_layer
import dedicated_layers_model
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    train_loader, val_loader, test_loader = utils.get_dataset_loaders()

    base_model = models.resnet18(weights='DEFAULT')
    base_model = base_model.requires_grad_(False)

    iou_model = dedicated_layers_model.get_model(base_model, train_loader.dataset.n_instruments)
    iou_model = iou_model.to(device)
    iou_model, iou_losses = dedicated_layers_model.train_single_layer(
        model=iou_model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        target='instrument', 
        loss_fn=utils.iou_loss, 
        device=device,
        predictor=utils.uniform_threshold_predictor,
        predictor_params=(0.6,),
        save_name_prefix='iou_loss',
        n_epochs=5)

    utils.save_model(iou_model, 'trained_models', 'iou_model_5_epochs')

if __name__ == '__main__':
    main()
