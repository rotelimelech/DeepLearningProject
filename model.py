import torch
import torch.nn as nn
from torchvision import models, transforms

class InstNoteIdentifier(torch.nn.Module):
    """
    A deep learning model to identify different instruments
    in a soundtrack, and what notes they are playing.

    The model uses resnet as its base mode, and adds two 
    parallel linear layers. One for identifying the notes
    and one for identifying the instruments. 
    To those two layers, and to the base model's output 
    via skip connection, we add another linear layer that 
    matches notes for instruments
    """

    def __init__(self, base_model, note_layer, inst_layer, output_size):
        super(InstNoteIdentifier, self).__init__()
        self.base_model = base_model
        self.note_layer = note_layer
        self.relu_notes = nn.ReLU(inplace=True)
        self.inst_layer = inst_layer
        self.relu_inst = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        base_model_out_features = list(base_model.children())[-1].out_features

        self.skip_conn_fc = nn.Linear(
            base_model_out_features + inst_layer.out_features + note_layer.out_features, 
            output_size, 
            bias=True)

    def forward(self, data):
        base_model_output = self.base_model(data)

        notes_detect_output = self.note_layer(base_model_output)
        notes_detect_output = self.relu_notes(notes_detect_output)

        inst_detect_output = self.inst_layer(base_model_output)
        inst_detect_output = self.relu_inst(inst_detect_output)

        skip_conn_vals = torch.cat([
            base_model_output, 
            notes_detect_output, 
            inst_detect_output],
            dim=1)

        out = self.skip_conn_fc(skip_conn_vals)
        out = self.sig(out)

        return out 

