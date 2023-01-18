from MusicNetManyhotNotes import MusicNet
from torchaudio import transforms
import torch
import numpy as np

BATCH_SIZE = 10

TARGET_INST = 0
TARGET_NOTES = 1
OUTPUT_THRESHOLD = 0.6


def get_dataset_loaders():
    train_dataset = MusicNet(
        '.\\MusicNet',
        # metadata_path='./MusicNet/all_metadata_processed_150123.csv',
        metadata_path='./MusicNet/all_metadata_processed_180123.csv',
        indexes_paths='./MusicNet/inst_and_note_index_150123.json',
        load_group='train',
        transform=transforms.Spectrogram()
    )
    train_loader =  torch.utils.data.DataLoader(train_dataset, 
        batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = MusicNet(
        '.\\MusicNet',
        # metadata_path='./MusicNet/all_metadata_processed_150123.csv',
        metadata_path='./MusicNet/all_metadata_processed_180123.csv',
        indexes_paths='./MusicNet/inst_and_note_index_150123.json',
        load_group='test',
        transform=transforms.Spectrogram()
    )
    test_loader =  torch.utils.data.DataLoader(test_dataset, 
        batch_size=BATCH_SIZE, shuffle=True) 
    return train_loader, test_loader


def get_test_score(test_loader, model, target):
    perfect_match = 0
    identified_instances = 0
    false_pos_instances = 0
    total_instances = 0

    with torch.no_grad():
        for batch_id, (batch_data, instrument, note) in enumerate(test_loader):
            batch_multi_channel = batch_data.repeat(1,3,1,1)
            try:
                output = model(batch_multi_channel)
                prediction = output > OUTPUT_THRESHOLD
                prediction[np.arange((len(output))), output.argmax(dim=1)] = True
                prediction = prediction.numpy()

                if target == TARGET_INST:
                    labels = (instrument == 1).numpy()
                else:
                    labels = (note == 1).numpy()
                
                perfect_match += (prediction == labels).all()
                identified_instances += ((prediction == labels) * (labels != 0)).sum()
                false_pos_instances += ((prediction != labels) * (prediction != 0)).sum()
                total_instances += labels.sum()
            except Exception as e:
                import ipdb
                ipdb.set_trace()

    n_test_samples = len(test_loader.dataset)
    return perfect_match/n_test_samples, \
        identified_instances/total_instances, \
        false_pos_instances/n_test_samples


def save_model(model, folder, name):
    now = datetime.now().strftime('%m%d_%H%M')
    torch.save(model, path_join(folder, f'{name}_{now}'))


def iou_loss(prediction, labels):
    false_positives = 0
    false_negatives = 0

    for sample_id in range(prediction.shape[0]):
        for i, label in enumerate(labels[sample_id,:]):
            if label == 1:
                # the result should be 1 (positive) but the model
                # misses - so this is a false negative
                false_negatives += (prediction[sample_id, i] - 1)**2
            else:
                false_positives += prediction[sample_id, i]**2

    # we give both errors an equal value
    n_ones = sum(sum(labels))

    return false_negatives/n_ones + false_positives/(prediction.shape[0]*prediction.shape[1]-n_ones)
