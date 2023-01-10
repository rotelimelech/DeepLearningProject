from torch.utils.data import Dataset
from os.path import join 
import pandas as pd
import os
import torch
import torchaudio

# values used from MusicNet labels
USED_COLUMNS = ['start_time', 'end_time', 'instrument', 'note']
# The NN should receive a vector in a standard length. 
# This is the median length of a note in the dataset
STANDARDIZE_SIZE = 10240


class MusicNet(Dataset):
	"""
	Implements the MusicNet dataset as a pytorch object.
	The original dataset contains music samples. 
	This implementation splits every track to small segments
	each contains a single note in a standardized length
	"""

	def __init__(self, dataset_path, groups=['train', 'test'], transform=None):
		self.dataset_path = dataset_path
		self.transform = transform

		if isinstance(groups, list):
			self.groups = groups
		else:
			self.groups = [groups, ]

		# loading the entire metadata when creating the object.
		# Since our "true samples" are individual notes found 
		# across different files, we must load all the separate
		# files metadata to index them properly.
		self.all_metadata = self._load_metadata(groups)

	def _load_metadata(self, groups):
		joined_metadata_cols = ['csv_id', 'group'] + USED_COLUMNS
		all_metadata = pd.DataFrame(columns=joined_metadata_cols)

		for group in groups:
			csvs_folder = join(self.dataset_path, f'{group}_labels')
			for csv_file in os.listdir(csvs_folder):
				if not csv_file.endswith('csv'):
					continue
				csv_id = csv_file.split('.')[0]

				metadata = pd.read_csv(join(csvs_folder, csv_file))
				metadata['csv_id'] = csv_id
				metadata['group'] = group

				# adding the current file to all the metadata
				all_metadata = pd.concat(
					[all_metadata, metadata[joined_metadata_cols]], 
					ignore_index=True)

		return all_metadata

	def _norm_waveform_len(self, waveform):
		"""
		The NN should receive a vector in a constant length. 
		This function casts a waveform to a standard length.
		If the waveform is longer then `STANDARDIZE_SIZE`, we 
		clip it to fit this size
		If it is bigger, we pad with copies of itself until it 
		is of `STANDARDIZE_SIZE` length
		"""
		_, length = waveform.shape
		if length >= STANDARDIZE_SIZE:
			return waveform[:,:STANDARDIZE_SIZE]
		else:
			copies_needed = 1 + STANDARDIZE_SIZE // length
			duplicated_wf = torch.cat([waveform for _ in range(copies_needed)], 1)
			return duplicated_wf[:,:STANDARDIZE_SIZE]

	def __getitem__(self, index):
		"""
		Returns the `index`-th item as:
		[waveform, instrument, note]
		Notice - self.transform acts on waveform
		"""
		sample_data = self.all_metadata.iloc[index]
		sample_wav_path = join(
			self.dataset_path, 
			f"{sample_data['group']}_data", 
			f"{sample_data['csv_id']}.wav")
		waveform, sample_rate = torchaudio.load(sample_wav_path)

		clipped_sample = waveform[:, sample_data['start_time']:sample_data['end_time']]
		clipped_sample = self._norm_waveform_len(clipped_sample)

		if self.transform:
			clipped_sample = self.transform(clipped_sample)
		
		return clipped_sample, sample_data['instrument'], sample_data['note']

	def __len__(self):
		return len(self.all_metadata)
