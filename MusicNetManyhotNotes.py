from torch.utils.data import Dataset
from os.path import join 
import numpy as np
import pandas as pd
import os
import json
import torch
import torchaudio

# values used from MusicNet labels
USED_COLUMNS = ['start_time', 'end_time', 'instrument', 'notes']
# The NN should receive a vector in a standard length. 
# This is the median length of a note in the dataset
STANDARDIZE_SIZE = 10240
TEST_TRAIN_VAL_SPLIT = {
	'train': [0, 0.6],
	'val': [0.6, 0.8],
	'test': [0.8, 1],
}

def _str_keys_to_ints(x):
	""" 
	In JSON file, keys must be strings. This function casts them
	back to ints
	"""
	for k in tuple(x.keys()):
		x[int(k)] = x.pop(k)
	return x

class MusicNet(Dataset):
	"""
	Implements the MusicNet dataset as a pytorch object.
	The original dataset contains music samples. 
	This implementation splits every track to small segments
	each contains a single note in a standardize length

	This class regards a single timespan where an instrument played
	more then one note as one example
	"""

	def __init__(self, dataset_path, metadata_path=None, indexes_paths=None, 
			load_group=None, nsynth_groups=['train', 'test'], transform=None,
			wav_block_shuffle=None):
		"""
		Loads and indexes metadata from Nsynth dataset.

		Parameters
		----------
		dataset_path: str
		 	Path to the extracted nsynth dataset.
		metadata_path: str
			Path to the pre-indexed dataset's metadata. 
			If supplied, will not index the csv files from `dataset_path`.
			If used, `indexes_paths` must also be supplied
		indexes_paths: str
			Path to a json file containing mapping between nsynth's note
			and instrument's ids to standardized indexes. (nsynth skips
			some ids in their mapping)
			If used, `metadata_path` must also be supplied
		load_group: str
			Subgroup of the dataset to load. One of ['test', 'train', 'val']
			Only used when loading a pre-indexed version of the dataset.
		groups: list or str
			Groups to load from nsynth's dataset
			If using a pre-indexed version of the dataset, will not refer 
			to those groups as the test and train groups.
			For this case, see `load_group`
		wav_block_shuffle: int
			A major bottleneck is disk access. We enable the user to choose
			to load samples from a single wav in a randomized order, drastically 
			reducing disk access 
		"""
		self.dataset_path = dataset_path
		self.transform = transform

		if isinstance(nsynth_groups, list):
			self.nsynth_groups = nsynth_groups
		else:
			self.nsynth_groups = [nsynth_groups, ]

		# Loading WAV from disk takes a long time. Here we sacrifice 
		# Some randomness in the data, by loading a single wav, and iterating
		# over it alone
		self.currently_loaded_wav = None
		self.currently_loaded_wav_id = None

		# loading the entire metadata when creating the object.
		# Since our "true samples" are individual notes found 
		# across different files, we must load all the separate
		# files metadata to index them properly.
		# This is a costly process, and we enable loading a previously 
		# processed metadata file instead.
		if metadata_path is None and indexes_paths is None:
			self._index_dataset(nsynth_groups)

		elif metadata_path is not None and indexes_paths is not None:
			self._load_preindexed_dataset(metadata_path, indexes_paths, 
				load_group, wav_block_shuffle)
		else:
			raise Exception('To load preprocessed metadata you must supply both ' + \
				'a matadata csv file and a note and instrument indexing json.')

	def _index_dataset(self, nsynth_groups):
		print('indexing notes and creating')
		self.all_metadata = self._load_metadata(nsynth_groups)
		
		# create a mapping between instrument and notes to ids in 
		# the manyhot vector
		unique_instruments = self.all_metadata.instrument.unique()
		self.n_instruments = len(unique_instruments)
		self.instrument_to_idx = {inst: i for i, inst in enumerate(unique_instruments)}
		
		unique_notes = np.unique(self.all_metadata.notes.sum())
		self.n_notes = len(unique_notes)
		self.note_to_idx = {note: i for i, note in enumerate(unique_notes)}

	def _load_preindexed_dataset(self, metadata_path, indexes_paths, load_group, wav_block_shuffle):
		print('loading cached metadata')
		self.all_metadata = pd.read_csv(metadata_path)
		self.all_metadata['notes'] = self.all_metadata['notes'].apply(json.loads)
		self.all_metadata['instrument'] = self.all_metadata['instrument'].apply(json.loads)

		with open(indexes_paths, 'r') as f:
			indexed_vals = json.load(f)

		self.instrument_to_idx = _str_keys_to_ints(indexed_vals['instrument_to_idx'])
		self.n_instruments = len(self.instrument_to_idx)

		self.note_to_idx = _str_keys_to_ints(indexed_vals['note_to_idx'])
		self.n_notes = len(self.note_to_idx)

		if load_group is not None:
			n_samples = len(self.all_metadata)
			start_index = int(TEST_TRAIN_VAL_SPLIT[load_group][0] * n_samples)
			end_index = int(TEST_TRAIN_VAL_SPLIT[load_group][1] * n_samples)
			self.all_metadata = self.all_metadata[start_index:end_index]

		if wav_block_shuffle is not None:
			self.all_metadata = self.all_metadata.sort_values('csv_id').reset_index(drop=True)
			self._block_shuffle(wav_block_shuffle)

	def _combine_multi_note_lines(self, df):
		"""
		At any given time, more than a single note might be played by an instrument.
		This leads to several entries with the same times but a different note.
		In this function we unite these entries to one
		"""
		grouped_entries = df.groupby(['start_time', 'end_time', 'instrument'])
		multinote_df = grouped_entries.apply(lambda x: list(x['note'])).reset_index()
		multinote_df.columns = ['start_time', 'end_time', 'instrument', 'notes'] 
		return multinote_df

	def _block_shuffle(self, group_size):
		n_entries = len(self.all_metadata)
		indexes = np.arange(0, n_entries, group_size)
		np.random.shuffle(indexes)

		shuffled_df = pd.DataFrame(columns=self.all_metadata.columns)
		for block_start in indexes:
			shuffled_df = pd.concat(
				[shuffled_df, self.all_metadata[block_start:block_start+group_size]],
				ignore_index=True
				)

		self.all_metadata = shuffled_df

	def _load_metadata(self, groups):
		joined_metadata_cols = ['csv_id', 'group'] + USED_COLUMNS
		all_metadata = pd.DataFrame(columns=joined_metadata_cols)

		for group in self.nsynth_groups:
			csvs_folder = join(self.dataset_path, f'{group}_labels')
			for csv_file in os.listdir(csvs_folder):
				if not csv_file.endswith('csv'):
					continue
				csv_id = csv_file.split('.')[0]

				metadata = pd.read_csv(join(csvs_folder, csv_file))
				new_metadata = pd.DataFrame(columns=joined_metadata_cols)
				time_stamps = np.unique(np.concatenate((metadata['start_time'], metadata['end_time'])))

				notes = metadata['note']
				instruments = metadata['instrument']
				relevant_notes = []
				relevant_instruments = []
				start_times = time_stamps[:-1]
				end_times = time_stamps[1:]
				for time_stamp in time_stamps[:-1]:
					relevant_notes.append((notes[(metadata['start_time'] <= time_stamp) * (metadata['end_time'] >= time_stamp)]).values)
					relevant_instruments.append((instruments[(metadata['start_time'] <= time_stamp) * (metadata['end_time'] >= time_stamp)]).values)

				new_metadata['start_time'] = start_times
				new_metadata['end_time'] = end_times
				new_metadata['instrument'] = relevant_instruments
				new_metadata['notes'] = relevant_notes
				new_metadata['csv_id'] = csv_id
				new_metadata['group'] = group

				# adding the current file to all of the metadata
				all_metadata = pd.concat(
					[all_metadata, new_metadata[joined_metadata_cols]],
					ignore_index=True)

		# Shuffle indexes
		all_metadata.sample(frac=1).reset_index(drop=True)
		return all_metadata


	def _to_manyhot_vector(self, on_indexes, data_type):
		if data_type == 'note':
			mapping = self.note_to_idx
			vect_len = self.n_notes
		else:
			mapping = self.instrument_to_idx
			vect_len = self.n_instruments
		
		manyhot_vec = np.zeros(vect_len)
		for idx in on_indexes:
			manyhot_vec[mapping[idx]] = 1

		return manyhot_vec

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
		# required wav is not cached
		if self.currently_loaded_wav_id != sample_data['csv_id']:
			sample_wav_path = join(
				self.dataset_path, 
				f"{sample_data['group']}_data", 
				f"{sample_data['csv_id']}.wav")
			waveform, sample_rate = torchaudio.load(sample_wav_path)

			self.currently_loaded_wav_id = sample_data['csv_id']
			self.currently_loaded_wav = waveform
		# skip file load and use cache 
		else:
			waveform = self.currently_loaded_wav

		clipped_sample = waveform[:, sample_data['start_time']:sample_data['end_time']]
		clipped_sample = self._norm_waveform_len(clipped_sample)

		if self.transform:
			clipped_sample = self.transform(clipped_sample)
		
		notes_vect = self._to_manyhot_vector(sample_data['notes'], 'note')
		instrument_vect = self._to_manyhot_vector(sample_data['instrument'], 'instrument')

		return clipped_sample, instrument_vect, notes_vect

	def __len__(self):
		return len(self.all_metadata)