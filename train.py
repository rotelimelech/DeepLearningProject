import torch
from MusicNet  import MusicNet
from torchaudio import transforms
from torchvision import models

"""
Train the model using MusicNet dataset
"""
BATCH_SIZE = 10


def main():
	dataset = MusicNet(
		'./MusicNet',
		# groups='train',
		transform=transforms.Spectrogram()
	)
	train_loader =  torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
	
	# The other model is broken :(
	model = models.resnet18(weights='DEFAULT')
	for batch_data, instrument, note in train_loader:
		# resnet excepts three channels of input
		batch_multi_channel = batch_data.repeat(1,3,1,1)
		output = model.forward(batch_multi_channel)
		print(output)
		print(output.shape)

		break
			
		
	# might need to add 
	# to make resnet use only one channel


if __name__ == '__main__':
	main()