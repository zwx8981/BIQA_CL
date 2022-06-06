import torch.nn as nn
import torch.nn.functional as F
class Autoencoder(nn.Module):
	"""
	The class defines the autoencoder model which takes in the features from the last convolutional layer of the
	"""
	def __init__(self, input_dims=128, code_dims=64):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
		nn.Linear(input_dims, code_dims),
		nn.ReLU())
		self.decoder = nn.Sequential(
		nn.Linear(code_dims, input_dims),
		nn.ReLU())

	def forward(self, x):
		encoded_x = self.encoder(x)
		reconstructed_x = self.decoder(encoded_x)
		reconstructed_x = F.normalize(reconstructed_x, p=2)
		return reconstructed_x


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=10):
	"""
	Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.

	"""
	lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
	print('lr is ' + str(lr))

	if (epoch % lr_decay_epoch == 0):
		print('LR is set to {}'.format(lr))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer