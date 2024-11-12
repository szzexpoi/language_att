import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_ori import resnet50


class Inception_Encoder(nn.Module):
	""" An inception-like encoder for saliency prediction.
	"""
	def __init__(self, input_size, embedding_size):
		super(Inception_Encoder, self).__init__()
		self.inception_1 = nn.Conv2d(input_size, embedding_size, kernel_size=1,
								padding="same", stride=1, dilation=1, bias=False)
		self.inception_2 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3,
								padding="same", stride=1, dilation=4, bias=False)
		self.inception_3 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3,
								padding="same", stride=1, dilation=8, bias=False)
		self.inception_4 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3,
								padding="same", stride=1, dilation=16, bias=False)

	def forward(self, x):
		""" Implementation of the inception data flow proposed
			in the DINet paper. Note that three of the branches
			are conditioned on the first conv layer, and there is
			a sum fusion along side the independent branches.

			Input:
				x: A Batch x N x H x W tensor encoding the visual
				features extracted from the backbone, where N is
				the number of filters for the features.
			Return:
				A Batch x M x H x W tensor encoding the features
				processed by the inception encoder, where M is the embedding
				size*4.

		"""
		x = torch.relu(self.inception_1(x))
		b_1 = torch.relu(self.inception_2(x))
		b_2 = torch.relu(self.inception_3(x))
		b_3 = torch.relu(self.inception_4(x))
		fused_b = b_1 + b_2 + b_3 # sum fusion

		return torch.cat([fused_b, b_1, b_2, b_3], dim=1)


class Simple_Decoder(nn.Module):
	""" A simple feed-forward decoder for saliency prediction.
	"""

	def __init__(self, input_size, embedding_size):
		super(Simple_Decoder, self).__init__()
		self.decoder_1 = nn.Conv2d(input_size, embedding_size, kernel_size=3,
						padding="same", stride=1, bias=False)
		self.decoder_2 = nn.Conv2d(embedding_size, embedding_size, kernel_size=3,
						padding="same", stride=1, bias=False)
		self.decoder_3= nn.Conv2d(embedding_size, 1, kernel_size=3,
						padding="same", stride=1, bias=False)

	def forward(self, x):
		""" A standard feed-forward flow of decoder.
			Note that at the end there is a rescaling
			operation.
		"""

		x = torch.relu(self.decoder_1(x))
		x = torch.relu(self.decoder_2(x))
		x = torch.sigmoid(self.decoder_3(x)) 
		x = F.interpolate(x, (240, 320)) # for a fair comparison
		return x
	

class DINet(nn.Module):
	""" A reimplementation of the saliency prediction model
		introduced in the following paper:
		https://arxiv.org/abs/1904.03571
	"""
	def __init__(self, embedding_size=512):
		super(DINet, self).__init__()
		self.dilated_backbone = resnet50(pretrained=True)
		self.dilate_resnet(self.dilated_backbone) # DINet use the same Dilated ResNet as SAM
		self.dilated_backbone = nn.Sequential(*list(
								self.dilated_backbone.children())[:-2])
		self.encoder = Inception_Encoder(2048, embedding_size)
		self.decoder = Simple_Decoder(embedding_size*4, embedding_size)

	def dilate_resnet(self, resnet):
		""" Converting standard ResNet50 into a dilated one.
		"""
		resnet.layer3[0].conv1.stride = 1
		resnet.layer3[0].downsample[0].stride = 1
		resnet.layer4[0].conv1.stride = 1
		resnet.layer4[0].downsample[0].stride = 1

		for block in resnet.layer3:
			block.conv2.dilation = 2
			block.conv2.padding = 2

		for block in resnet.layer4:
			block.conv2.dilation = 4
			block.conv2.padding = 4

	def forward(self, image):
		""" Data flow for DINet. Most of the key
			components have been implemented in
			separate modules.
		"""

		x = self.dilated_backbone(image)
		x = self.encoder(x)
		x = self.decoder(x)

		return x