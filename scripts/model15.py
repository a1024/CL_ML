import numpy as np

import torch
from torch import nn

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

class Predictor(nn.Module):
	def __init__(self):
		super(Predictor, self).__init__()
		self.act=nn.LeakyReLU()

		#M15 error attention
		self.rfield=15

		#image
		self.conv01x=nn.Conv2d( 3, 64, 3, 1, 1)#input [3, self.rfield, W]
		self.conv02x=nn.Conv2d(64, 64, 3, 1, [0, 1])
		self.conv03x=nn.Conv2d(64, 64, 3, 1, [0, 1])
		self.conv04x=nn.Conv2d(64, 64, 3, 1, [0, 1])

		#error
		self.conv01e=nn.Conv2d( 3, 64, 3, 1, 1)
		self.conv02e=nn.Conv2d(64, 64, 3, 1, [0, 1])
		self.conv03e=nn.Conv2d(64, 64, 3, 1, [0, 1])
		self.conv04e=nn.Conv2d(64, 64, 3, 1, [0, 1])

		#attention
		self.conv05=nn.Conv2d(64, 64, 3, 1, [0, 1])
		self.conv06=nn.Conv2d(64, 64, 3, 1, [0, 1])
		self.conv07=nn.Conv2d(64, 64, 3, 1, [0, 1])
		self.conv08=nn.Conv2d(64, 64, 3, 1, [0, 1])
		self.conv09=nn.Conv2d(64,  3, 3, 1, 1)#output [3, 1, W]  the predicted row

	def predict(self, slice, error):
		slice=self.act(self.conv01x(slice))
		slice=self.act(self.conv02x(slice))
		slice=self.act(self.conv03x(slice))
		slice=self.act(self.conv04x(slice))

		error=self.act(self.conv01e(error))
		error=self.act(self.conv02e(error))
		error=self.act(self.conv03e(error))
		error=self.act(self.conv04e(error))

		pred=torch.zeros(slice.shape, device=slice.device)
		slice=slice.permute(0, 2, 3, 1)
		error=error.permute(0, 2, 3, 1)
		slice=slice.reshape(slice.shape[0], slice.shape[1], slice.shape[2], 8, 8)
		error=error.reshape(slice.shape[0], slice.shape[1], slice.shape[2], 8, 8)
		pred=torch.matmul(error, slice)
		pred=pred.reshape(pred.shape[0], pred.shape[1], pred.shape[2], 64)
		pred=pred.permute(0, 3, 1, 2)

		#for ky in range(8):#SLOW
		#	for kx in range(8):
		#		for ki in range(8):
		#			idx_dst=ky<<3|kx
		#			idx_left=ky<<3|ki
		#			idx_right=ki<<3|kx
		#			pred[:, idx_dst:idx_dst+1, :, :]+=torch.mul(error[:, idx_left:idx_left+1, :, :], slice[:, idx_right:idx_right+1, :, :])

		#pred=torch.mul(slice, torch.add(torch.abs(error), 1))#bad for training

		pred=self.act(self.conv05(pred))
		pred=self.act(self.conv06(pred))
		pred=self.act(self.conv07(pred))
		pred=self.act(self.conv08(pred))
		pred=self.act(self.conv09(pred))
		return pred

	def forward(self, x):
		error=x
		pred=None
		for ky in range(self.rfield, x.shape[2]):
			slice_x=x[:, :, ky-self.rfield:ky, :]
			slice_e=error[:, :, ky-self.rfield:ky, :]
			pred_row=self.predict(slice_x, slice_e)
			if pred is not None:
				pred=torch.cat((pred, pred_row), dim=2)
			else:
				pred=pred_row
			act_row=x[:, :, ky:ky+1, :]
			error[:, :, ky:ky+1, :]=torch.sub(act_row, pred_row)
		truth=x[:, :, self.rfield:, :]
		return pred, truth

