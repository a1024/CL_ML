#import torch
from torch import nn

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

class Predictor(nn.Module):
	def __init__(self, reach):
		super(Predictor, self).__init__()

		self.act=nn.LeakyReLU()
		self.act0=nn.Sigmoid()

		self.reach=reach
		self.nnb=reach*(reach+1)*2
		self.nparams=self.nnb*18+6

		self.C=self.nnb

		self.d7_1=nn.Linear(self.nnb, self.C)
		self.d7_2=nn.Linear(self.C, self.C)
		self.d7_3=nn.Linear(self.C, 1)

		self.d6_1=nn.Linear(self.nnb+1, self.C)
		self.d6_2=nn.Linear(self.C, self.C)
		self.d6_3=nn.Linear(self.C, 1)

		self.d5_1=nn.Linear(self.nnb+2, self.C)
		self.d5_2=nn.Linear(self.C, self.C)
		self.d5_3=nn.Linear(self.C, 1)

		self.d4_1=nn.Linear(self.nnb+3, self.C)
		self.d4_2=nn.Linear(self.C, self.C)
		self.d4_3=nn.Linear(self.C, 1)

		self.d3_1=nn.Linear(self.nnb+4, self.C)
		self.d3_2=nn.Linear(self.C, self.C)
		self.d3_3=nn.Linear(self.C, 1)

		self.d2_1=nn.Linear(self.nnb+5, self.C)
		self.d2_2=nn.Linear(self.C, self.C)
		self.d2_3=nn.Linear(self.C, 1)

		self.d1_1=nn.Linear(self.nnb+6, self.C)
		self.d1_2=nn.Linear(self.C, self.C)
		self.d1_3=nn.Linear(self.C, 1)

		self.d0_1=nn.Linear(self.nnb+7, self.C)
		self.d0_2=nn.Linear(self.C, self.C)
		self.d0_3=nn.Linear(self.C, 1)

	def pred_bit7(self, nb):
		x=self.act (self.d7_1(nb))
		x=self.act (self.d7_2(x))
		x=self.act0(self.d7_3(x))
		return x

	def pred_bit6(self, nb):
		x=self.act (self.d6_1(nb))
		x=self.act (self.d6_2(x))
		x=self.act0(self.d6_3(x))
		return x

	def pred_bit5(self, nb):
		x=self.act (self.d5_1(nb))
		x=self.act (self.d5_2(x))
		x=self.act0(self.d5_3(x))
		return x

	def pred_bit4(self, nb):
		x=self.act (self.d4_1(nb))
		x=self.act (self.d4_2(x))
		x=self.act0(self.d4_3(x))
		return x

	def pred_bit3(self, nb):
		x=self.act (self.d3_1(nb))
		x=self.act (self.d3_2(x))
		x=self.act0(self.d3_3(x))
		return x

	def pred_bit2(self, nb):
		x=self.act (self.d2_1(nb))
		x=self.act (self.d2_2(x))
		x=self.act0(self.d2_3(x))
		return x

	def pred_bit1(self, nb):
		x=self.act (self.d1_1(nb))
		x=self.act (self.d1_2(x))
		x=self.act0(self.d1_3(x))
		return x

	def pred_bit0(self, nb):
		x=self.act (self.d0_1(nb))
		x=self.act (self.d0_2(x))
		x=self.act0(self.d0_3(x))
		return x
