#import torch
from torch import nn

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

class BitPred(nn.Module):
	def __init__(self, ci, c2, c3):
		super(BitPred, self).__init__()
		self.act=nn.LeakyReLU()
		self.act0=nn.Sigmoid()
		self.layer01=nn.Linear(ci, c2)
		self.layer02=nn.Linear(c2, c3)
		self.layer03=nn.Linear(c3, c3)
		self.layer04=nn.Linear(c3, c3)
		self.layer05=nn.Linear(c3, c3)
		self.layer06=nn.Linear(c3, c3)
		self.layer07=nn.Linear(c3, 1)
	def forward(self, x):
		x=self.act (self.layer01(x))
		x=self.act (self.layer02(x))
		x=self.act (self.layer03(x))
		x=self.act (self.layer04(x))
		x=self.act (self.layer05(x))
		x=self.act (self.layer06(x))
		x=self.act0(self.layer07(x))
		
		#x=self.act (self.layer01(x))
		#t=self.act (self.layer02(x))
		#x=self.act (self.layer03(t))
		#t=self.act (self.layer04(x)+t)
		#x=self.act (self.layer05(t))
		#x=self.act (self.layer06(x)+t)
		#x=self.act0(self.layer07(x))
		return x

class Predictor(nn.Module):
	def __init__(self, reach):
		super(Predictor, self).__init__()

		nnb=reach*(reach+1)*2*3		#{neighbors,  threshold=(min+max)/2,  2 previous channels,  7 previous bits}

		self.bit7=BitPred(nnb+1+2+0, 96, 64)
		self.bit6=BitPred(nnb+1+2+1, 96, 64)
		self.bit5=BitPred(nnb+1+2+2, 96, 64)
		self.bit4=BitPred(nnb+1+2+3, 96, 64)
		self.bit3=BitPred(nnb+1+2+4, 96, 64)
		self.bit2=BitPred(nnb+1+2+5, 96, 64)
		self.bit1=BitPred(nnb+1+2+6, 96, 64)
		self.bit0=BitPred(nnb+1+2+7, 96, 64)

		self.pred=[self.bit7, self.bit6, self.bit5, self.bit4, self.bit3, self.bit2, self.bit1, self.bit0]

	def forward(self, x, bpos):
		return self.pred[bpos](x)
