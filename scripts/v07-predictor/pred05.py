#import torch
from torch import nn

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

class Predictor(nn.Module):
	def __init__(self, reach):
		super(Predictor, self).__init__()

		self.act=nn.LeakyReLU()
		self.act0=nn.Sigmoid()

		#self.reach=reach
		#self.nnb=reach*(reach+1)*2

		self.C=reach*(reach+1)*2*3+1+2+7	#{neighbors,  threshold=(min+max)/2,  2 previous channels,  7 previous bits}

		self.params=nn.ParameterList()
		self.layer01=nn.Linear(self.C, self.C//2)
		self.layer02=nn.Linear(self.C//2, self.C//4)
		self.layer03=nn.Linear(self.C//4, self.C//4)
		self.layer04=nn.Linear(self.C//4, self.C//4)
		self.layer05=nn.Linear(self.C//4, self.C//4)
		self.layer06=nn.Linear(self.C//4, self.C//4)
		self.layer07=nn.Linear(self.C//4, 1)

	def forward(self, x):
		x=self.act (self.layer01(x))
		x=self.act (self.layer02(x))
		x=self.act (self.layer03(x))
		x=self.act (self.layer04(x))
		x=self.act (self.layer05(x))
		x=self.act (self.layer06(x))
		x=self.act0(self.layer07(x)).squeeze()
		
		#x=self.act (self.layer01(x))
		#t=self.act (self.layer02(x))
		#x=self.act (self.layer03(t))
		#t=self.act (self.layer04(x)+t)
		#x=self.act (self.layer05(t))
		#x=self.act (self.layer06(x)+t)
		#x=self.act0(self.layer07(x)).squeeze()
		return x
