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
		#self.nparams=self.nnb*18+6

		self.C=self.nnb*3+1

		self.params=nn.ParameterList()
		self.layer01=[]
		self.layer02=[]
		self.layer03=[]
		self.layer04=[]

		for idx in range(24):
			l01=nn.Linear(self.C+(idx>>3), self.C//2)	#append prev subpixel of current pixel and threshold=(min+max)/2 to neighbors
			l02=nn.Linear(self.C//2, self.C//4)
			l03=nn.Linear(self.C//4, self.C//4)
			l04=nn.Linear(self.C//4, 1)
			self.params.append(l01.weight)
			self.params.append(l01.bias)
			self.params.append(l02.weight)
			self.params.append(l02.bias)
			self.params.append(l03.weight)
			self.params.append(l03.bias)
			self.params.append(l04.weight)
			self.params.append(l04.bias)
			self.layer01.append(l01)
			self.layer02.append(l02)
			self.layer03.append(l03)
			self.layer04.append(l04)

	def forward(self, nb, workidx):
		x=self.act (self.layer01[workidx](nb))
		x=self.act (self.layer02[workidx](x))
		x=self.act (self.layer03[workidx](x))
		x=self.act0(self.layer04[workidx](x)).squeeze()
		return x
