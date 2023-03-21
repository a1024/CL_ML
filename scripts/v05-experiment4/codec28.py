import torch
from torch import nn

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()
		self.act=nn.LeakyReLU()
		self.padT=nn.ZeroPad2d((1, 1, 2, 0))#(left, right, top, bottom)
		self.padL=nn.ZeroPad2d((1, 0, 0, 0))

		#per-pixel predictor
		self.C=32
		self.nlayers=8

		#top part
		self.ct01=nn.Conv2d(     3, self.C, [2, 3], 1, 0)
		self.ct02=nn.Conv2d(self.C, self.C, 3, 1, 0)
		self.ct03=nn.Conv2d(self.C, self.C, 3, 1, 0)
		self.ct04=nn.Conv2d(self.C, self.C, 3, 1, 0)
		self.ct05=nn.Conv2d(self.C, self.C, 3, 1, 0)
		self.ct06=nn.Conv2d(self.C, self.C, 3, 1, 0)
		self.ct07=nn.Conv2d(self.C, self.C, 3, 1, 0)
		self.ct08=nn.Conv2d(self.C,      6, 3, 1, 0)

		#left part
		self.cl01=nn.Conv2d(     3, self.C, [1, 1], 1, 0)
		self.cl02=nn.Conv2d(self.C, self.C, [1, 2], 1, 0)
		self.cl03=nn.Conv2d(self.C, self.C, [1, 2], 1, 0)
		self.cl04=nn.Conv2d(self.C, self.C, [1, 2], 1, 0)
		self.cl05=nn.Conv2d(self.C, self.C, [1, 2], 1, 0)
		self.cl06=nn.Conv2d(self.C, self.C, [1, 2], 1, 0)
		self.cl07=nn.Conv2d(self.C, self.C, [1, 2], 1, 0)
		self.cl08=nn.Conv2d(self.C,      6, [1, 2], 1, 0)

		self.b01=nn.BatchNorm2d(self.C)
		self.b02=nn.BatchNorm2d(self.C)
		self.b03=nn.BatchNorm2d(self.C)
		self.b04=nn.BatchNorm2d(self.C)
		self.b05=nn.BatchNorm2d(self.C)
		self.b06=nn.BatchNorm2d(self.C)
		self.b07=nn.BatchNorm2d(self.C)

	def predict(self, x):
		truth=x
		
		#x=self.b01(self.act(self.ct01(self.padT(x))[:, :, :-1, :]))
		#x=self.b02(self.act(self.ct02(self.padT(x))))
		#x=self.b03(self.act(self.ct03(self.padT(x))))
		#x=self.b04(self.act(self.ct04(self.padT(x))))
		#x=self.b05(self.act(self.ct05(self.padT(x))))
		#x=self.b06(self.act(self.ct06(self.padT(x))))
		#x=self.b07(self.act(self.ct07(self.padT(x))))
		#x=         self.act(self.ct08(self.padT(x)))

		x=self.b01(self.act(self.ct01(self.padT(x))[:, :, :-1, :]+self.cl01(self.padL(x))[:, :, :, :-1]))
		x=self.b02(self.act(self.ct02(self.padT(x))+self.cl02(self.padL(x))))
		x=self.b03(self.act(self.ct03(self.padT(x))+self.cl03(self.padL(x))))
		x=self.b04(self.act(self.ct04(self.padT(x))+self.cl04(self.padL(x))))
		x=self.b05(self.act(self.ct05(self.padT(x))+self.cl05(self.padL(x))))
		x=self.b06(self.act(self.ct06(self.padT(x))+self.cl06(self.padL(x))))
		x=self.b07(self.act(self.ct07(self.padT(x))+self.cl07(self.padL(x))))
		x=         self.act(self.ct08(self.padT(x))+self.cl08(self.padL(x)))

		truth=truth[:, :, self.nlayers:, :]
		x=x[:, :, self.nlayers:, :]

		pred, conf=torch.split(x, 3, dim=1)
		return pred, conf, truth
