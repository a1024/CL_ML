import torch
from torch import nn

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

class GenBlock(nn.Module):
	def __init__(self, ci, cb, co, resample, shortcut):#resample: -1: downsample, 0: leave as is, 1: upsample
		super(GenBlock, self).__init__()
		self.act=nn.LeakyReLU()

		if shortcut:
			if resample==-1:
				self.shortcut=nn.Conv2d(ci, co, 3, 2, 1)#downsample
			elif resample==1:
				self.shortcut=nn.ConvTranspose2d(ci, co, 3, 2, 1, 1)#upsample
			else:
				self.shortcut=nn.Conv2d(ci, co, 1, 1, 0)
		else:
			self.shortcut=None


		self.conv1=nn.Conv2d(ci, cb, 1, 1, 0)

		if resample==-1:
			self.conv2=nn.Conv2d(cb, cb, 3, 2, 1)#downsample
		elif resample==1:
			self.conv2=nn.ConvTranspose2d(cb, cb, 3, 2, 1, 1)#upample
		else:
			self.conv2=nn.Conv2d(cb, cb, 3, 1, 1)#else

		self.conv3=nn.Conv2d(cb, co, 1, 1, 0)

	def forward(self, x):
		if self.shortcut is not None:
			t=self.act(self.shortcut(x))
		x=self.act(self.conv1(x))
		x=self.act(self.conv2(x))
		x=self.conv3(x)
		if self.shortcut is not None:
			x=torch.add(x, t)
		x=self.act(x)
		return x

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()

		self.sc01=  2
		self.sc02=  8
		self.sc03= 32
		self.sc04=128

		self.a_b01=GenBlock(  3, 64,  64, -1, 1)
		self.a_b02=GenBlock( 64, 32, 128, -1, 1)
		self.a_b03=GenBlock(128, 32, 256, -1, 1)
		self.a_b04=GenBlock(256, 32, self.sc04, -1, 1)

		self.s_b01=GenBlock(self.sc04, 32, 256-self.sc03//2, 1, 1)
		self.s_b02=GenBlock(256+self.sc03//2, 32, 128-self.sc02//2, 1, 1)
		self.s_b03=GenBlock(128+self.sc02//2, 32,  64-self.sc01//2, 1, 1)
		self.s_b04=GenBlock( 64+self.sc01//2, 64,   3, 1, 1)

	def encode(self, x):
		x=self.a_b01(x)
		a=x[:, :self.sc01, :, :]	#[B, 1, 32, 32]
		x=self.a_b02(x)
		b=x[:, :self.sc02, :, :]	#[B, 4, 16, 16]
		x=self.a_b03(x)
		c=x[:, :self.sc03, :, :]	#[B, 64, 8, 8]
		x=self.a_b04(x)			#[B, 256, 4, 4]
		return [a, b, c, x]

	def decode(self, y):
		a, b, c, x=y[0], y[1], y[2], y[3]
		x=self.s_b01(x)
		x=torch.cat((c, x), dim=1)
		x=self.s_b02(x)
		x=torch.cat((b, x), dim=1)
		x=self.s_b03(x)
		x=torch.cat((a, x), dim=1)
		x=self.s_b04(x)
		return x
