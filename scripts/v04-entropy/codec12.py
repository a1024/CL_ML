import torch
from torch import nn

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

class GenBlock(nn.Module):
	def __init__(self, ci, cb, co, resample, shortcut):#resample: -1: downsample, 0: leave as is, 1: upsample
		super(GenBlock, self).__init__()
		self.act=nn.LeakyReLU()

		self.bn=nn.BatchNorm2d(ci)

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
		x=self.bn(x)
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
		self.upsample=nn.Upsample(scale_factor=2, mode='bilinear')

		self.W=96
		self.B=32
		self.O=8

		self.a_b01=GenBlock(     3,        self.B, self.W, 0, 1)
		self.a_b02=GenBlock(self.W,        self.B, self.W, 0, 1)
		self.a_b03=GenBlock(self.W,        self.B, self.W, 0, 1)
		self.a_b04=GenBlock(self.W,        self.B, self.W, -1, 1)	#[B, 3, 32, 32]
		self.a_b05=GenBlock(self.W-self.O, self.B, self.W, 0, 1)
		self.a_b06=GenBlock(self.W,        self.B, self.W, 0, 1)
		self.a_b07=GenBlock(self.W,        self.B, self.W, 0, 1)
		self.a_b08=GenBlock(self.W,        self.B, self.W, -1, 1)	#[B, 3, 16, 16]
		self.a_b09=GenBlock(self.W-self.O, self.B, self.W, 0, 1)
		self.a_b10=GenBlock(self.W,        self.B, self.W, 0, 1)
		self.a_b11=GenBlock(self.W,        self.B, self.W, 0, 1)
		self.a_b12=GenBlock(self.W,        self.B, self.W, -1, 1)	#[B, 3, 8, 8]
		self.a_b13=GenBlock(self.W-self.O, self.B, self.W, 0, 1)
		self.a_b14=GenBlock(self.W,        self.B, self.W, 0, 1)
		self.a_b15=GenBlock(self.W,        self.B, self.W, 0, 1)
		self.a_b16=GenBlock(self.W,        self.B, self.O, -1, 1)	#[B, 3, 4, 4]

		self.s_b01=GenBlock(self.O, self.B, self.W,        0, 1)	#+3*4->8
		self.s_b02=GenBlock(self.W, self.B, self.W,        0, 1)
		self.s_b03=GenBlock(self.W, self.B, self.W,        0, 1)
		self.s_b04=GenBlock(self.W, self.B, self.W-self.O, 0, 1)
		self.s_b05=GenBlock(self.W, self.B, self.W,        0, 1)	#+3*8->16
		self.s_b06=GenBlock(self.W, self.B, self.W,        0, 1)
		self.s_b07=GenBlock(self.W, self.B, self.W,        0, 1)
		self.s_b08=GenBlock(self.W, self.B, self.W-self.O, 0, 1)
		self.s_b09=GenBlock(self.W, self.B, self.W,        0, 1)	#+3*16->32
		self.s_b10=GenBlock(self.W, self.B, self.W,        0, 1)
		self.s_b11=GenBlock(self.W, self.B, self.W,        0, 1)
		self.s_b12=GenBlock(self.W, self.B, self.W-self.O, 0, 1)
		self.s_b13=GenBlock(self.W, self.B, self.W,        0, 1)	#+3*32->64
		self.s_b14=GenBlock(self.W, self.B, self.W,        0, 1)
		self.s_b15=GenBlock(self.W, self.B, self.W,        0, 1)
		self.s_b16=GenBlock(self.W, self.B, self.W,        0, 1)
		self.s_b17=GenBlock(self.W, self.B,      3,        0, 1)

	def encode(self, x):
		x=self.a_b01(x)
		x=self.a_b02(x)
		x=self.a_b03(x)
		x=self.a_b04(x)
		x, a=torch.split(x, self.W-self.O, dim=1)	#[B, 1, 32, 32]
		x=self.a_b05(x)
		x=self.a_b06(x)
		x=self.a_b07(x)
		x=self.a_b08(x)
		x, b=torch.split(x, self.W-self.O, dim=1)	#[B, 1, 16, 16]
		x=self.a_b09(x)
		x=self.a_b10(x)
		x=self.a_b11(x)
		x=self.a_b12(x)
		x, c=torch.split(x, self.W-self.O, dim=1)	#[B, 1, 16, 16]
		x=self.a_b13(x)
		x=self.a_b14(x)
		x=self.a_b15(x)
		x=self.a_b16(x)				#[B, 3, 8, 8]
		return [a, b, c, x]

	def decode(self, y):
		a, b, c, x=y
		x=self.upsample(x)
		x=self.s_b01(x)
		x=self.s_b02(x)
		x=self.s_b03(x)
		x=self.s_b04(x)
		x=torch.cat((x, c), dim=1)
		x=self.upsample(x)
		x=self.s_b05(x)
		x=self.s_b06(x)
		x=self.s_b07(x)
		x=self.s_b08(x)
		x=torch.cat((x, b), dim=1)
		x=self.upsample(x)
		x=self.s_b09(x)
		x=self.s_b10(x)
		x=self.s_b11(x)
		x=self.s_b12(x)
		x=torch.cat((x, a), dim=1)
		x=self.upsample(x)
		x=self.s_b13(x)
		x=self.s_b14(x)
		x=self.s_b15(x)
		x=self.s_b16(x)
		x=self.s_b17(x)
		return x
