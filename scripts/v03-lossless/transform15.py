import torch
from torch import nn

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

class DownBlock(nn.Module):
	def __init__(self, ci, cb, co, activ):
		super(DownBlock, self).__init__()
		self.activ=activ
		self.shortcut=nn.Conv2d(ci, co, 3, 2, 1)#downsample
		self.conv1=nn.Conv2d(ci, cb, 1, 1, 0)
		self.conv2=nn.Conv2d(cb, cb, 3, 2, 1)	#downsample
		self.conv3=nn.Conv2d(cb, co, 1, 1, 0)
		self.conv4=nn.Conv2d(co, cb, 1, 1, 0)
		self.conv5=nn.Conv2d(cb, cb, 3, 1, 1)
		self.conv6=nn.Conv2d(cb, co, 1, 1, 0)

	def forward(self, x):
		t=self.shortcut(x)
		x=self.activ(self.conv1(x))
		x=self.activ(self.conv2(x))
		x=self.activ(self.conv3(x))
		x=self.activ(self.conv4(x))
		x=self.activ(self.conv5(x))
		x=self.activ(self.conv6(x)+t)
		return x

class UpBlock(nn.Module):
	def __init__(self, ci, cb, co, activ):#dimensions must be powers of two
		super(UpBlock, self).__init__()
		self.activ=activ
		self.shortcut=nn.ConvTranspose2d(ci, co, 3, 2, 1, 1)	#upsample
		self.conv1=nn.Conv2d(ci, cb, 1, 1, 0)
		self.conv2=nn.Conv2d(cb, cb, 3, 1, 1)
		self.conv3=nn.Conv2d(cb, co, 1, 1, 0)
		self.conv4=nn.Conv2d(co, cb, 1, 1, 0)
		self.conv5=nn.ConvTranspose2d(cb, cb, 3, 2, 1, 1)	#upample
		self.conv6=nn.Conv2d(cb, co, 1, 1, 0)

	def forward(self, x):
		t=self.shortcut(x)
		x=self.activ(self.conv1(x))
		x=self.activ(self.conv2(x))
		x=self.activ(self.conv3(x))
		x=self.activ(self.conv4(x))
		x=self.activ(self.conv5(x))
		x=self.activ(self.conv6(x)+t)
		return x

class NLTransform(nn.Module):
	def __init__(self):
		super(NLTransform, self).__init__()
		self.activ=nn.LeakyReLU()
		self.dnsmp=nn.AvgPool2d(2)
		self.upsmp=nn.Upsample(scale_factor=2)

		#T15: best-practices
		self.nOptions=4

		self.a_b1=DownBlock(3+self.nOptions, 16, 64, self.activ)
		self.a_b2=DownBlock(64, 16, 128, self.activ)
		self.a_b3=DownBlock(128, 32, 256, self.activ)
		self.a_b4=DownBlock(256, 32, 256, self.activ)

		self.s_b1=UpBlock(256+self.nOptions, 32, 256, self.activ)
		self.s_b2=UpBlock(256, 32, 128, self.activ)
		self.s_b3=UpBlock(128, 16, 64, self.activ)
		self.s_b4=UpBlock(64, 16, 3, self.activ)

	def prep_input(self, x, option, device):
		rateinfo=torch.empty(1).fill_(option).long()
		rateinfo=nn.functional.one_hot(rateinfo, self.nOptions).float().to(device)
		rateinfo=rateinfo[None, None]
		rateinfo=rateinfo.expand(x.shape[0], x.shape[2], x.shape[3], -1)
		rateinfo=rateinfo.transpose(1, 3)
		x=torch.cat((x, rateinfo), dim=1)
		return x

	def encode(self, x, option, device):# no nonlinearity after last operation
		x=self.prep_input(x, option, device)
		
		x=self.a_b1(x)
		x=self.a_b2(x)
		x=self.a_b3(x)
		x=self.a_b4(x)
		return x

	def decode(self, x, option, device):# clamp [0, 1] after last operation
		x=self.prep_input(x, option, device)
		
		x=self.s_b1(x)
		x=self.s_b2(x)
		x=self.s_b3(x)
		x=self.s_b4(x)

		x=torch.clamp(x, min=0, max=1)
		return x
