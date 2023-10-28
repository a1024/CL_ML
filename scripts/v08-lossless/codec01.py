#2023-09-25
import torch
from torch import nn

#codec01: L3C-Pytorch

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

class Block(nn.Module):
	def __init__(self, nch):
		super(Block, self).__init__()
		self.conv01=nn.Conv2d(nch, nch, 3, stride=1, padding=1)
		self.conv02=nn.Conv2d(nch, nch, 3, stride=1, padding=1)
	def forward(self, x):
		return nn.functional.leaky_relu(self.conv02(nn.functional.leaky_relu(self.conv01(x)))+x)

class Encoder(nn.Module):
	def __init__(self, cin):
		super(Encoder, self).__init__()
		self.conv00=nn.Conv2d(cin, 64, 5, stride=2, padding=2)

		self.block00=Block(64)
		self.block01=Block(64)
		self.block02=Block(64)
		self.block03=Block(64)
		self.block04=Block(64)
		self.block05=Block(64)
		self.block06=Block(64)
		self.block07=Block(64)
		self.conv01=nn.Conv2d(64, 64, 3, stride=1, padding=1)

		self.conv02=nn.Conv2d(64, 64, 3, stride=1, padding=1)
		self.conv03=nn.Conv2d(64, 5, 1, stride=1, padding=0)
	def forward(self, x):
		x=self.conv00(x)
		t=self.block00(x)
		t=self.block01(t)
		t=self.block02(t)
		t=self.block03(t)
		t=self.block04(t)
		t=self.block05(t)
		t=self.block06(t)
		t=self.block07(t)
		x=self.conv01(t)+x
		next=self.conv02(x)
		qnext=self.conv03(x)
		#qnext=qnext+(torch.rand(qnext.shape, device=qnext.device)-0.5)*(2/65536)
		qnext=torch.clamp(qnext, -1, 1)
		return next, qnext

class Decoder(nn.Module):
	def __init__(self, cout):
		super(Decoder, self).__init__()
		self.conv00=nn.Conv2d(5, 64, 1, stride=1, padding=0)

		self.block00=Block(64)
		self.block01=Block(64)
		self.block02=Block(64)
		self.block03=Block(64)
		self.block04=Block(64)
		self.block05=Block(64)
		self.block06=Block(64)
		self.block07=Block(64)
		self.conv01=nn.Conv2d(64, 64, 3, stride=1, padding=1)

		self.conv02=nn.Conv2d(64, 256, 3, stride=1, padding=1)
		self.upsample=nn.PixelShuffle(2)
		self.conv03a=nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1)
		self.conv03b=nn.Conv2d(64, 64, 3, stride=1, padding=2, dilation=2)
		self.conv03c=nn.Conv2d(64, 64, 3, stride=1, padding=4, dilation=4)
		self.conv04=nn.Conv2d(64, cout, 1)
	def forward(self, z, f):
		x=self.conv00(z)+f

		t=self.block00(x)
		t=self.block01(t)
		t=self.block02(t)
		t=self.block03(t)
		t=self.block04(t)
		t=self.block05(t)
		t=self.block06(t)
		t=self.block07(t)
		x=self.conv01(t)+x

		x=self.conv02(x)
		x=self.upsample(x)
		a=self.conv03a(x)
		b=self.conv03b(x)
		c=self.conv03c(x)
		f=a+b+c
		#f=torch.cat((a, b, c), dim=1)
		p=self.conv04(f)
		return f, p

def calc_csize(x, binsize, mean, conf, wmix):
	loss=torch.sigmoid((x+binsize/2-mean)*conf)-torch.sigmoid((x-binsize/2-mean)*conf)
	den=torch.sigmoid((1+binsize/2-mean)*conf)-torch.sigmoid((-1-binsize/2-mean)*conf)
	den=torch.clamp(den, 0.001, None)
	loss=torch.sum(wmix*loss/den, dim=3)	#combined probability
	loss=-torch.log2(loss)			#csize
	loss=loss.sum()
	return loss

def calc_csize_x(rgb, params, K):
	binsize=2/256
	c=3
	rgb=torch.reshape(rgb, [rgb.shape[0], rgb.shape[1], rgb.shape[2]*rgb.shape[3]])	#[B, 3, H*W]
	rgb=torch.transpose(rgb, 1, 2)							#[B, H*W, 3]
	rgb=torch.unsqueeze(rgb, -1).repeat(1, 1, 1, K)					#[B, H*W, 3, K]

	params=torch.reshape(params, [params.shape[0], params.shape[1], params.shape[2]*params.shape[3]])	#[B, 12*K, H*W]
	params=torch.transpose(params, 1, 2)									#[B, H*W, 12*K]
	params=torch.reshape(params, [params.shape[0], params.shape[1], params.shape[2]//K, K])			#[B, H*W, 12, K]

	wmix, mean, conf, lamb=torch.split(params, c, dim=2)							#[B, H*W, 3, K]
	wmix=torch.softmax(wmix, dim=3)
	mean=torch.clamp(mean, -1, 1)
	conf=torch.clamp(conf.abs(), 1, 100)
	alpha, beta, gamma=torch.split(lamb, 1, dim=2)	#[B, H*W, 1, K]
	wmix0, wmix1, wmix2=torch.split(wmix, 1, dim=2)
	mean0, mean1, mean2=torch.split(mean, 1, dim=2)
	conf0, conf1, conf2=torch.split(conf, 1, dim=2)
	r, g, b=torch.split(rgb, 1, dim=2)
	return calc_csize(r, binsize, mean0, conf0, wmix0) + calc_csize(g, binsize, mean1+alpha*r, conf1, wmix1) + calc_csize(b, binsize, mean2+beta*r+gamma*g, conf2, wmix2)

def calc_csize_z(z, params, K):#image has 5 channels, params have 15*K channels
	binsize=2/25	#paper says b=1/12
	c=5
	z=torch.reshape(z, [z.shape[0], z.shape[1], z.shape[2]*z.shape[3]])		#[B, 5, H*W]
	z=torch.transpose(z, 1, 2)							#[B, H*W, 5]
	z=torch.unsqueeze(z, -1).repeat(1, 1, 1, K)					#[B, H*W, 5, K]

	params=torch.reshape(params, [params.shape[0], params.shape[1], params.shape[2]*params.shape[3]])	#[B, 15*K, H*W]
	params=torch.transpose(params, 1, 2)									#[B, H*W, 15*K]
	params=torch.reshape(params, [params.shape[0], params.shape[1], params.shape[2]//K, K])			#[B, H*W, 15, K]

	wmix, mean, conf=torch.split(params, c, dim=2)								#[B, H*W, 5, K]
	wmix=torch.softmax(wmix, dim=3)
	mean=torch.clamp(mean, -1, 1)
	conf=torch.clamp(conf.abs(), 1, 100)
	return calc_csize(z, binsize, mean, conf, wmix)

class Codec(nn.Module):				#L3C-Pytorch
	def __init__(self):
		super(Codec, self).__init__()
		self.usize=0
		self.csize=0

		#loss_func=nn.CrossEntropyLoss()	#X

		self.K=10	# number of estimators

		self.enc01=Encoder(3)
		self.enc02=Encoder(64)
		self.enc03=Encoder(64)
		self.enc04=Encoder(64)

		self.dec04=Decoder(15*self.K)
		self.dec03=Decoder(15*self.K)
		self.dec02=Decoder(15*self.K)
		self.dec01=Decoder(12*self.K)
	def forward(self, x):
		e1, z1=self.enc01(x)
		e2, z2=self.enc02(e1)
		e3, z3=self.enc03(e2)#z3 is bypass (uniform probability)

		f4=torch.zeros(z3.shape[0], 64, z3.shape[2], z3.shape[3], dtype=z3.dtype, device=z3.device)
		f3, p3=self.dec03(z3, f4)
		f2, p2=self.dec02(z2, f3)
		f1, p1=self.dec01(z1, f2)

		size3=z3.nelement()
		size2=calc_csize_z(z2, p3, self.K)
		size1=calc_csize_z(z1, p2, self.K)
		size0=calc_csize_x(x, p1, self.K)
		size = size3 + size2 + size1 + size0
		invCR=size/(x.nelement()*8)

		self.usize+=x.nelement()
		self.csize+=x.nelement()*invCR.item()
		return invCR, 'U%14.2f  C%14.2f  CR%10f'%(x.nelement(), x.nelement()*invCR.item(), 1/invCR.item())

	def epoch_start(self):
		self.usize=0
		self.csize=0
	def epoch_end(self):
		return self.csize/self.usize, '%14f'%(self.usize/self.csize)#invCR
	def checkpoint_msg(self):
		pass
