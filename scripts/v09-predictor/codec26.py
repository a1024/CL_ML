import torch
from torch import nn
import math

#codec26: """inspired""" by LLICTI, jointly predicts interleaved YUV channels		2024-05-06Mo

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def rct(x):#RCT_JPEG2000
	r, g, b=torch.split(x, 1, 1)
	r=r-g
	b=b-g
	g=g+(r+b)*0.25
	x=torch.cat((r, g, b), 1)
	return x

def calc_RMSE(x):
	return 255*torch.sqrt(torch.mean(torch.square(x)))

def calc_invCR(x):
	x=torch.fmod(x+1, 2)-1
	entropy=0
	b, c, h, w=x.shape
	nch=b*c; res=w*h
	x=x.reshape(nch, res)
	for k in range(nch):
		prob=torch.histc(x[k, :], 256, -1, 1)/res
		entropy+=torch.sum(-prob*torch.nan_to_num(torch.log2(prob), 0, 0, 0)).item()
	return entropy/(8*nch)

def safe_inv(x):
	if x!=0:
		return 1/x
	return math.inf

class PixelPred(nn.Module):
	def __init__(self, nlayers, ci, nch, co):
		super(PixelPred, self).__init__()
		self.layers=nn.ModuleList()
		ninputs=ci
		kl=0
		while kl<nlayers-1:
			self.layers.add_module('dense%02d'%kl, nn.Linear(ninputs, nch))
			ninputs=nch
			kl+=1
		self.layers.add_module('dense%02d'%kl, nn.Linear(nch, co))
	def forward(self, x):
		nlayers=len(self.layers)
		for kl in range(nlayers-1):
			x=nn.functional.leaky_relu(self.layers.get_submodule('dense%02d'%kl)(x))
		return torch.clamp(self.layers.get_submodule('dense%02d'%(nlayers-1))(x), -1, 1)

class CausalConv(nn.Module):
	def __init__(self, reach, nch):#nch must end with 1
		super(CausalConv, self).__init__()

		self.reach=reach
		self.conv00T=nn.Conv2d(1, nch[0], (reach, reach<<1|1))#(Kh, Kw)
		self.conv00L=nn.Conv2d(1, nch[0], (1, reach), bias=False)

		self.layers=nn.ModuleList()
		for kl in range(len(nch)-1):
			self.layers.add_module('conv%02d'%kl, nn.Conv2d(nch[kl], nch[kl+1], 1))
	def forward(self, x):
		xt=self.conv00T(nn.functional.pad(x[:, :, :-1, :], (self.reach, self.reach, self.reach, 0)))#(L, R, T, B)
		xl=self.conv00L(nn.functional.pad(x[:, :, :, :-1], (self.reach, 0, 0, 0)))
		x=nn.functional.leaky_relu(xt+xl)

		nlayers=len(self.layers)
		for kl in range(0, nlayers-1):
			x=nn.functional.leaky_relu(self.layers.get_submodule('conv%02d'%kl)(x))
		return torch.clamp(self.layers.get_submodule('conv%02d'%(nlayers-1))(x), -1, 1)

def median3(a, b, c):
	return torch.max(torch.min(a, torch.max(b, c)), torch.min(b, c))

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()

		#deinterleave image per blocks size 2x2
  		#predictors (A, B, C) predict (SE, NE, SW) respectively
		self.npreds=4
		nch=3*self.npreds
		self.convA0a=nn.Conv2d(3, nch, 4)
		self.convB0a=nn.Conv2d(3, nch, (3, 4), bias=False)
		self.convB0b=nn.Conv2d(3, nch, (4, 3))
		self.convC0a=nn.Conv2d(3, nch, (4, 3), bias=False)
		self.convC0b=nn.Conv2d(3, nch, (3, 4), bias=False)
		self.convC0c=nn.Conv2d(3, nch, 4)

		self.convA1=nn.Conv2d(nch, nch, 3, padding=1)
		self.convA2=nn.Conv2d(nch, nch, 3, padding=1)
		self.convA3=nn.Conv2d(nch, nch, 3, padding=1)
		self.convA4=nn.Conv2d(nch, nch, 3, padding=1)
		self.convA5=nn.Conv2d(nch, nch, 3, padding=1)
		self.convA6=nn.Conv2d(nch, nch, 3, padding=1)
		self.convA7=nn.Conv2d(nch,   3, 3, padding=1)

		self.convB1=nn.Conv2d(nch, nch, 3, padding=1)
		self.convB2=nn.Conv2d(nch, nch, 3, padding=1)
		self.convB3=nn.Conv2d(nch, nch, 3, padding=1)
		self.convB4=nn.Conv2d(nch, nch, 3, padding=1)
		self.convB5=nn.Conv2d(nch, nch, 3, padding=1)
		self.convB6=nn.Conv2d(nch, nch, 3, padding=1)
		self.convB7=nn.Conv2d(nch,   3, 3, padding=1)

		self.convC1=nn.Conv2d(nch, nch, 3, padding=1)
		self.convC2=nn.Conv2d(nch, nch, 3, padding=1)
		self.convC3=nn.Conv2d(nch, nch, 3, padding=1)
		self.convC4=nn.Conv2d(nch, nch, 3, padding=1)
		self.convC5=nn.Conv2d(nch, nch, 3, padding=1)
		self.convC6=nn.Conv2d(nch, nch, 3, padding=1)
		self.convC7=nn.Conv2d(nch,   3, 3, padding=1)

		#self.convA0a=nn.Conv2d(3, 3, 4, 2, 3, 2)
		#self.convB0a=nn.Conv2d(3, 3, (3, 4), 2, (2, 3), 2)
		#self.convB0b=nn.Conv2d(3, 3, (4, 3), 2, (3, 2), 2)
		#self.convC0a=nn.Conv2d(3, 3, (4, 3))

		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR
		self.csum1=0#invCR
		self.csum2=0#invCR
		self.count=0
	def forward(self, x):
		bsize, c, h, w=x.shape

		x=rct(x)
		nw=x[:, :, 0::2, 0::2]#blue circle
		ne=x[:, :, 0::2, 1::2]#green rhomb
		sw=x[:, :, 1::2, 0::2]#black pentagon
		se=x[:, :, 1::2, 1::2]#maroon square
		t=nn.functional.leaky_relu(self.convA0a(nn.functional.pad(nw, (1, 2, 1, 2))))
		preda=nn.functional.leaky_relu(self.convA1(t))
		preda=nn.functional.leaky_relu(self.convA2(preda))
		preda=nn.functional.leaky_relu(self.convA3(preda))
		preda=nn.functional.leaky_relu(self.convA4(preda))
		preda=nn.functional.leaky_relu(self.convA5(preda))
		preda=nn.functional.leaky_relu(self.convA6(preda+t))
		preda=se-torch.clamp(self.convA7(preda), -1, 1)

		t=nn.functional.leaky_relu(self.convB0a(nn.functional.pad(nw, (1, 2, 1, 1)))+self.convB0b(nn.functional.pad(se, (1, 1, 2, 1))))
		predb=nn.functional.leaky_relu(self.convB1(t))
		predb=nn.functional.leaky_relu(self.convB2(predb))
		predb=nn.functional.leaky_relu(self.convB3(predb))
		predb=nn.functional.leaky_relu(self.convB4(predb))
		predb=nn.functional.leaky_relu(self.convB5(predb))
		predb=nn.functional.leaky_relu(self.convB6(predb+t))
		predb=ne-torch.clamp(self.convB7(predb), -1, 1)

		t=nn.functional.leaky_relu(self.convC0a(nn.functional.pad(nw, (1, 1, 1, 2)))+self.convC0b(nn.functional.pad(se, (2, 1, 1, 1)))+self.convC0c(nn.functional.pad(ne, (2, 1, 1, 2))))
		predc=nn.functional.leaky_relu(self.convC1(t))
		predc=nn.functional.leaky_relu(self.convC2(predc))
		predc=nn.functional.leaky_relu(self.convC3(predc))
		predc=nn.functional.leaky_relu(self.convC4(predc))
		predc=nn.functional.leaky_relu(self.convC5(predc))
		predc=nn.functional.leaky_relu(self.convC6(predc+t))
		predc=sw-torch.clamp(self.convC7(predc), -1, 1)

		delta=torch.cat((preda, predb, predc), 1)
		loss1=calc_RMSE(delta)

		with torch.no_grad():
			x=torch.cat((se, ne, sw), 1)
			loss0=calc_RMSE(x).item()
			invCR0=calc_invCR(preda)
			invCR1=calc_invCR(predb)
			invCR2=calc_invCR(predc)
		self.esum0+=loss0*bsize
		self.esum1+=loss1.item()*bsize
		self.csum0+=invCR0*bsize
		self.csum1+=invCR1*bsize
		self.csum2+=invCR2*bsize
		self.count+=bsize
		return loss1, 'RMSE%8.4lf %8.4f  iCR%6.4f %6.4f %6.4f'%(loss0, loss1.item(), invCR0, invCR1, invCR2)
	def epoch_start(self):
		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR
		self.csum1=0#invCR
		self.csum2=0#invCR
		self.count=0
	def epoch_end(self):#the returned string must be part of a valid filename
		invCR0=self.csum0/self.count
		invCR1=self.csum1/self.count
		invCR2=self.csum2/self.count
		rmse0=self.esum0/self.count
		rmse1=self.esum1/self.count
		return invCR1, 'RMSE%8.4lf %8.4f  iCR%6.4f %6.4f %6.4f'%(rmse0, rmse1, invCR0, invCR1, invCR2)
	def checkpoint_msg(self):
		pass

