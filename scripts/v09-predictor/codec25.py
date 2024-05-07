import torch
from torch import nn
import math

#codec25: CUSTOM4 nonliear predictor

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def calc_RMSE(x):
	return 255*torch.sqrt(torch.mean(torch.square(x)))

def calc_invCR(x):
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

		self.reach=4
		self.npred=4
		self.conv0T=nn.Conv2d(3, 3*self.npred, (self.reach, self.reach<<1|1), bias=False)
		self.conv0L=nn.Conv2d(3, 3*self.npred, (1, self.reach))
		self.conv1=nn.Conv2d(3*self.npred, 3*self.npred, 1, groups=3)
		self.conv2=nn.Conv2d(3*self.npred, 3, 1, groups=3)

		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0
	def forward(self, x):
		bsize, c, h, w=x.shape
		r, g, b=torch.split(x, 1, dim=1)
		rg=r-g
		bg=b-g
		yuv=torch.cat((rg, g, bg), dim=1)
		pred=nn.functional.leaky_relu(self.conv0T(nn.functional.pad(yuv[:, :, :-1, :], (self.reach, self.reach, self.reach, 0))) + self.conv0L(nn.functional.pad(yuv[:, :, :, :-1], (self.reach, 0, 0, 0))))#(L, R, T, B)
		north=nn.functional.pad(yuv[:, :, :-1, :], (0, 0, 1, 0))
		west=nn.functional.pad(yuv[:, :, :, :-1], (1, 0, 0, 0))
		northeast=nn.functional.pad(yuv[:, :, :-1, 1:], (0, 1, 1, 0))
		vmin=torch.min(west, torch.min(north, northeast))
		vmax=torch.max(west, torch.max(north, northeast))
		pred=median3(pred, vmin, vmax)
		pcr, py, pcb=torch.split(pred, 1, dim=1)
		pcb=pcb+g
		pcr=pcr+g
		pred=torch.cat((pcr, py, pcb), dim=1)
		pred=torch.clamp(pred, -1, 1)
		delta=x-pred

		loss1=calc_RMSE(delta)

		with torch.no_grad():
			loss0=calc_RMSE(x).item()
			invCR0=calc_invCR(x)
			invCR1=calc_invCR(torch.fmod(delta+1, 2)-1)
		self.esum0+=loss0*bsize
		self.esum1+=loss1.item()*bsize
		self.csum0+=invCR0*bsize
		self.csum1+=invCR1*bsize
		self.count+=bsize
		return loss1, 'RMSE%8.4lf %8.4f  CR%6.4f %6.4f %6.4f'%(loss0, loss1.item(), safe_inv(invCR0), safe_inv(invCR1), safe_inv(invCR1)*invCR0)
	def epoch_start(self):
		self.esum0=0
		self.esum1=0
		self.csum0=0
		self.csum1=0
		self.count=0
	def epoch_end(self):#the returned string must be part of a valid filename
		invCR0=self.csum0/self.count
		invCR1=self.csum1/self.count
		rmse0=self.esum0/self.count
		rmse1=self.esum1/self.count
		return invCR1, 'RMSE%8.4lf %8.4f  CR%6.4f %6.4f %6.4f'%(rmse0, rmse1, safe_inv(invCR0), safe_inv(invCR1), safe_inv(invCR1)*invCR0)
	def checkpoint_msg(self):
		pass

