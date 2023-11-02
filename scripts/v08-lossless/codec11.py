import torch
from torch import nn
import math

#codec11: causal pixel predictor with aux buffers (C06 with different hyperparams)

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def get_nb(x, reach, kx, ky):
	n, h, w=x.shape
	return torch.cat((x[:, ky-reach:ky, kx-reach:kx+reach+1].reshape(n, -1), x[:, ky, kx-reach:kx].reshape(n, -1)), dim=1)

def calc_RMSE(x):
	return 255*torch.sqrt(torch.mean(torch.square(x)))

def calc_invCR(x):
	entropy=0
	nch, res=x.shape
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

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()

		self.reach=2		#pixel reach > error/aux reach
		self.nlayers=16

		self.blocksize=32	#just to improve train time

		self.ci=2*(self.reach+1)*self.reach
		self.nch=self.ci
		self.co=1

		self.pred01=PixelPred(self.nlayers, self.ci, self.nch, self.co)

		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0

	def forward(self, x):
		b, c, h, w=x.shape
		h=h//self.blocksize*self.blocksize
		w=w//self.blocksize*self.blocksize
		x=x[:, :, :h, :w]

		res=w*h
		blockarea=self.blocksize*self.blocksize
		nblocks=res//blockarea
		nbc=b*c*nblocks

		x=x.unfold(2, self.blocksize, self.blocksize).unfold(3, self.blocksize, self.blocksize).reshape(nbc, self.blocksize, self.blocksize)
		xp=nn.functional.pad(x, (self.reach, self.reach, self.reach, self.reach))

		deltas=torch.zeros(nbc, 0, self.blocksize, dtype=x.dtype, device=x.device)
		for ky in range(self.blocksize):
			row=torch.zeros(nbc, 1, 0, dtype=x.dtype, device=x.device)
			for kx in range(self.blocksize):
				nb=get_nb(xp, self.reach, self.reach+kx, self.reach+ky)

				delta=x[:, ky:ky+1, kx:kx+1]-self.pred01(nb).view(nbc, 1, 1)
				delta=torch.fmod( delta +1, 2)-1	#[-1, 1]

				row=torch.cat((row, delta.view(nbc, 1, 1)), dim=2)
			deltas=torch.cat((deltas, row), dim=1)

		x=x.view(b*c, res)
		deltas=deltas.view(b*c, res)

		loss1=calc_RMSE(deltas)

		with torch.no_grad():
			loss0=calc_RMSE(x).item()
			invCR0=calc_invCR(x)
			invCR1=calc_invCR(deltas)
		self.esum0+=loss0*b
		self.esum1+=loss1.item()*b
		self.csum0+=invCR0*b
		self.csum1+=invCR1*b
		self.count+=b
		return loss1, 'RMSE%7.2lf ->%7.2f  CR%7.2f ->%7.2f'%(loss0, loss1.item(), safe_inv(invCR0), safe_inv(invCR1))

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
		return invCR1, 'RMSE%9.4lf %9.4f  CR%9.4f %9.4f'%(rmse0, rmse1, safe_inv(invCR0), safe_inv(invCR1))
	def checkpoint_msg(self):
		pass
