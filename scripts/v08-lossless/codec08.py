import torch
from torch import nn
import math

#codec07: parallel block predictor TODO aux buffers

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def calc_RMSE(x):
	return 255*torch.sqrt(torch.mean(torch.square(x)))

def calc_invCR(x):
	entropy=0
	b, c, h, w=x.shape
	nch=b*c; res=w*h
	x=x.view(nch, res)
	for k in range(nch):
		prob=torch.histc(x[k, :], 256, -1, 1)/res
		entropy+=torch.sum(-prob*torch.nan_to_num(torch.log2(prob), 0, 0, 0)).item()
	return entropy/(8*nch)

def safe_inv(x):
	if x!=0:
		return 1/x
	return math.inf

class PixelPred(nn.Module):
	def __init__(self, nch):
		super(PixelPred, self).__init__()
		self.layers=nn.ModuleList()
		for kl in range(1, len(nch)):
			self.layers.add_module('dense%02d'%(kl-1), nn.Linear(nch[kl-1], nch[kl]))
	def forward(self, x):
		nlayers=len(self.layers)
		for kl in range(nlayers-1):
			x=nn.functional.leaky_relu(self.layers.get_submodule('dense%02d'%kl)(x))
		return torch.clamp(self.layers.get_submodule('dense%02d'%(nlayers-1))(x), -1, 1)

class Predictor(nn.Module):
	def __init__(self, blocksize):
		super(Predictor, self).__init__()

		self.blocksize=blocksize
		blockarea=blocksize*blocksize
		self.preds=nn.ModuleList()
		for kp in range(2, blockarea):
			nch=[(kp<<1)-1]
			k=kp
			while k>=1:
				nch.append(k)
				k>>=1
			self.preds.add_module('pred%02d'%kp, PixelPred(nch))
	def forward(self, x):#[B, 1, NB, 64]
		errors=x[:, :, :, 1:2]-x[:, :, :, 0:1]
		k=2
		for pred in self.preds:
			delta=x[:, :, :, k:k+1]-pred(torch.cat((x[:, :, :, :k], errors), dim=3))
			delta=torch.fmod(delta+1, 2)-1		#[-1, 1]
			errors=torch.cat((errors, delta), dim=3)
			k+=1
		return errors

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()

		self.blocksize=4

		self.pred_c0=Predictor(self.blocksize)
		self.pred_c1=Predictor(self.blocksize)
		self.pred_c2=Predictor(self.blocksize)

		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0

	def forward(self, x):
		b, c, h, w=x.shape
		res=w*h
		blockarea=self.blocksize*self.blocksize
		nblocks=res//blockarea
		x=x.unfold(2, self.blocksize, self.blocksize).unfold(3, self.blocksize, self.blocksize).reshape(b, c, nblocks, blockarea)
		c0, c1, c2=torch.split(x, 1, dim=1)
		e0=self.pred_c0(c0)
		e1=self.pred_c1(c1)
		e2=self.pred_c2(c2)
		deltas=torch.cat((e0, e1, e2), dim=1)

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
