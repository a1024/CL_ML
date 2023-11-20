import torch
from torch import nn
import math

#codec16-20231120: super resolution-based parallel predictor

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


def deinterleave_x(x):
	return x[:, :, :, ::2], x[:, :, :, 1::2]

def deinterleave_y(x):
	return x[:, :, ::2, :], x[:, :, 1::2, :]

def join(ye, yo, xo):
	return torch.cat((torch.cat((ye, yo), dim=2), xo), dim=3)

class Predictor(nn.Module):
	def __init__(self, nch):#nch must end with 1
		super(Predictor, self).__init__()

		self.layers=nn.ModuleList()
		for kl in range(len(nch)-1):
			self.layers.add_module('conv%02d'%kl, nn.Conv2d(nch[kl], nch[kl+1], 3, 1, 1))	#3x3 conv
			#self.layers.add_module('conv%02d'%kl, nn.Conv2d(nch[kl], nch[kl+1], 1))	#1x1 conv
	def forward(self, x):
		nlayers=len(self.layers)
		for kl in range(nlayers-1):
			x=nn.functional.leaky_relu(self.layers.get_submodule('conv%02d'%kl)(x))
		return torch.clamp(self.layers.get_submodule('conv%02d'%(nlayers-1))(x), -1, 1)

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()

		#C16_01_1x1		1.6963@10

		#C16_02_3x3		
		self.pred_x=Predictor([1, 16, 16, 16, 1])
		self.pred_y=Predictor([1, 16, 16, 16, 1])

		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0

	def forward(self, x):
		b, c, h, w=x.shape
		x=x.view(b*c, 1, h, w)#a batch of channels, because of conv2d

		x1e, x1o=deinterleave_x(x)
		x1o=x1o-self.pred_x(x1e)
		y1e, y1o=deinterleave_y(x1e)
		y1o=y1o-self.pred_y(y1e)

		x2e, x2o=deinterleave_x(y1e)
		x2o=x2o-self.pred_x(x2e)
		y2e, y2o=deinterleave_y(x2e)
		y2o=y2o-self.pred_y(y2e)

		x3e, x3o=deinterleave_x(y2e)
		x3o=x3o-self.pred_x(x3e)
		y3e, y3o=deinterleave_y(x3e)
		y3o=y3o-self.pred_y(y3e)

		x4e, x4o=deinterleave_x(y3e)
		x4o=x4o-self.pred_x(x4e)
		y4e, y4o=deinterleave_y(x4e)
		y4o=y4o-self.pred_y(y4e)

		y3e=join(y4e, y4o, x4o)
		y2e=join(y3e, y3o, x3o)
		y1e=join(y2e, y2o, x2o)
		deltas=join(y1e, y1o, x1o)
		deltas=torch.fmod(deltas+1, 2)-1	#[-1, 1]

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
		return loss1, 'RMSE%9.4lf ->%9.4f  CR%9.4f ->%9.4f'%(loss0, loss1.item(), safe_inv(invCR0), safe_inv(invCR1))

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
