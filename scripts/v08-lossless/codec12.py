import torch
from torch import nn
import math

#codec12: causal predictor

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
	def __init__(self, reach, nlayers, nch):
		super(CausalConv, self).__init__()

		self.reach=reach
		self.conv00T=nn.Conv2d(1, nch, (reach, reach<<1|1))#(Kh, Kw)
		self.conv00L=nn.Conv2d(1, nch, (1, reach), bias=False)

		self.layers=nn.ModuleList()
		for kl in range(1, nlayers-1):
			self.layers.add_module('conv%02d'%kl, nn.Conv2d(nch, nch, 1))
		self.layers.add_module('conv%02d'%(nlayers-1), nn.Conv2d(nch, 1, 1))
	def forward(self, x):
		xt=self.conv00T(nn.functional.pad(x[:, :, :-1, :], (self.reach, self.reach, self.reach, 0)))#(L, R, T, B)
		xl=self.conv00L(nn.functional.pad(x[:, :, :, :-1], (self.reach, 0, 0, 0)))
		x=nn.functional.leaky_relu(xt+xl)

		nlayers=len(self.layers)
		for kl in range(1, nlayers-1):
			x=nn.functional.leaky_relu(self.layers.get_submodule('conv%02d'%kl)(x)+x)
		return torch.clamp(self.layers.get_submodule('conv%02d'%(nlayers-1))(x), -1, 1)

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()

		#self.pred01=CausalConv(2, 16, 12)	#C12_01		kodim13		1.88@100  1.95@100+110  1.95@100+110+100  1.93@100+110+200
		#self.pred01=CausalConv(2, 8, 32)	#C12_02		kodim13		1.96@20  1.97@20+10
		#self.pred01=CausalConv(3, 16, 32)	#C12_03		kodim13		1.94@20
		#self.pred01=CausalConv(4, 4, 16)	#C12_04		kodim13		1.95@30
		#self.pred01=CausalConv(7, 6, 64)	#C12_05		kodim13		1.86@10  1.95@20  1.95@30
		#self.pred01=CausalConv(7, 8, 48)	#C12_06		kodim13		1.87@5*167056
		#self.pred01=CausalConv(2, 32, 32)	#C12_07		kodim13		1.93@10
		self.pred01=CausalConv(2, 16, 32)	#C12_07		kodim13		?

		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0

	def forward(self, x):
		b, c, h, w=x.shape
		x=x.view(b*c, 1, h, w)#a batch of channels, because of conv2d

		deltas=torch.fmod( x-self.pred01(x) +1, 2)-1

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
