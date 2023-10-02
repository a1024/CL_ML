import torch
from torch import nn
#import math

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def pred_clampedgrad(x):
	n=nn.functional.pad(x[:, :, :-1, :], (0, 0, 1, 0))
	w=nn.functional.pad(x[:, :, :, :-1], (1, 0, 0, 0))
	nw=nn.functional.pad(x[:, :, :-1, :-1], (1, 0, 1, 0))
	pred=n+w-nw
	vmin=torch.minimum(n, w)
	vmax=torch.maximum(n, w)
	pred=torch.clamp(pred, vmin, vmax)
	x=x-pred
	return x

def calc_entropy_differentiable(x, nbits):
	nlevels=1<<nbits
	x=(x+1)*((nlevels-1)/2)		#[-1, 1] -> [0, nlevels-1]
	entropy=torch.zeros([x.shape[0], x.shape[1]], device=x.device)
	den=1/(x.shape[2]*x.shape[3])
	for sym in range(nlevels):#https://github.com/hyk1996/pytorch-differentiable-histogram/blob/master/differentiable_histogram.py
		prob=torch.relu(1-(x-sym).abs())#https://stackoverflow.com/questions/59850816/how-to-implement-tent-activation-function-in-pytorch
		prob=torch.sum(prob, dim=[2, 3])
		prob*=den
		term=-prob*torch.log2(prob)
		term=torch.nan_to_num(term, nan=0, posinf=0, neginf=0)
		entropy+=term
	entropy=torch.mean(entropy)
	entropy*=1/nbits
	return entropy

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()
		self.sum=0
		self.count=0

		self.param=nn.ParameterList([	#YCoCb-R
			nn.Parameter(torch.tensor([-1. ])), nn.Parameter(torch.tensor([ 0. ])),
			nn.Parameter(torch.tensor([ 0.5])), nn.Parameter(torch.tensor([ 0. ])),
			nn.Parameter(torch.tensor([ 0. ])), nn.Parameter(torch.tensor([-1. ])),
			nn.Parameter(torch.tensor([ 0. ])), nn.Parameter(torch.tensor([ 0. ])),
			nn.Parameter(torch.tensor([ 0. ])), nn.Parameter(torch.tensor([ 0.5])),
			nn.Parameter(torch.tensor([ 0. ])), nn.Parameter(torch.tensor([ 0. ]))
		])
		#self.param=nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(12)])
		#self.p01=nn.Parameter(torch.rand(1, device=device))
	def forward(self, x):
		r0, g0, b0=torch.split(x, 1, dim=1)
		r1=r0+(self.param[ 0]*g0+self.param[ 1]*b0)
		g1=g0+(self.param[ 2]*r1+self.param[ 3]*b0)
		b1=b0+(self.param[ 4]*r1+self.param[ 5]*g1)
		r2=r1+(self.param[ 6]*g1+self.param[ 7]*b1)
		g2=g1+(self.param[ 8]*r2+self.param[ 9]*b1)
		b2=b1+(self.param[10]*r2+self.param[11]*g2)
		x=torch.cat((r2, g2, b2), dim=1)
		x=torch.fmod(x+1, 2)-1#[-1, 1]

		x=pred_clampedgrad(x)
		#x=torch.fmod(x+1, 2)-1#[-1, 1]

		loss=calc_entropy_differentiable(x, 8)
		#loss=torch.mean(torch.square(x))

		self.sum+=loss.item()
		self.count+=1
		return loss, '%14f'%(1/loss.item())

	def epoch_start(self):
		self.sum=0
		self.count=0
	def epoch_end(self):
		if self.count:
			loss=self.sum/self.count
		else:
			loss=0
		return loss, '%14f'%(1/loss)
	def checkpoint_msg(self):
		for idx in range(len(self.param)):
			if idx&1:
				end='\n'
			else:
				end='    '
			print('%21.17f%s'%(self.param[idx].item(), end), end='')

