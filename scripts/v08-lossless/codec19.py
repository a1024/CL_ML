import torch
from torch import nn
import math

#codec19-20231124: multi-pass recurrent self-attention causal predictor

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def calc_RMSE(x):
	return 128*torch.sqrt(torch.mean(torch.square(x)))

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

def median3(a, b, c):
	return torch.max(torch.min(a, torch.max(b, c)), torch.min(b, c))

class CausalConv(nn.Module):
	def __init__(self, reach, curr, ci, co):#nch must end with 1
		super(CausalConv, self).__init__()

		self.reach=reach
		self.curr=curr
		self.conv00T=nn.Conv2d(ci, co, (reach, reach<<1|1))#(Kh, Kw)
		self.conv00L=nn.Conv2d(ci, co, (1, reach+curr), bias=False)
	def forward(self, x):
		xt=self.conv00T(nn.functional.pad(x[:, :, :-1, :], (self.reach, self.reach, self.reach, 0)))#(L, R, T, B)
		xl=self.conv00L(nn.functional.pad(x[:, :, :, :(x.size(3) if self.curr else -1)], (self.reach, 0, 0, 0)))
		return xt+xl

class Predictor(nn.Module):
	def __init__(self, nch):
		super(Predictor, self).__init__()

		self.conv00=CausalConv(1, 0, 1, nch)
		self.conv01=CausalConv(1, 1, nch, nch)
		self.conv02=CausalConv(1, 1, nch, nch)
		self.conv03=CausalConv(1, 1, nch, 3)
	def forward(self, x):
		x=nn.functional.leaky_relu(self.conv00(x))
		x=nn.functional.leaky_relu(self.conv01(x))
		x=nn.functional.leaky_relu(self.conv02(x))
		a, b, c=torch.split(torch.clamp(self.conv03(x), -1, 1), 1, dim=1)
		return median3(a, b, c)

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()
		

		#C19_01		1.9650@10  1.9733@20  1.9682@30
		self.pred01=Predictor(16)
		self.pred02=Predictor(16)


		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0

	def forward(self, x):
		b, c, h, w=x.shape
		b2=b*c
		x=x.view(b2, 1, h, w)#a batch of channels, because of conv2d


		deltas=torch.fmod(x-self.pred01(x)+1, 2)-1		#[-1, 1]
		deltas=torch.fmod(deltas-self.pred02(deltas)+1, 2)-1


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
