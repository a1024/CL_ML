import torch
from torch import nn
import math

#codec22-20231224: adaptive RCT

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def safe_inv(x):
	if x!=0:
		return 1/x
	return math.inf

def calc_RMSE(x):
	return 128*torch.sqrt(torch.mean(torch.square(x)))

def calc_invCR(x):
	entropy=0
	bsize, c, h, w=x.shape
	nch=bsize*c; res=w*h
	x=x.reshape(nch, res)
	for k in range(nch):
		prob=torch.histc(x[k, :], 256, -1, 1)/res
		entropy+=torch.sum(-prob*torch.nan_to_num(torch.log2(prob), 0, 0, 0)).item()
	return entropy/(8*nch)

def pred_clampgrad(x):
	n=nn.functional.pad(x[:, :, :-1, :], (0, 0, 1, 0))
	w=nn.functional.pad(x[:, :, :, :-1], (1, 0, 0, 0))
	nw=nn.functional.pad(x[:, :, :-1, :-1], (1, 0, 1, 0))
	return torch.clamp(n+w-nw, torch.min(n, w), torch.max(n, w))

def median3(a, b, c):
	return torch.max(torch.min(a, torch.max(b, c)), torch.min(b, c))

class Predictor(nn.Module):
	def __init__(self, ci, nc, co):
		super(Predictor, self).__init__()

		self.co=co
		self.conv00=nn.Conv2d(ci, nc, 3, 1, 1)
		self.conv01=nn.Conv2d(nc, nc, 3, 1, 1)
		self.conv02=nn.Conv2d(nc, nc, 3, 1, 1)
		self.conv03=nn.Conv2d(nc, nc, 3, 1, 1)
		self.conv04=nn.Conv2d(nc, nc, 3, 1, 1)
		self.conv05=nn.Conv2d(nc, nc, 3, 1, 1)
		self.conv06=nn.Conv2d(nc, nc, 3, 1, 1)
		self.conv07=nn.Conv2d(nc, co*3, 3, 1, 1)
	def forward(self, x):
		x=nn.functional.leaky_relu(self.conv00(x))
		x=nn.functional.leaky_relu(self.conv01(x))
		x=nn.functional.leaky_relu(self.conv02(x))
		x=nn.functional.leaky_relu(self.conv03(x))
		x=nn.functional.leaky_relu(self.conv04(x))
		x=nn.functional.leaky_relu(self.conv05(x))
		x=nn.functional.leaky_relu(self.conv06(x))
		a, b, c=torch.split(torch.clamp(self.conv07(x), -1, 1), self.co, dim=1)
		return median3(a, b, c)
	
class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()

		self.pred_r=Predictor(2, 16, 1)
		self.pred_g=Predictor(2, 16, 1)
		self.pred_b=Predictor(2, 16, 1)

		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0
	def forward(self, x):
		bsize, c, h, w=x.shape
		r, g, b=torch.split(x, 1, dim=1)
		r=r-self.pred_r(torch.cat((g, b), dim=1))
		g=g-self.pred_r(torch.cat((r, b), dim=1))
		b=b-self.pred_r(torch.cat((r, g), dim=1))

		x2=torch.cat((r, g, b), dim=1)
		deltas=x2-pred_clampgrad(x2)

		loss1=calc_RMSE(deltas)

		with torch.no_grad():
			loss0=calc_RMSE(x).item()
			invCR0=calc_invCR(x)
			invCR1=calc_invCR(deltas)
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
