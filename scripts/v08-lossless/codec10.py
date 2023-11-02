import torch
from torch import nn
import math

#codec10: causal pixel predictor with aux buffers (C06 with different hyperparams)

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def get_nb(x, reach, kx, ky):
	b, c, h, w=x.shape
	nb=torch.cat((x[:, :, ky-reach:ky, kx-reach:kx+reach+1].reshape(b, c, -1), x[:, :, ky, kx-reach:kx].reshape(b, c, -1)), dim=2)
	c0, c1, c2=torch.split(nb, c//3, dim=1)
	return c0.view(b, -1), c1.view(b, -1), c2.view(b, -1)

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

class Predictor(nn.Module):
	def __init__(self, ci, nch, co):
		super(Predictor, self).__init__()
		self.dense01=nn.Linear(ci, nch)
		self.dense02=nn.Linear(nch, nch)
		self.dense03=nn.Linear(nch, nch)
		self.dense04=nn.Linear(nch, nch)
		self.dense05=nn.Linear(nch, nch)
		self.dense06=nn.Linear(nch, nch)
		self.dense07=nn.Linear(nch, nch)
		self.dense08=nn.Linear(nch, co)
	def forward(self, x):
		x=nn.functional.leaky_relu(self.dense01(x))
		x=nn.functional.leaky_relu(self.dense02(x))
		x=nn.functional.leaky_relu(self.dense03(x))
		x=nn.functional.leaky_relu(self.dense04(x))
		x=nn.functional.leaky_relu(self.dense05(x))
		x=nn.functional.leaky_relu(self.dense06(x))
		x=nn.functional.leaky_relu(self.dense07(x))
		x=torch.clamp(self.dense08(x), -1, 1)
		return x

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()

		self.reach_p=2		#pixel reach > error/aux reach
		self.reach_e=1
		self.naux=3
		self.blocksize=256	#just to improve train time

		self.ci=2*(self.reach_p+1)*self.reach_p + 2*(self.reach_e+1)*self.reach_e*(self.naux+1)
		self.co=self.naux+1

		self.pred01=Predictor(self.ci, 24, self.co)
		self.pred02=Predictor(self.ci, 48, self.co)#luma predictor is larger
		self.pred03=Predictor(self.ci, 24, self.co)

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

		x=x.unfold(2, self.blocksize, self.blocksize).unfold(3, self.blocksize, self.blocksize).reshape(b, c, nblocks, blockarea)
		x=x.transpose(1, 2).reshape(b*nblocks, c, self.blocksize, self.blocksize)	#a batch of blocks
		b, c, h, w=x.shape

		deltas=torch.zeros(b, c*(self.naux+1), self.reach_p+1, w, dtype=x.dtype, device=x.device)
		zeros=torch.zeros(b, c*(self.naux+1), 1, w, dtype=x.dtype, device=x.device)
		for ky in range(self.reach_p, h-self.reach_p):
			row=torch.zeros(b, c*(self.naux+1), 1, 0, dtype=x.dtype, device=x.device)
			for kx in range(self.reach_p, w-self.reach_p):
				nb_p0, nb_p1, nb_p2=get_nb(x, self.reach_p, kx, ky)#pixels
				nb_e0, nb_e1, nb_e2=get_nb(deltas, self.reach_e, kx, ky)#errors with aux data {aux[0], aux[1], ...aux[n-1], pred}

				a0, p0=torch.split(self.pred01(torch.cat((nb_p0, nb_e0), dim=1)), self.naux, dim=1)#{aux data, ...pred}
				a1, p1=torch.split(self.pred02(torch.cat((nb_p1, nb_e1), dim=1)), self.naux, dim=1)
				a2, p2=torch.split(self.pred03(torch.cat((nb_p2, nb_e2), dim=1)), self.naux, dim=1)

				p0=(torch.fmod( x[:, 0:1, ky, kx]-p0 +1, 2)-1)		#[-1, 1]
				p1=(torch.fmod( x[:, 1:2, ky, kx]-p1 +1, 2)-1)
				p2=(torch.fmod( x[:, 2:3, ky, kx]-p2 +1, 2)-1)

				row=torch.cat((row, torch.cat((a0, p0, a1, p1, a2, p2), dim=1).view(b, c*(self.naux+1), 1, 1)), dim=3)
				deltas=torch.cat((deltas[:, :, :-1, :], torch.cat((zeros[:, :, :, :self.reach_p], row, zeros[:, :, :, kx+1:]), dim=3)), dim=2)
			deltas=torch.cat((deltas, zeros), dim=2)
		channels=torch.split(deltas, 1, dim=1)#remove aux data
		deltas=torch.cat((channels[(self.naux+1)*0-1], channels[(self.naux+1)*1-1], channels[(self.naux+1)*2-1]), dim=1)#leave only the deltas

		loss1=calc_RMSE(deltas)

		with torch.no_grad():
			loss0=calc_RMSE(x).item()
			invCR0=calc_invCR(x)
			invCR1=calc_invCR(deltas[:, :, self.reach_p:, self.reach_p:-self.reach_p])
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
