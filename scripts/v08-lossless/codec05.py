import torch
from torch import nn
import math

#codec03: causal pixel predictor

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def get_nb(x, reach, kx, ky):
	b, c, h, w=x.shape
	return torch.cat((x[:, :, ky-reach:ky, kx-reach:kx+reach+1].reshape(b, c, -1), x[:, :, ky:ky+1, kx-reach:kx].reshape(b, c, -1)), dim=2)

def calc_RMSE(x):
	return 255*torch.sqrt(torch.mean(torch.square(x)))

def calc_invCR(x):
	entropy=0
	b, c, h, w=x.shape
	nch=b*c
	res=h*w
	for kb in range(b):
		for kc in range(c):
			prob=torch.histc(x[kb, kc, :, :], 256, -1, 1)/res
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

		self.reach=1
		self.naux=8
		self.ci=2*(self.reach+1)*self.reach*(2+self.naux)
		self.co=self.naux+1

		self.pred01=Predictor(self.ci, 32, self.co)
		self.pred02=Predictor(self.ci, 64, self.co)#a larger model for luma
		self.pred03=Predictor(self.ci, 32, self.co)

		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0

	def forward(self, x):
		'''
		b, c, h, w=x.shape
		deltas=torch.zeros(b, c, self.reach+1, w, dtype=x.dtype, device=x.device)
		auxbuf=torch.zeros(b, c*self.naux, self.reach+1, w, dtype=x.dtype, device=x.device)
		zeros=torch.zeros(b, c, 1, w, dtype=x.dtype, device=x.device)
		for ky in range(self.reach, h-self.reach):
			row=torch.zeros(b, c, 1, 0, dtype=x.dtype, device=x.device)
			auxrow=torch.zeros(b, c*self.naux, 1, 0, dtype=x.dtype, device=x.device)
			for kx in range(self.reach, w-self.reach):
				nb_c0, nb_c1, nb_c2=torch.split(get_nb(x, self.reach, kx, ky), 1, dim=1)
				nb_e0, nb_e1, nb_e2=torch.split(get_nb(deltas, self.reach, kx, ky), 1, dim=1)
				nb_a0, nb_a1, nb_a2=torch.split(get_nb(auxbuf, self.reach, kx, ky), 1, dim=1)

				a0, c0=torch.split(self.pred01(torch.cat((nb_c0, nb_e0, nb_a0), dim=1).view(b, -1)), self.naux, dim=1)
				a1, c1=torch.split(self.pred02(torch.cat((nb_c1, nb_e1, nb_a1), dim=1).view(b, -1)), self.naux, dim=1)
				a2, c2=torch.split(self.pred03(torch.cat((nb_c2, nb_e2, nb_a2), dim=1).view(b, -1)), self.naux, dim=1)

				x2=torch.cat((c0, c1, c2), dim=1).view(b, c, 1, 1)
				aux=torch.cat((a0, a1, a2), dim=1).view(b, c*self.naux, 1, 1)

				delta=x[:, :, ky:ky+1, kx:kx+1]-x2
				delta=torch.fmod(delta+1, 2)-1		#[-1, 1]

				row=torch.cat((row, delta), dim=3)
				deltas=torch.cat((deltas[:, :, :-1, :], torch.cat((zeros[:, :, :, :self.reach], row, zeros[:, :, :, kx+1:]), dim=3)), dim=2)

				auxrow==torch.cat((auxrow, aux), dim=3)
				auxbuf=torch.cat((auxbuf[:, :, :-1, :], torch.cat((zeros[:, :, :, :self.reach], auxrow, zeros[:, :, :, kx+1:]), dim=3)), dim=2)
			deltas=torch.cat((deltas, zeros), dim=2)
			auxbuf=torch.cat((auxbuf, zeros), dim=2)
		'''

		b, c, h, w=x.shape
		deltas=torch.zeros(b, c*(self.naux+1), self.reach+1, w, dtype=x.dtype, device=x.device)
		zeros=torch.zeros(b, c*(self.naux+1), 1, w, dtype=x.dtype, device=x.device)
		for ky in range(self.reach, h-self.reach):
			row=torch.zeros(b, c*(self.naux+1), 1, 0, dtype=x.dtype, device=x.device)
			for kx in range(self.reach, w-self.reach):
				nb_p0, nb_p1, nb_p2=torch.split(get_nb(x, self.reach, kx, ky), 1, dim=1)#pixels
				nb_e0, nb_e1, nb_e2=torch.split(get_nb(deltas, self.reach, kx, ky), self.naux+1, dim=1)#errors with aux data

				a0, p0=torch.split(self.pred01(torch.cat((nb_p0, nb_e0), dim=1).view(b, -1)), self.naux, dim=1)#{aux data, ...pred}
				a1, p1=torch.split(self.pred02(torch.cat((nb_p1, nb_e1), dim=1).view(b, -1)), self.naux, dim=1)
				a2, p2=torch.split(self.pred03(torch.cat((nb_p2, nb_e2), dim=1).view(b, -1)), self.naux, dim=1)

				p0=(torch.fmod( x[:, 0:1, ky, kx]-p0 +1, 2)-1)		#[-1, 1]
				p1=(torch.fmod( x[:, 1:2, ky, kx]-p1 +1, 2)-1)
				p2=(torch.fmod( x[:, 2:3, ky, kx]-p2 +1, 2)-1)

				row=torch.cat((row, torch.cat((a0, p0, a1, p1, a2, p2), dim=1).view(b, c*(self.naux+1), 1, 1)), dim=3)
				deltas=torch.cat((deltas[:, :, :-1, :], torch.cat((zeros[:, :, :, :self.reach], row, zeros[:, :, :, kx+1:]), dim=3)), dim=2)
			deltas=torch.cat((deltas, zeros), dim=2)
		channels=torch.split(deltas, 1, dim=1)#remove aux data
		deltas=torch.cat((channels[(self.naux+1)*0-1], channels[(self.naux+1)*1-1], channels[(self.naux+1)*2-1]), dim=1)#leave only the deltas

		loss1=calc_RMSE(deltas)

		with torch.no_grad():
			loss0=calc_RMSE(x).item()
			invCR0=calc_invCR(x)
			invCR1=calc_invCR(deltas[:, :, self.reach:, self.reach:-self.reach])
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
