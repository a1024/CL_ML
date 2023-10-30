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
	def __init__(self, r, nch):
		super(Predictor, self).__init__()
		self.reach=r
		self.nnb=2*(self.reach+1)*self.reach
		self.ci=self.nnb*2
		self.dense01=nn.Linear(self.ci, nch)
		self.dense02=nn.Linear(nch, nch)
		self.dense03=nn.Linear(nch, nch)
		self.dense04=nn.Linear(nch, nch)
		self.dense05=nn.Linear(nch, nch)
		self.dense06=nn.Linear(nch, nch)
		self.dense07=nn.Linear(nch, nch)
		self.dense08=nn.Linear(nch, 1)
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

		self.reach=3
		self.nnb=2*(self.reach+1)*self.reach
		self.ci=self.nnb*2

		self.pred01=Predictor(3, 16)#reach must be same everywhere TODO remove limitation by early splitting
		self.pred02=Predictor(3, 32)#a larger model for luma
		self.pred03=Predictor(3, 16)

		#self.dense01=nn.Linear(self.ci, self.ci)
		#self.dense02=nn.Linear(self.ci, self.ci)
		#self.dense03=nn.Linear(self.ci, self.ci)
		#self.dense04=nn.Linear(self.ci, self.ci)
		#self.dense05=nn.Linear(self.ci, self.nnb)
		#self.dense06=nn.Linear(self.nnb, self.nnb//2)
		#self.dense07=nn.Linear(self.nnb//2, 1)

		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0

	def forward(self, x):
		b, c, h, w=x.shape
		deltas=torch.zeros(b, c, self.reach+1, w, dtype=x.dtype, device=x.device)
		zeros=torch.zeros(b, c, 1, w, dtype=x.dtype, device=x.device)
		for ky in range(self.reach, h-self.reach):
			row=torch.zeros(b, c, 1, 0, dtype=x.dtype, device=x.device)
			for kx in range(self.reach, w-self.reach):
				x2=torch.cat((get_nb(x, self.reach, kx, ky), get_nb(deltas, self.reach, kx, ky)), dim=2)

				cm, y, cb=torch.split(x2, 1, dim=1)
				cm=self.pred01(cm)
				y =self.pred02(y )
				cb=self.pred03(cb)
				x2=torch.cat((cm, y, cb), dim=1)

				#x2=nn.functional.leaky_relu(self.dense01(x2))
				#x2=nn.functional.leaky_relu(self.dense02(x2))
				#x2=nn.functional.leaky_relu(self.dense03(x2))
				#x2=nn.functional.leaky_relu(self.dense04(x2))
				#x2=nn.functional.leaky_relu(self.dense05(x2))
				#x2=nn.functional.leaky_relu(self.dense06(x2))
				#x2=torch.clamp(self.dense07(x2), -1, 1)

				delta=x[:, :, ky:ky+1, kx:kx+1]-x2.view(b, c, 1, 1)
				delta=torch.fmod(delta+1, 2)-1		#[-1, 1]

				row=torch.cat((row, delta), dim=3)
				deltas=torch.cat((deltas[:, :, :-1, :], torch.cat((zeros[:, :, :, :self.reach], row, zeros[:, :, :, kx+1:]), dim=3)), dim=2)
			deltas=torch.cat((deltas, zeros), dim=2)

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
