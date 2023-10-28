import torch
from torch import nn

#codec03: causal pixel predictor

#Conv2d:		Dout = floor((Din + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
#ConvTranspose2d:	Dout = (Din-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1

def get_nb(x, reach, kx, ky):
	b, c, h, w=x.shape
	return torch.cat((x[:, :, ky-reach:ky, kx-reach:kx+reach+1].reshape(b, c, -1), x[:, :, ky:ky+1, kx-reach:kx].reshape(b, c, -1)), dim=2)

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()

		self.reach=3
		self.nnb=2*(self.reach+1)*self.reach
		self.ci=self.nnb*2
		self.dense01=nn.Linear(self.ci, self.ci*2)
		self.dense02=nn.Linear(self.ci*2, self.nnb)
		self.dense03=nn.Linear(self.nnb, self.nnb//2)
		self.dense04=nn.Linear(self.nnb//2, 1)

		self.sum=0
		self.count=0

	def forward(self, x):
		b, c, h, w=x.shape
		deltas=torch.zeros(b, c, self.reach+1, w, dtype=x.dtype, device=x.device)
		zeros=torch.zeros(b, c, 1, w, dtype=x.dtype, device=x.device)
		for ky in range(h-self.reach*2):
			ky2=ky+self.reach
			row=torch.zeros(b, c, 1, 0, dtype=x.dtype, device=x.device)
			for kx in range(w-self.reach*2):
				kx2=kx+self.reach
				x2=torch.cat((get_nb(x, self.reach, kx2, ky2), get_nb(deltas, self.reach, kx2, ky2)), dim=2)
				x2=nn.functional.tanh(self.dense01(x2))
				x2=nn.functional.tanh(self.dense02(x2))
				x2=nn.functional.tanh(self.dense03(x2))
				x2=nn.functional.tanh(self.dense04(x2))
				delta=x[:, :, ky2:ky2+1, kx2:kx2+1]-x2.view(b, c, 1, 1)
				row=torch.cat((row, delta), dim=3)
				deltas=torch.cat((deltas[:, :, :-1, :], torch.cat((zeros[:, :, :, :self.reach], row, zeros[:, :, :, kx2+1:]), dim=3)), dim=2)
			deltas=torch.cat((deltas, zeros), dim=2)
		loss=torch.sqrt(torch.mean(torch.square(deltas)))

		with torch.no_grad():
			prob=torch.histc(deltas, 256, -1, 1)/deltas.nelement()
			invCR=torch.sum(-prob*torch.nan_to_num(torch.log2(prob), 0, 0, 0)).item()/8
		self.sum+=invCR
		self.count+=1
		return loss, 'RMSE %14f  CR %14f'%(255*loss.item(), 1/invCR)

	def epoch_start(self):
		self.sum=0
		self.count=0
	def epoch_end(self):
		invCR=self.sum/self.count
		return invCR, '%14f'%(1/invCR)
	def checkpoint_msg(self):
		pass
