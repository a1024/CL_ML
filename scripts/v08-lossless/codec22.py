#2023-12-19Tu
import torch
from torch import nn
import math

#codec22: RCT test

def calc_invCR(x, start, end, nbins):
	entropy=0
	b, c, h, w=x.shape
	nch=b*c; res=w*h
	x=x.reshape(nch, res)
	for k in range(nch):
		prob=torch.histc(x[k, :], nbins, start, end)/res
		entropy+=torch.sum(-prob*torch.nan_to_num(torch.log2(prob), 0, 0, 0)).item()
	return entropy/(8*nch)

def RCT_subg(x):
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	b-=g

	x=torch.cat((g, b, r), dim=1)#{Y, Cb, Cr}
	return x

def RCT_JPEG2000(x):
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	b-=g
	g+=(r+b)*0.25

	x=torch.cat((g, b, r), dim=1)#{Y, Cb, Cr}
	return x

def RCT_YCoCg_R(x):
	r, g, b=torch.split(x, 1, dim=1)

	r-=b
	b+=r*0.5
	g-=b
	b+=g*0.5

	x=torch.cat((b, g, r), dim=1)#{Y, Cg, Co}
	return x

def RCT_YCbCr_R_v1(x):
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	g+=r*0.5
	b-=g
	g+=b*0.5

	x=torch.cat((g, b, r), dim=1)#{Y, Cb, Cr}
	return x

def RCT_YCbCr_R_v2(x):
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	g+=r*0.5
	b-=g
	g+=(2*b-r)*0.125

	x=torch.cat((g, b, r), dim=1)#{Y, Cb, Cr}
	return x

def RCT_YCbCr_R_v3(x):
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	g+=r*0.5
	b-=g
	g+=(2*b+r)*0.125

	x=torch.cat((g, b, r), dim=1)#{Y, Cb, Cr}
	return x

def RCT_YCbCr_R_v4(x):
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	g+=r*0.5
	b-=g
	g+=b/3

	x=torch.cat((g, b, r), dim=1)#{Y, Cb, Cr}
	return x

def RCT_YCbCr_R_v5(x):
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	g+=r*0.5
	b-=g
	g+=b*0.375

	x=torch.cat((g, b, r), dim=1)#{Y, Cb, Cr}
	return x

def RCT_YCbCr_R_v6(x):
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	g+=r*0.5
	b-=g
	g+=b*0.4375

	x=torch.cat((g, b, r), dim=1)#{Y, Cb, Cr}
	return x

def pred_cg(x):
	N=nn.functional.pad(x[:, :, :-1, :], (0, 0, 1, 0))
	W=nn.functional.pad(x[:, :, :, :-1], (1, 0, 0, 0))
	NW=nn.functional.pad(x[:, :, :-1, :-1], (1, 0, 1, 0))
	vmin=torch.min(N, W)
	vmax=torch.max(N, W)
	pred=torch.clamp(N+W-NW, torch.min(N, W), torch.max(N, W))
	x=torch.fmod(x-pred+1, 2)-1
	return x

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()
		self.csizes=[0]*19
		self.usize=0

		self.dummy=nn.Parameter(torch.tensor([1.]))

	def forward(self, x):
		funcs=[
			RCT_subg,
			RCT_JPEG2000,
			RCT_YCoCg_R,
			RCT_YCbCr_R_v1,
			RCT_YCbCr_R_v2,
			RCT_YCbCr_R_v3,
			RCT_YCbCr_R_v4,
			RCT_YCbCr_R_v5,
			RCT_YCbCr_R_v6,
		]
		self.usize+=x.nelement()
		for kt in range(len(funcs)):
			for ma in range(2):
				if ma:
					t=funcs[kt](x)
					t=torch.fmod(t+1, 2)-1
					t=pred_cg(t)
					invCR=calc_invCR(t, -1, 1, 256)
				else:
					t=funcs[kt](x)
					t=pred_cg(t)
					invCR=calc_invCR(t, -2, 2, 512)
				self.csizes[kt<<1|ma]+=invCR*x.nelement()
		t=pred_cg(x)
		invCR=calc_invCR(t, -1, 1, 256)
		self.csizes[18]+=invCR*x.nelement()
		return 0, ''

	def epoch_start(self):
		pass
	def epoch_end(self):
		return 0, ''
	def checkpoint_msg(self):
		print('usize %d'%self.usize)
		for ks in range(len(self.csizes)):
			print('%2d %20f'%(ks, self.csizes[ks]))
