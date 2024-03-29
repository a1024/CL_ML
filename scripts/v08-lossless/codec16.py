import torch
from torch import nn
import math

#codec16-20231120: DWT-based parallel predictor

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

class FilterIden(nn.Module):
	def __init__(self, kernel, nch):#nch must end with 1
		super(FilterIden, self).__init__()

		self.layers=nn.ModuleList()
		for kl in range(len(nch)-1):
			self.layers.add_module('conv%02d'%kl, nn.Conv2d(nch[kl], nch[kl+1], kernel, 1, kernel//2))
	def forward(self, x):
		nlayers=len(self.layers)
		x=nn.functional.leaky_relu(self.layers.conv00(x))
		for kl in range(1, nlayers-1):
			x=nn.functional.leaky_relu(self.layers.get_submodule('conv%02d'%kl)(x)+x)
		return torch.clamp(self.layers.get_submodule('conv%02d'%(nlayers-1))(x), -1, 1)

class Filter(nn.Module):
	def __init__(self, kernel, nch):#nch must end with 1
		super(Filter, self).__init__()

		self.layers=nn.ModuleList()
		for kl in range(len(nch)-1):
			self.layers.add_module('conv%02d'%kl, nn.Conv2d(nch[kl], nch[kl+1], kernel, 1, kernel//2))
	def forward(self, x):
		nlayers=len(self.layers)
		x=nn.functional.leaky_relu(self.layers.conv00(x))
		for kl in range(1, nlayers-1):
			x=nn.functional.leaky_relu(self.layers.get_submodule('conv%02d'%kl)(x))
		return torch.clamp(self.layers.get_submodule('conv%02d'%(nlayers-1))(x), -1, 1)

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()


		#C16_01_1x1		1.6963@10

		#C16_02_3x3	4L16C		1.8861@10  1.9371@20  1.9249@30  1.9379@60
		#self.pred_x=Predictor([1, 16, 16, 16, 1])
		#self.pred_y=Predictor([1, 16, 16, 16, 1])

		#C16_03		8L32C		1.9303@50
		#nch=32
		#self.pred_x=Predictor([1, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.pred_y=Predictor([1, nch, nch, nch, nch, nch, nch, nch, 1])

		#C16_04		8L16C-res-5x5		1.8731@10  1.8356@20
		#nch=16
		#self.pred_x=Predictor([1, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.pred_y=Predictor([1, nch, nch, nch, nch, nch, nch, nch, 1])

		#C16_05		8L16C-res-7x7		1.8065@10  1.8631@20
		#nch=16
		#self.pred_x=Predictor([1, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.pred_y=Predictor([1, nch, nch, nch, nch, nch, nch, nch, 1])

		#C16_06		8L16C-res-7x7		1.6762@10
		#nch=16
		#self.predict_x=Filter([1, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.update_x =Filter([1, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.predict_y=Filter([1, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.update_y =Filter([1, nch, nch, nch, nch, nch, nch, nch, 1])

		#C16_07		10L-16C-res-5x5			 14500		1.7997@10  1.9216@20  1.9234@30
		#nch=8
		#self.predict_x=Filter(5, [1, nch, nch, nch, 1])
		#self.update_x =Filter(5, [1, nch, nch, nch, 1])
		#self.predict_y=Filter(5, [1, nch, nch, nch, 1])
		#self.update_y =Filter(5, [1, nch, nch, nch, 1])

		#C16_08		10L-16C-iden-5x5-PUCR		417160		1.9034@10  1.8497@20
		#nch=16
		#self.predict_x=FilterIden(5, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.update_x =Filter    (5, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.correct_x=Filter    (5, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.refine_x =Filter    (5, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.predict_y=FilterIden(5, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.update_y =Filter    (5, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.correct_y=Filter    (5, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.refine_y =Filter    (5, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])

		#C16_09		24L-16C-iden-3x3-PUCR		410760		1.8295@10  1.9173@20  1.9073@30
		#nch=16
		#self.predict_x=FilterIden(3, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.update_x =Filter    (3, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.correct_x=Filter    (3, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.refine_x =Filter    (3, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.predict_y=FilterIden(3, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.update_y =Filter    (3, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.correct_y=Filter    (3, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])
		#self.refine_y =Filter    (3, [1, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, nch, 1])


		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0

	def forward(self, x):
		b, c, h, w=x.shape
		x=x.view(b*c, 1, h, w)#a batch of channels, because of conv2d

		x1e, x1o=deinterleave_x(x)
		x1o=x1o-self.predict_x(x1e)
		x1e=x1e+self.update_x(x1o)
		x1o=x1o-self.correct_x(x1e)
		x1e=x1e+self.refine_x(x1o)
		y1e, y1o=deinterleave_y(x1e)
		y1o=y1o-self.predict_y(y1e)
		y1e=y1e+self.update_y(y1o)
		y1o=y1o-self.correct_y(y1e)
		y1e=y1e+self.refine_y(y1o)

		x2e, x2o=deinterleave_x(y1e)
		x2o=x2o-self.predict_x(x2e)
		x2e=x2e+self.update_x(x2o)
		x2o=x2o-self.correct_x(x2e)
		x2e=x2e+self.refine_x(x2o)
		y2e, y2o=deinterleave_y(x2e)
		y2o=y2o-self.predict_y(y2e)
		y2e=y2e+self.update_y(y2o)
		y2o=y2o-self.correct_y(y2e)
		y2e=y2e+self.refine_y(y2o)

		x3e, x3o=deinterleave_x(y2e)
		x3o=x3o-self.predict_x(x3e)
		x3e=x3e+self.update_x(x3o)
		x3o=x3o-self.correct_x(x3e)
		x3e=x3e+self.refine_x(x3o)
		y3e, y3o=deinterleave_y(x3e)
		y3o=y3o-self.predict_y(y3e)
		x3e=x3e+self.update_y(x3o)
		y3o=y3o-self.correct_y(y3e)
		y3e=y3e+self.refine_y(y3o)
		
		x4e, x4o=deinterleave_x(y3e)
		x4o=x4o-self.predict_x(x4e)
		x4e=x4e+self.update_x(x4o)
		x4o=x4o-self.correct_x(x4e)
		x4e=x4e+self.refine_x(x4o)
		y4e, y4o=deinterleave_y(x4e)
		y4o=y4o-self.predict_y(y4e)
		x4e=x4e+self.update_y(x4o)
		y4o=y4o-self.correct_y(y4e)
		y4e=y4e+self.refine_y(y4o)
		
		x5e, x5o=deinterleave_x(y4e)
		x5o=x5o-self.predict_x(x5e)
		x5e=x5e+self.update_x(x5o)
		x5o=x5o-self.correct_x(x5e)
		x5e=x5e+self.refine_x(x5o)
		y5e, y5o=deinterleave_y(x5e)
		y5o=y5o-self.predict_y(y5e)
		x5e=x5e+self.update_y(x5o)
		y5o=y5o-self.correct_y(y5e)
		y5e=y5e+self.refine_y(y5o)

		y4e   =join(y5e, y5o, x5o)
		y3e   =join(y4e, y4o, x4o)
		y2e   =join(y3e, y3o, x3o)
		y1e   =join(y2e, y2o, x2o)
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
