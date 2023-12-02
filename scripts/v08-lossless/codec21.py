import torch
from torch import nn
import math

#codec19-20231125: multi-buffer multi-pass recurrent self-attention causal predictor

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

def calc_min_RMSE(x):
	b, c, h, w=x.shape
	x=torch.square(x).transpose(0, 1).view(c, b*h*w)
	return 128*torch.sqrt(torch.min(torch.mean(x, dim=1)))

def calc_min_invCR(x):
	b, c, h, w=x.shape
	res=w*h
	x=x.reshape(b, c, res)
	entropy=0
	for kc in range(c):
		e2=0
		for kb in range(b):
			prob=torch.histc(x[kb, kc, :], 256, -1, 1)/res
			e2+=torch.sum(-prob*torch.nan_to_num(torch.log2(prob), 0, 0, 0)).item()
		if kc==0 or entropy>e2:
			entropy=e2
	return entropy/(8*b)

def safe_inv(x):
	if x!=0:
		return 1/x
	return math.inf

def median3(a, b, c):
	return torch.max(torch.min(a, torch.max(b, c)), torch.min(b, c))

def inspect(name, x):
	print(name)
	for k in range(x.shape[1]):
		print(x[0, k, 0:5, 0:5])
	print('')

def val2str(x):
	i=int(x.item()*0x1000000)
	if i<0:
		return '-0x%08X,'%(-i)
	return ' 0x%08X,'%i
	#return ' %20.16f,'%val.item()

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

	def to_string(self, name):
		s=''

		s+='static const int '+name+'_w[]=\n'
		s+='{\n'
		for ko in range(self.conv00T.weight.shape[0]):
			for ki in range(self.conv00T.weight.shape[1]):
				filt=torch.cat((self.conv00T.weight[ko, ki].view(-1), self.conv00L.weight[ko, ki].view(-1)), dim=0)
				s+='\t'
				for val in filt:
					s+=val2str(val)
				s+='\n'
		s+='};\n'

		s+='static const int '+name+'_b[]=\n'
		s+='{\n'
		s+='\t'
		for val in self.conv00T.bias:
			s+=val2str(val)
		s+='\n'
		s+='};\n'

		return s

class Predictor(nn.Module):
	def __init__(self, reach, ci, nch, co):
		super(Predictor, self).__init__()

		self.co=co
		self.conv00=CausalConv(reach, 0,  ci, nch)
		self.conv01=CausalConv(reach, 1, nch, nch)
		self.conv02=CausalConv(reach, 1, nch, nch)
		self.conv03=CausalConv(reach, 1, nch, nch)
		self.conv04=CausalConv(reach, 1, nch, nch)
		self.conv05=CausalConv(reach, 1, nch, nch)
		self.conv06=CausalConv(reach, 1, nch, nch)
		self.conv07=CausalConv(reach, 1, nch, co*3)
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

	#def to_string(self, name):
	#	s=''
	#	s+=self.conv00.to_string(name+'_c0')
	#	s+=self.conv01.to_string(name+'_c1')
	#	s+=self.conv02.to_string(name+'_c2')
	#	s+=self.conv03.to_string(name+'_c3')
	#	return s

class Codec(nn.Module):
	def __init__(self):
		super(Codec, self).__init__()


		#C21_01		1.9659@10
		self.cmid=2
		self.pred01a=Predictor(1, 1, 16, 1)
		self.pred01b=Predictor(1, 1, 16, 1)
		self.pred01c=Predictor(1, 1, 16, 1)


		self.esum0=0#RMSE - before
		self.esum1=0#RMSE - after
		self.csum0=0#invCR - before
		self.csum1=0#invCR - after
		self.count=0

	#def fwd_debug(self, x):#
	#	b, c, h, w=x.shape
	#	b2=b*c
	#	x=x.view(b2, 1, h, w)#a batch of channels, because of conv2d
	#
	#	inspect('x', x)
	#
	#	part1a=self.pred01a(x)
	#	part1b=self.pred01b(x)
	#	part1c=self.pred01c(x)
	#	pred1=median3(part1a, part1b, part1c)
	#	delta1=x-pred1
	#	delta1=torch.fmod(delta1+1, 2)-1
	#
	#	inspect('part1a', part1a)
	#	inspect('part1b', part1b)
	#	inspect('part1c', part1c)
	#	inspect('pred1', pred1)
	#	inspect('x', x)
	#	inspect('delta1', delta1)
	#
	#	part2a=self.pred02a(delta1)
	#	part2b=self.pred02b(delta1)
	#	part2c=self.pred02c(delta1)
	#	pred2=median3(part2a, part2b, part2c)
	#	delta2=delta1-pred2
	#	delta2=torch.fmod(delta2+1, 2)-1
	#
	#	inspect('part2a', part2a)
	#	inspect('part2b', part2b)
	#	inspect('part2c', part2c)
	#	inspect('pred2', pred2)
	#	inspect('delta1', delta1)
	#	inspect('delta2', delta2)
	#
	#	loss1=calc_RMSE(delta2)
	#	cr=1/calc_min_invCR(delta2)
	#	return loss1, 'MSG %f'%cr


	def forward(self, x):
		b, c, h, w=x.shape
		b2=b*c
		x=x.view(b2, 1, h, w)#a batch of channels, because of conv2d


		deltas=torch.fmod(x.repeat(1, self.cmid, 1, 1)-median3(self.pred01a(x), self.pred01b(x), self.pred01c(x))+1, 2)-1		#[-1, 1]


		loss1=calc_RMSE(deltas)

		with torch.no_grad():
			loss0=calc_min_RMSE(x).item()
			invCR0=calc_min_invCR(x)
			invCR1=calc_min_invCR(deltas)
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

	#def to_string(self):
	#	s=''
	#	s+=self.pred01a.to_string('pred1a')
	#	s+=self.pred01b.to_string('pred1b')
	#	s+=self.pred01c.to_string('pred1c')
	#	s+=self.pred02a.to_string('pred2a')
	#	s+=self.pred02b.to_string('pred2b')
	#	s+=self.pred02c.to_string('pred2c')
	#	return s
