#2023-02-14Tu
import os
from PIL import Image
import math
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

import time
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
from torchsummary import summary

#name		params		sec/epoch	RMSE@100
#M01-K4		 338648		6.8260512	114.859905
#M02-K6		 806232		4.6558596	126.223620463515
#M03-K2		1006104		5.2097142	131.379894787635
#M04-K6		2101344		6.2042316	112.08402610353
#M05-K7		2190416		6.7232346	118.18160328135

#M06-K6		 517472		6.5869116	116.9923015929

#M07-conf	 515258		5.6286504	 12.81913691682

#name			params		sec/epoch	train@100		best test
#M08-pred		 514889		 5.0347398	 12.08207591073		 8.57789626338@1700
#M09-large		 639689		 6.6802074	 71.61097034238		44.09953683621
#M10-M08-bn-drop	 518933		 6.567411	101.645483988915	53.290856033505
#M11-M08-blockbn	 516409		 5.8487718	 26.242025343285	18.64337470374@100
#M12-M08-drop		 514889		 5.6818374	 79.25493040116		 9.04642632228@100
#M13-shallow		 772489		 5.0270928	 11.489903441595	 7.07981335011@100	ratio=1.3720283857037

#M14-row		 262019		 5.1864354	 17.011302003495	ratio=1.001061384226, pred=62.63% @400

#name			params		sec/epoch	train@100		best test
#M15-att-ewm		 374595		26.0312046	 20.20738890486		r=1.003878769661 p=61.62% @100 x 0.0001
#M15-att-mm		 374595		 6.79136@32	 20.550402138795	r=1.107674644810 p=56.53% @100	//1:50.863130@128 //31.67 mins@256?
#M16-att-mm-neck	 291555		26.67 mins@224?
#M17-block		 583059		14.5354662	 36.62173994985		r 1.109025129981 p 55.11% @100 blocky
#M18-sconv		 559235		16.3057266	 26.888148483589	r 1.118695859912 p 57.76% @100 blocky
#M19-row		 581827		56.9553		 60@10 stuck		vanishing gradient
#M20-resrow		 594179		60.22224	 22.282792276512@30	vanishing gradient
#M21-resrow-bn		 594819		33.004496	 19.701889990533	1.175142930510		//r 1.139497545854 p 59.72% @200	//27.912438621811@30	gradient vanishes after a while


#name			params		sec/epoch	train@100		best test
#C01-multires		5890433		16.131165	stuck
#C02-simple		1087302				stuck			//2239046 stuck
#C03-trivial		 305382		 7.3799532	 9.158906909376		game on			//669926 stuck
#C04-W64D15		 189420		 8.4128634	12.934512204576		checkboard pattern
#C05-W256D12		1523635		14.0388084	 9.447993428465		blurry
#C06-W128D15		1282953		 5.9219958	 6.521447350921		8bit,lambda=0: crisp but bad color, 2bit,lambda=0.001: noisy @1000 QUANTIZE INSTEAD OF ADDING NOISE

#C07-pred		 695183		17.725816	24.395361444897		PSNR 18.85  BPP 1.66 (0.444 bypass)  @500 lambda 0.0001  no color

#C08-3bit		 428891		 5.7715668	15.347845008896		checkboard @200
#C09-heavy		 851216		 7.444292	21.186716620891		PSNR 16.66  BPP 0.1172  @1000 lambda 0.0001  low freq noise, checkboard	//checkboard @100
#C10-upsmp		 701456		 7.871121	20.871285020741		PSNR 16.37  BPP 0.1176  @1000	pixelated
#C11-3c			 902976		 9.3268104	15.434808229835		PSNR 17.23  BPP 0.3260  @200	PSNR 16.93 BPP 0.2145 @1400 grainy banding
#C12-8c			 999820		10.4657478	12.186279430929		PSNR 17.01  BPP 0.4786  @200 grainy banding




## config ##
import codec12 as currentcodec
modelname='C12'
pretrained=1	# when changing model design, rename old saved model, then assign pretrained=0 for first train
save_records=0

epochs=200
lr=0.001	#always start with high rate
batch_size=32	# <=24, increase batch size instead of decreasing learning rate
train_block=64

clip_grad=1	# enable if got nan
use_adam=1	# disable if got nan
use_flickr_dataset=0

model_summary=0
debug_model=0

g_rate=1
g_rate_factor=0.0001




use_cuda=0
device_name='cpu'
if torch.cuda.is_available() and torch.cuda.device_count()>0:
	use_cuda=1
	device_name='cuda:0'
else:
	use_cuda=0
print('%s, LR=%f, Batch=%d, Records=%d'%(device_name, lr, batch_size, save_records))
device=torch.device(device_name)

if debug_model:#https://stackoverflow.com/questions/61397176/how-to-keep-matplotlib-from-stealing-focus
	matplotlib.use("Qt5agg")

def cropTL(image):
	return torchvision.transforms.functional.crop(image, 0, 0, 64, 64)

def ensureChannels(x):
	global device
	if x.shape[0]==1:
		x=x.repeat([3, 1, 1])
	elif x.shape[0]==4:
		x=x[:3, :, :]
	return x

class GenericDataLoader(Dataset):#https://www.youtube.com/watch?v=ZoZHd0Zm3RY
	def __init__(self, path, is_test):
		global train_block
		super(GenericDataLoader, self).__init__()
		self.path=path
		self.filenames=[]
		for name in os.listdir(path):
			fullname=os.path.join(path, name)
			if os.path.isfile(fullname):
				name=name.lower()
				if (name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png')) and not name.endswith('bayer.png'):
					self.filenames.append(fullname)
		self.nsamples=len(self.filenames)

		if is_test:
			self.transforms_x=torchvision.transforms.Compose([
				#torchvision.transforms.Lambda(cropTL),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(ensureChannels)
			])
		else:#train
			self.transforms_x=torchvision.transforms.Compose([
				torchvision.transforms.Resize(train_block),#
				torchvision.transforms.RandomCrop(train_block),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(ensureChannels)
			])

	def __len__(self):
		return self.nsamples

	def __getitem__(self, index):
		if index>=self.nsamples:
			return 0
		image_x=Image.open(self.filenames[index])
		image_x=self.transforms_x(image_x)
		image_x=image_x.to(device)
		return image_x


class CompressorModel(nn.Module):
	def __init__(self):
		super(CompressorModel, self).__init__()
		self.qnoise=torch.distributions.Uniform(-0.5, 0.5)
		self.codec=currentcodec.Codec()

	def forward(self, x):
		global g_rate
		nlevels=1<<g_rate
		invamp=1/(nlevels-1)

		y=self.codec.encode(x)

		if type(y) is list:
			yhat=[]
			for idx in range(len(y)):
				y[idx]=torch.mul(y[idx], nlevels-1)#variable amplitude
				y[idx]=torch.add(y[idx], self.qnoise.sample(y[idx].shape).to(device))	# add quantization noise and clamp [0, 1]
				y[idx]=torch.clamp(y[idx], min=0, max=nlevels-1)
				yhat.append(y[idx])
				y[idx]=torch.mul(y[idx], invamp)
		else:
			yhat=torch.mul(y, nlevels-1)
			yhat=torch.add(yhat, self.qnoise.sample(yhat.shape).to(device))
			yhat=torch.clamp(yhat, min=0, max=nlevels-1)
			y=torch.mul(yhat, invamp)

		#cbits=self.codec.predict(yhat)

		xhat=self.codec.decode(y)

		#return xhat, cbits
		return xhat, yhat

	def test(self, x):
		global g_rate
		nlevels=1<<g_rate
		invamp=1/(nlevels-1)

		y=self.codec.encode(x)

		if type(y) is list:
			yhat=[]
			for idx in range(len(y)):
				y[idx]=torch.mul(y[idx], nlevels-1)#variable amplitude
				y[idx]=torch.round(y[idx])
				y[idx]=torch.clamp(y[idx], min=0, max=nlevels-1)
				yhat.append(y[idx])
				y[idx]=torch.mul(y[idx], invamp)
		else:
			yhat=torch.mul(y, nlevels-1)
			yhat=torch.torch.round(yhat)
			yhat=torch.clamp(yhat, min=0, max=nlevels-1)
			y=torch.mul(yhat, invamp)

		#cbits=self.codec.predict(yhat)

		xhat=self.codec.decode(y)

		return xhat, yhat#, cbits*0.125


def load_model(model, filename):#https://github.com/liujiaheng/compression
	with open(filename, 'rb') as file:
		global device
		pretrained_dict = torch.load(file, map_location=device)
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)

def save_tensor_as_grid(x, nrows, name):
	x=torch.clamp(x*255, min=0, max=255).cpu()
	grid=torchvision.utils.make_grid(x, nrow=nrows)
	grid=grid.permute(1, 2, 0)
	grid=grid.numpy()
	grid=grid.astype(np.uint8)
	image=Image.fromarray(grid)
	image.save(name, format='PNG')


def plot_grad_flow(model):#https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/6
	ave_grads=[]
	layers=[]
	for name, param in model.named_parameters():
		if(param.requires_grad) and ('bias' not in name):
			if name.endswith('.weight'):
				name=name[:-7]
			layers.append(name)
			#ave_grads.append(math.log10(param.grad.cpu().abs().mean()))
			ave_grads.append(param.grad.cpu().abs().mean())
	plt.clf()
	plt.plot(ave_grads, alpha=0.3, color='b')
	plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color='k' )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation='vertical')
	plt.xlim(xmin=0, xmax=len(ave_grads))
	plt.xlabel('Layers')
	plt.ylabel('Log10(Av. gradient)')
	plt.title('Gradient flow')
	plt.grid(True)

	plt.ion()
	plt.show()
	plt.pause(0.5)

def get_params(model):#https://discuss.pytorch.org/t/plot-magnitude-of-gradient-in-sdm-with-training-step/101806
	with torch.no_grad():
		params=None
		for name, param in model.named_parameters():
			temp=param.data.view(param.data.nelement()).cpu().detach()
			if params is None:
				params=temp
			else:
				params=torch.cat((params, temp))
	return params


#dataset average dimensions:
#		samples		width/8		height/8	mindim
#AWM		606		399.533003	293.800330
#kodak		24		88		72
#flicker-2W	20K						102

if use_flickr_dataset:
	dataset_train=GenericDataLoader('E:/ML/dataset-flicker_2W', False)
else:
	#dataset_train=GenericDataLoader('E:/ML/dataset-AWM', False)
	dataset_train=GenericDataLoader('E:/ML/dataset-AWM-renamed', False)

dataset_test=GenericDataLoader('E:/ML/dataset-kodak', True)
#dataset_test=GenericDataLoader('E:/ML/dataset-kodak-bayer', True)

train_size=dataset_train.__len__()
test_size=dataset_test.__len__()
train_loader=DataLoader(dataset_train, batch_size=batch_size)
test_loader=DataLoader(dataset_test, batch_size=1)#different size images can't be stacked together

model=CompressorModel()
if pretrained:
	load_model(model, modelname+'.pth.tar')
if use_cuda:
	model=model.cuda()

if model_summary:
	print(model)
	summary(model, (3, 64, 64))#
learned_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('%s has %d parameters'%(modelname, learned_params))

if use_adam:
	optimizer=optim.Adam(params=model.parameters(), lr=lr, eps=0.0001)#https://discuss.pytorch.org/t/nan-after-50-epochs/75835/4
else:
	optimizer=optim.SGD(params=model.parameters(), lr=lr)
if use_cuda:
	scaler=torch.cuda.amp.GradScaler()
loss_func=nn.MSELoss()



def calc_entropy(x, nbits):
	nlevels=1<<nbits
	x=(torch.round(x)+0.5).flatten().cpu().numpy().astype(np.uint8)
	hist, bins=np.histogram(x, bins=nlevels, range=(0, nlevels))
	entropy=0
	den=1/x.size
	for freq in hist:
		if freq!=0:
			prob=freq*den
			entropy+=-prob*np.log2(prob)
	entropy/=nbits
	return entropy

def calc_entropy_differentiable(x):
	global g_rate
	nlevels=1<<g_rate
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
	entropy*=1/g_rate
	return entropy

#class DiffEntropy(nn.Module):
#	def __init__(self):
#		self.conv=nn.Conv2d(3, 3, 2, 2, 0, bias=False)
#		for param in self.conv.parameters:

def calc_groupentropy(x):
	weight_POT=torch.tensor([[[[1, 2, 4, 8], [16, 32, 64, 128]]]], dtype=x.dtype, device=x.device).expand(x.shape[1], -1, -1, -1)
	x=nn.functional.conv2d(x, weight_POT, stride=(2, 4), groups=x.shape[1])

	x=(torch.round(x)+0.5).flatten().cpu().numpy().astype(np.uint8)
	hist, bins=np.histogram(x, bins=256, range=(0, 256))
	entropy=0
	den=1/x.size
	for freq in hist:
		if freq!=0:
			prob=freq*den
			entropy-=prob*np.log2(prob)
	entropy*=0.125
	return entropy

def calc_pairwiseentropy_differentiable(x):#for binary masks only
	weight_POT=torch.tensor([[[[1, 2], [4, 8]]]], dtype=x.dtype, device=x.device).expand(x.shape[1], -1, -1, -1)
	x=nn.functional.conv2d(x, weight_POT, stride=2, groups=x.shape[1])
	
	den=1/(x.shape[2]*x.shape[3])
	entropy=torch.zeros([x.shape[0], x.shape[1]], device=x.device)
	for sym in range(16):#https://github.com/hyk1996/pytorch-differentiable-histogram/blob/master/differentiable_histogram.py
		prob=torch.relu(1-(x-sym).abs())#https://stackoverflow.com/questions/59850816/how-to-implement-tent-activation-function-in-pytorch
		prob=torch.sum(prob, dim=[2, 3])
		prob*=den
		term=-prob*torch.log2(prob)
		term=torch.nan_to_num(term, nan=0, posinf=0, neginf=0)
		entropy+=term
	entropy=torch.mean(entropy)
	entropy*=0.25
	return entropy

def calc_loss(x):
	#global g_rate
	#xhat, cbits=model(x)
	#E=cbits*(g_rate/(8*x.nelement()))

	xhat, yhat=model(x)
	if type(yhat) is list:
		E=torch.zeros(1, device=x.device)
		for yk in yhat:
			E+=calc_pairwiseentropy_differentiable(yk)*yk.nelement()
			#E+=calc_entropy_differentiable(yk)*yk.nelement()
	else:
		E=calc_pairwiseentropy_differentiable(yhat)*yhat.nelement()
		#E=calc_entropy_differentiable(yhat)*yhat.nelement()
	E*=g_rate/(x.nelement()*8)

	D=loss_func(x, xhat)

	if g_rate_factor:
		L=D+g_rate_factor*E
	else:
		L=D

	return L, 255*math.sqrt(D.item()), 1/E.item()


if use_cuda:
	global_free, total=torch.cuda.mem_get_info(device)
	print('Using %f/%f MB of GPU memory'%((total-global_free)/(1024*1024), total/(1024*1024)))

#torch.autograd.set_detect_anomaly(True)
start=time.time()
min_loss=-1
nbatches=(train_size+batch_size-1)//batch_size
#inv_nbatches=batch_size//train_size#X
p0=get_params(model)
distance_prev=0
rmse=0
ratio=0
for epoch in range(epochs):
	progress=0
	rmse=0
	ratio=0
	for x in train_loader:#TRAIN loop
		if use_cuda:
			with torch.cuda.amp.autocast(dtype=torch.float16):#https://pytorch.org/docs/master/notes/amp_examples.html
				L, current_rmse, current_ratio=calc_loss(x)	#1 compute the objective function forward
		else:
			L, current_rmse, current_ratio=calc_loss(x)

		if not math.isfinite(L.item()):
			print('Loss=%f. ABORTING.\t\t'%L.item())
			exit(0)

		model.zero_grad()			#2 cleaning the gradients

		if use_cuda:
			scaler.scale(L).backward()	#3 accumulate the partial derivatives of L wrt params

			if clip_grad:			#4 clip gradient to avoid nan	https://discuss.pytorch.org/t/gradient-clipping-with-torch-cuda-amp/88359/2
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

			scaler.step(optimizer)		#5 step in the opposite direction of the gradient
			scaler.update()
		else:
			L.backward()
			if clip_grad:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
			optimizer.step()

		progress+=x.shape[0]
		rmse+=current_rmse
		ratio+=current_ratio

		print('%d/%d = %5.2f%%  RMSE %16.12f BPP %14.12f\t\t'%(progress, train_size, 100*progress/train_size, current_rmse, 8/current_ratio), end='\r')
	rmse/=nbatches
	ratio/=nbatches

	to_save=not save_records
	record=''
	if min_loss<0 or min_loss>rmse:
		to_save=1
		if min_loss>rmse:
			record=' %10f%%'%(100.*(rmse-min_loss)/min_loss)
		min_loss=rmse
	if to_save:
		torch.save(model.state_dict(), modelname+'.pth.tar')

	if debug_model:
		plot_grad_flow(model)

	t2=time.time()
	print('\t\t\t\t', end='\r')

	pk=get_params(model)
	distance_current=torch.norm(pk-p0).item()
	distance_delta=distance_current-distance_prev
	distance_prev=distance_current

	psnr=20*math.log10(255/rmse)
	print('Epoch %3d [%10f,%10f]  RMSE %16.12f PSNR %13.9f BPP %12.9f  elapsed %10f '%(epoch+1, distance_current, distance_delta, rmse, psnr, 8/ratio, (t2-start)/60), end='')
	print(str(timedelta(seconds=t2-start))+record)

end=time.time()
print('Train elapsed: '+str(timedelta(seconds=end-start)))

if epochs:
	if save_records:
		load_model(model, modelname+'.pth.tar')
	torch.save(model.state_dict(), '%s-%s-psnr%f-bpp%f.pth.tar'%(modelname, time.strftime('%Y%m%d-%H%M%S'), 20*math.log10(255/rmse), 8/ratio))


test_idx=0
rmse=0
usize=0
csize0=0
csize1=0
t_filt=0
for x in test_loader:#TEST loop
	with torch.no_grad():
		t1=time.time()
		#xhat, yhat, current_csize0=model.test(x)
		xhat, yhat=model.test(x)
		t2=time.time()

		current_rmse=255*math.sqrt(loss_func(x, xhat).item())

		current_usize=x.nelement()

		if type(yhat) is list:
			current_csize0=0
			for yk in yhat:
				if g_rate==1:
					entropy=calc_groupentropy(yk)
				else:
					entropy=calc_entropy(yk, g_rate)
				current_csize0+=math.ceil(entropy*yk.nelement()*g_rate/8)
		else:
			if g_rate==1:
				entropy=calc_groupentropy(yhat)
			else:
				entropy=calc_entropy(yhat, g_rate)
			current_csize0=math.ceil(entropy*yhat.nelement()*g_rate/8)

		diff=x-xhat
		entropy=calc_entropy(torch.abs(diff*2), 8)
		current_csize1=current_csize0+math.ceil(x.nelement()*entropy)

		current_psnr=20*math.log10(255/current_rmse)
		print('Test %2d  rmse %16.12f psnr %13.9f  lossy %7d BPP %16.12f  lossless %7d BPP %16.12f  elapsed %9f'%(test_idx+1, current_rmse, current_psnr, current_csize0, current_csize0*8/current_usize, current_csize1, current_csize1*8/current_usize, t2-t1))

		if test_idx+1==21:
			sample=torch.cat((x, xhat, diff+0.5), dim=3)
			current_bpp=current_csize0*8/current_usize
			fn='%s-%s-%d'%(modelname, time.strftime('%Y%m%d-%H%M%S'), test_idx+1)
			save_tensor_as_grid(sample, 1, 'results/'+fn+'-psnr%f-bpp%f.PNG'%(current_psnr, current_bpp))
			if type(yhat) is list:
				save_idx=0
				for yk in yhat:
					if yk.shape[1]>3:
						yk=torch.transpose(yk, 0, 1)#this assumes batch size of 1
						#yk=torch.reshape(yk, [yk.shape[1]//3, 3, yk.shape[2], yk.shape[3]])
					#elif yk.shape[1]==1:
					#	yk=torch.cat((yk, yk, yk), dim=1)
					save_tensor_as_grid(yk, 4, 'results/'+fn+'-L%d.PNG'%save_idx)
					save_idx+=1
			else:
				if yhat.shape[1]>3:
					yhat=torch.transpose(yhat, 0, 1)#this assumes batch size of 1
				save_tensor_as_grid(yhat, 4, 'results/'+fn+'-L.PNG')
		usize+=current_usize
		csize0+=current_csize0
		csize1+=current_csize1
		rmse+=current_rmse
		t_filt+=t2-t1
		test_idx+=1

if test_idx:
	rmse/=test_idx
	print('Average rmse %16.12f psnr %13.9f  size %7d lossy %7d BPP %13.9f  lossless %7d BPP %13.9f  filt %f sec'%(rmse, 20*math.log10(255/rmse), usize/test_idx, csize0/test_idx, csize0*8/usize, csize1/test_idx, csize1*8/usize, t_filt/test_idx))
print('Finished on '+time.strftime('%Y%m%d-%H%M%S'))




