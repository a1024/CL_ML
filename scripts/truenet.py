#2023-01-20Fr
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

#project truenet transforms:
#name		params	sec/epoch	RMSE@100,2bpp
#T01-ReLU-64	 154758	 5.376054	 6.808156349932

#T02-DIV-48	 176844	 4.898037	//no learning 135
#T02-DIV-64	 172300	 5.097936	//no learning epsilon=10: 129,  lr=0.001: 141,  lr=0.0001: 141
#T02-DIV-64sq	 172300	 4.787565	//no learning 137
#T02-MUL-64	 172300	 4.862647	//nan
#T02-MUL-64ep	 172300	 4.331159	//no learning 146
#T02-ATAN2-48	 176844	 4.77475	//no learning 141
#T02-as-T03-64	 172300			//learns

#T03-MUL-48	 176844	 4.727081	10.345335775570
#T04-SQ-64	 154758	 4.53526	11.672446839739

#T05-MulCNN	  22710	 5.0358364	23.367347321258
#T06-D16W16	  66726	 5.7918894	 8.570989782054
#T07-D16W32	 262470	 5.8315806	 5.599996403603
#T08-D24W32	 410438	 6.2579982	 6.054643493880
#T09-D32W32	 558406	 6.765879	 5.520869318294		7.904818214632@1788,0.25bpp	4.713818340789@800,1bpp (>J2K)

#T10-D32W56M	 488726	11.985597	//lr=0.001: nan, lr=0.0001: nan
#T10-D32W40M	 436486	 6.8828357	//sqrt(abs(lo*hi)) 129, sqrt(abs(xy)) 122, sqrt(abs) 107, square() nan

#name		params	sec/epoch	RMSE@100
#T11-D32W48-VR	1309171	 6.0053514	35.720994738894
#T12-D32W48-VR	1262694	 6.734627	22.315794696987




## config ##
import transform12 as currenttransform
modelname='T12'
pretrained=1	# when changing model design, rename old saved model, then assign pretrained=0 for first train
save_records=0

epochs=100
batch_size=24	# increase batch size instead of decreasing learning rate
lr=0.001

clip_grad=0	# enable if got nan
use_adam=1	# disable if got nan
use_flickr_dataset=0




use_cuda=0
device_name='cpu'
if torch.cuda.is_available() and torch.cuda.device_count()>0:
	use_cuda=1
	device_name='cuda:0'
else:
	use_cuda=0
print('%s, LR=%f, Batch=%d'%(device_name, lr, batch_size))
device=torch.device(device_name)

def is_bayer_sample(name):
	k=0
	for c in name:
		if c.isdigit():
			k+=1
		else:
			break
	if k==0:		# must start with a number
		return False

	n2=name[k:].lower()
	return n2=='-original.png'# only one of these must follow

def is_image_sample(name):
	name=name.lower()
	return name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png')

def cropTL(image):
	return torchvision.transforms.functional.crop(image, 0, 0, 64, 64)

def ensureChannels(x):
	if x.size(0)==1:
		x=x.repeat([3, 1, 1])
	global device
	x=x.to(device)
	return x

class GenericDataLoader(Dataset):
	def __init__(self, path, is_test):
		super(GenericDataLoader, self).__init__()
		self.path=path
		self.filenames=[]
		for name in os.listdir(path):
			fullname=os.path.join(path, name)
			if os.path.isfile(fullname) and is_image_sample(name):
				self.filenames.append(fullname)
		self.nsamples=len(self.filenames)

		if is_test:
			self.transforms_x=torchvision.transforms.Compose([
				torchvision.transforms.Lambda(cropTL),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(ensureChannels)
			])
		else:#train
			self.transforms_x=torchvision.transforms.Compose([
				torchvision.transforms.Resize(64),#
				torchvision.transforms.RandomCrop(64),
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
		assert(image_x.size(0)==3 and image_x.size(1)==64 and image_x.size(2)==64)
		return image_x

#https://www.youtube.com/watch?v=ZoZHd0Zm3RY
class BayerDataLoader(Dataset):
	def __init__(self, path, is_test):
		super(BayerDataLoader, self).__init__()
		self.path=path
		self.filenames=[]
		for name in os.listdir(path):
			fullname=os.path.join(path, name)
			if os.path.isfile(fullname) and is_bayer_sample(name):
				self.filenames.append(fullname)
		self.nsamples=len(self.filenames)
		self.filenames.sort()

		if is_test:
			self.transforms_x=torchvision.transforms.Compose([
				torchvision.transforms.Lambda(cropTL),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(ensureChannels)
			])
		else:#train
			self.transforms_x=torchvision.transforms.Compose([
				torchvision.transforms.Resize(64),#
				torchvision.transforms.RandomCrop(64),
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
		assert(image_x.size(0)==3 and image_x.size(1)==64 and image_x.size(2)==64)
		return image_x

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
	grid=grid.astype('uint8')
	image=Image.fromarray(grid)
	image.save(name, format='PNG')

#old models from transform.py:
#MODEL09	 64c, 4*2 layers, 154758 params
# X	MODEL10	 16c, 32*2 layers, 140966 params, 6.501282 GPUsec/epoch
#MODEL11	 32c, 32*2 layers, 558406 params, 8.106746 GPUsec/epoch, 2:03.139256 CPUmin/epoch
# X	MODEL16	 64c, 8*2 layers, 450182 params, 1:14.673212 CPUmin/epoch
#MODEL17	256c, 29*2 layers, 197702 params, 5.824032 RTXsec/epoch, 10.65126 COLABsec/epoch, 4:37.886096 CPUmin/epoch
# X	MODEL18	256c, (14*3+2)*2 layers, 327110 params, 7.0600824 RTXsec/epoch
# X	MODEL19	 64c, 8*2 layers, 241926 params, 3.4412382 RTXsec/epoch
#MODEL20	 64c, 10*2 layers, 389638 params, 3.801721446 RTXsec/epoch
#MODEL21	 64c, 16*2 layers, 832774 params, 4.5877278 RTXsec/epoch
#MODEL22	512c, 28*2 layers, 460938 params, 7.091537 RTXsec/epoch


class CompressorNet(nn.Module):
	def __init__(self, device):
		super(CompressorNet, self).__init__()
		self.device=device
		self.qnoise=torch.distributions.Uniform(-0.5, 0.5)
		self.avpool2=nn.AvgPool2d(2)
		self.avpool4=nn.AvgPool2d(4)
		self.upsample2=nn.Upsample(scale_factor=2, mode='bilinear')
		self.upsample4=nn.Upsample(scale_factor=4, mode='bilinear')
		self.upsample_nearest=nn.Upsample(scale_factor=2)

		self.transform=currenttransform.NLTransform()

	def forward(self, x):
		option=np.random.randint(self.transform.nOptions)

		rate, nbits, downsample=self.transform.prep_rate(option, x.shape, device)

		x=self.transform.encode(x, rate)

		if downsample==2:
			x=self.avpool4(x)
		elif downsample==1:
			x=self.avpool2(x)
		x+=torch.mul(self.qnoise.sample(x.shape).to(self.device), 1/(1<<(nbits-1)))# add quantization noise and clamp [0, 1]
		x=torch.clamp(x, min=0, max=1)
		if downsample==2:
			x=self.upsample4(x)
		elif downsample==1:
			x=self.upsample2(x)

		x=self.transform.decode(x, rate)
		return x, nbits, downsample
	
	# option: {0, ...31}	== downsample<<3 | nbits
	# returns xhat, latent_y, BPP
	def test(self, x, option):
		with torch.no_grad():
			rate, nbits, downsample=self.transform.prep_rate(option, x.shape, device)
			amplitude=1<<(nbits-1)
			invamp=1/amplitude

			y=self.transform.encode(x, rate)

			if downsample==2:
				y=self.avpool4(y)
			elif downsample==1:
				y=self.avpool2(y)
			y=torch.clamp(torch.round(y*amplitude), min=0, max=amplitude)# round and clamp to [0, 2^nbits-1]
			if downsample==2:
				x=self.upsample4(y)
				y=self.upsample_nearest(y)
				y=self.upsample_nearest(y)
			elif downsample==1:
				x=self.upsample2(y)
				y=self.upsample_nearest(y)
			x=torch.mul(x, invamp)

			x=self.transform.decode(x, rate)
		
		return x, y, nbits/(1<<(downsample)), nbits, downsample

#dataset average dimensions:
#		samples		width/8		height/8	mindim
#AWM		606		399.533003	293.800330
#kodak		24		88		72
#flicker-2W	20K						102

if use_flickr_dataset:
	dataset_train=GenericDataLoader('E:/ML/dataset-flicker_2W', False)
else:
	dataset_train=BayerDataLoader('E:/ML/dataset-AWM-bayer', False)
dataset_test=BayerDataLoader('E:/ML/dataset-kodak-bayer', True)
test_size=dataset_test.__len__()
train_size=dataset_train.__len__()
train_loader=DataLoader(dataset_train, batch_size=batch_size)
test_loader=DataLoader(dataset_test, batch_size=test_size)# Kodak dataset: one batch of 24 images

model=CompressorNet(device)
if pretrained:
	load_model(model, modelname+'.pth.tar')
if use_cuda:
	model=model.cuda()

learned_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Model has %d parameters'%learned_params)

if use_adam:
	optimizer=optim.Adam(params=model.parameters(), lr=lr, eps=0.0001)#https://discuss.pytorch.org/t/nan-after-50-epochs/75835/4
else:
	optimizer=optim.SGD(params=model.parameters(), lr=lr)
if use_cuda:
	scaler=torch.cuda.amp.GradScaler()
loss_func=nn.MSELoss()

start=time.time()

min_loss=-1
rmse=0
nbatches=train_size/batch_size
for epoch in range(epochs):
	progress=0
	rmse=0
	av_bpp=0
	for x in train_loader:#train loop
		if use_cuda:
			with torch.cuda.amp.autocast(dtype=torch.float16):#https://pytorch.org/docs/master/notes/amp_examples.html
				xhat, nbits, downsample=model(x)		#1 forward
				J=loss_func(xhat, x)	#2 compute the objective function
		else:
			xhat, nbits, downsample=model(x)
			J=loss_func(xhat, x)
		
		current_mse=J.item()
		current_mse=255*math.sqrt(current_mse)

		if not math.isfinite(current_mse):
			print('current_mse=%f. ABORTING.\t\t'%current_mse)
			exit(0)
		rmse+=current_mse

		model.zero_grad()	#3 cleaning the gradients
		if use_cuda:
			scaler.scale(J).backward()

			if clip_grad:		#clip gradient to avoid nan	https://discuss.pytorch.org/t/gradient-clipping-with-torch-cuda-amp/88359/2
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

			scaler.step(optimizer)
			scaler.update()
		else:
			J.backward()		#4 accumulate the partial derivatives of J wrt params
			if clip_grad:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
			optimizer.step()	#5 step in the opposite direction of the gradient

		progress+=x.size(0)
		current_bpp=nbits/(1<<downsample)
		av_bpp+=current_bpp
		print('%d/%d = %5.2f%% RMSE %16.12f  %f BPP\t\t'%(progress, train_size, 100*progress/train_size, current_mse, current_bpp), end='\r')

	to_save=not save_records
	record=''
	if min_loss<0 or min_loss>rmse:
		to_save=1
		if min_loss>rmse:
			record=' %10f%%'%(100.*(rmse-min_loss)/min_loss)
		min_loss=rmse
	if to_save:
		torch.save(model.state_dict(), modelname+'.pth.tar')

	t2=time.time()
	print('\t\t\t\t', end='\r')
	rmse/=nbatches
	av_bpp/=nbatches
	psnr=20*math.log10(255/rmse)
	print('Epoch %3d RMSE %16.12f PSNR %13.9f  %5.3f BPP elapsed %10f'%(epoch+1, rmse, psnr, av_bpp, (t2-start)/60), end=' ')
	print(str(timedelta(seconds=t2-start))+record)

end=time.time()
print('Train elapsed: '+str(timedelta(seconds=end-start)))

if save_records:
	load_model(model, modelname+'.pth.tar')
torch.save(model.state_dict(), '%s-%s-rmse%.9f.pth.tar'%(modelname, time.strftime("%Y%m%d-%H%M%S"), rmse))

elapsed=0
test_loss=0
test_batches=0
for x in test_loader:
	sample=None
	result=None
	for option in range(model.transform.nOptions):
		t1=time.time()
		xhat, y, BPP, nbits, downsample=model.test(x, option)
		t2=time.time()
		elapsed=t2-t1
		amplitude=1<<(nbits-1)
		invamp=1/amplitude

		with torch.no_grad():
			J=loss_func(xhat, x)
			current_loss=J.item()
			current_loss=255*math.sqrt(current_loss)
			test_loss+=current_loss
			test_batches+=1

			sample_hi=torch.cat((x, xhat), dim=3)			#x,    xhat
			sample_lo=torch.cat((y*invamp, x-xhat+0.5), dim=3)	#code, diff
			sample=torch.cat((sample_hi, sample_lo), dim=2)

		if result==None:
			result=torch.empty(sample.shape[0], sample.shape[1], sample.shape[2], 0)
		result=torch.cat((result, sample.cpu()), dim=3)

	save_tensor_as_grid(result, 1, 'results/%s-%s-%f.PNG'%(modelname, time.strftime("%Y%m%d-%H%M%S"), current_loss))# Kodak dataset: one batch of 24 images

elapsed_str=str(timedelta(seconds=elapsed))
if test_batches!=0:
	test_loss/=test_batches
	elapsed/=test_size
print('Test RMSE %.12f PSNR %13.9f elapsed %s (%f per sample)'%(test_loss, 20*math.log10(255/test_loss), elapsed_str, elapsed))

