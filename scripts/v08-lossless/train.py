#2023-08-17Th
import os
from PIL import Image
import math
import numpy as np
#import random

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

import json
import time
from datetime import timedelta
#import matplotlib
import matplotlib.pyplot as plt
from torchsummary import summary
#import nullcontext




## config ##
from codec16 import Codec
modelname='C16_02'
resume=1		# !!! SET RESUME=1 AFTER FIRST RUN !!!
save_records=0		# this wastes progress

epochs=10
use_optim='adam'	# use 'sgd' if got nan or overfit
lr=0.001000		#always start with high learning rate (0.005 for Adam, 0.1 for SGD), bumping up lr later loses progress
#lr=0.00001*0.75**6	#C01-L3C
batch_size=4		#32, <=24, increase batch size instead of decreasing learning rate
train_crop=512		#128, 192: batch_size=8
cache_rebuild=0		#set to 1 if train_crop was changed
shuffle=True
reduce_lr_on_plateau=0	#slows down when validation flattens
detect_anomalies=0	#enable for debugging CRASHES
force_cpu=0		#GPU is faster

clip_grad=1		# enable if got nan
model_summary=0
plot_grad=0		# FOCUS-STEALING POP-UP		0 disabled   1 plot grad   2 plot log10 grad
weight_decay=0#.0035	# increase if overfit

justexportweights=0

laptop=0
if laptop:
	path_train='C:/Projects/datasets/dataset-train'
	path_val='C:/Projects/datasets/dataset-CLIC30'
	path_test='C:/Projects/datasets/dataset-kodak'
else:
	#path_train='C:/datasets'		# 167056 samples DON'T EXCEED CROP 255
	path_train='C:/datasets2'		#    903 samples
	#path_train='C:/datasets2/CLIC303'	#    303 samples WH 2048*1320
	#path_train='D:/ML/datasets-train'	# caltech256 + flickr + imagenet1000
	#path_train='D:/ML/datasets-train/dataset-caltech256'
	#path_train='D:/ML/dataset-openimages'
	#path_train='D:/ML/dataset-openimages/images'
	#path_train='D:/ML/dataset-CLIC'		# best at 1:1
	#path_train='D:/ML/dataset-AWM'
	#path_train='D:/ML/dataset-CLIC-small'
	#path_train='D:/ML/dataset-AWM-small'
	#path_train='D:/ML/dataset-CLIC30'	#30 samples
	#path_train='D:/ML/dataset-natural'
	#path_train='D:/ML/dataset-kodak'	#CHEAT
	#path_train='D:/ML/dataset-kodim13'	#CHEAT

	path_val='D:/ML/dataset-CLIC30'
	#path_val='D:/ML/dataset-AWM-small'
	#path_val=None

	path_test='D:/ML/dataset-kodak'
	#path_test='D:/ML/dataset-CLIC30'




use_cuda=0
device_name='cpu'
if not force_cpu and torch.cuda.is_available() and torch.cuda.device_count()>0:
	use_cuda=1
	device_name='cuda:0'
print('%s  Epochs %d  %s  %s LR %f  WD %f  Batch %d  Crop %d  Records %d  Train on \'%s\'  Resume %d'%(time.strftime('%Y-%m-%d %H:%M:%S'), epochs, device_name, use_optim, lr, weight_decay, batch_size, train_crop, save_records, path_train, resume))
device=torch.device(device_name)

#if plot_grad:#https://stackoverflow.com/questions/61397176/how-to-keep-matplotlib-from-stealing-focus
#	matplotlib.use('Qt5agg')

def cropTL(image):#PIL image
	return torchvision.transforms.functional.crop(image, 0, 0, train_crop, train_crop)
	#return torchvision.transforms.functional.crop(image, image.height-train_crop, image.width-train_crop, train_crop, train_crop)#64

def ensureChannels(x):
	global device
	if x.shape[0]==1:
		x=x.repeat([3, 1, 1])
	elif x.shape[0]==4:
		x=x[:3, :, :]
	return x

def get_filenames(path, is_test):
	minw=0
	maxw=0
	minh=0
	maxh=0
	filenames=[]
	for root, dirs, files in os.walk(path):
		for name in files:
			fn=os.path.join(root, name)
			try:
				with Image.open(fn) as im:
					width, height=im.size

					if (train_crop!=0 and width>=train_crop and height>=train_crop) or (is_test or train_crop==0):
					#if (train_crop!=0 and width>=train_crop and height>=train_crop) or (is_test or train_crop==0 and width<=512 and height<=512):
						filenames.append(fn)

						count=len(filenames)
						if is_test==0 and count%128==0:
							print('Found %d applicable samples'%(count+1), end='\r')

						if minw==0 or minw>width:
							minw=width
						if maxw==0 or maxw<width:
							maxw=width
						if minh==0 or minh>height:
							minh=height
						if maxh==0 or maxh<height:
							maxh=height
			except:
				continue
	if is_test==0:
		print('Found %d applicable samples'%len(filenames))
		print('dimensions: %dx%d ~ %dx%d'%(minw, minh, maxw, maxh))
	return filenames

class GenericDataLoader(Dataset):#https://www.youtube.com/watch?v=ZoZHd0Zm3RY
	def __init__(self, path, is_test):
		global train_crop, cache_rebuild
		super(GenericDataLoader, self).__init__()
		self.path=path
		self.filenames=[]

		cachename=os.path.join(path, 'cache.txt')
		if is_test:
			self.filenames=get_filenames(path, is_test)
		else:
			try:
				assert(cache_rebuild==0)
				with open(cachename, 'r') as file:
					self.filenames=json.load(file)
				print('Found %d applicable samples'%len(self.filenames))
				assert(type(self.filenames) is list and len(self.filenames)>0)
			except:
				print('Rebuilding cache')
				self.filenames=get_filenames(path, is_test)
				assert(len(self.filenames)>0)
				with open(cachename, 'w') as file:
					json.dump(self.filenames, file, indent=0)

		if is_test==1:#test at 1:1
			self.transforms_x=torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(ensureChannels)
			])
		elif is_test==2:#validation
			if train_crop!=0:
				self.transforms_x=torchvision.transforms.Compose([
					torchvision.transforms.Lambda(cropTL),
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Lambda(ensureChannels)
				])
			else:
				self.transforms_x=torchvision.transforms.Compose([
					torchvision.transforms.ToTensor(),
					torchvision.transforms.Lambda(ensureChannels)
				])
		elif train_crop!=0:#train
			self.transforms_x=torchvision.transforms.Compose([
				#torchvision.transforms.Lambda(cropTL),#
				#torchvision.transforms.Resize(train_crop),#
				torchvision.transforms.RandomCrop(train_crop),
				torchvision.transforms.RandomHorizontalFlip(),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(ensureChannels)
			])
		else:#train 1:1, batch_size=1
			self.transforms_x=torchvision.transforms.Compose([
				torchvision.transforms.RandomHorizontalFlip(),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(ensureChannels)
			])

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, index):
		if index>=len(self.filenames):
			return None
		image_x=Image.open(self.filenames[index])
		image_x=self.transforms_x(image_x)
		#image_x=image_x.to(device)#https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
		return image_x, self.filenames[index]

	def get_size(self, index):
		return os.path.getsize(self.filenames[index])


def color_transform_YCmCb_int(x):#YCoCg-R with g & b swapped
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	g+=(r*0.5).int()
	b-=g
	g+=(b*0.5).int()

	x=torch.cat((r, g, b), dim=1)#{Co, Y, Cb}
	return x

def color_transform_YCmCb(x):#YCoCg-R with g & b swapped
	r, g, b=torch.split(x, 1, dim=1)

	r-=g
	g+=r*0.5
	b-=g
	g+=b*0.5

	x=torch.cat((r, g, b), dim=1)#{Co, Y, Cb}
	return x


def color_transform_YCoCg(x):#YCoCg-R
	r, g, b=torch.split(x, 1, dim=1)

	r-=b
	b+=r*0.5
	g-=b
	b+=g*0.5
	x=torch.cat((r, g, b), dim=1)#{Co, Y, Cg}

	#print('YCoCg [%f, %f] [%f, %f] [%f, %f]'%(y.min(), y.max(), co.min(), co.max(), cg.min(), cg.max()))
	return x


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



def exportweights(filename, model):
	maxdim=0
	with open(filename, 'w') as file:
		for name, param in model.named_parameters():
			file.write(name)
			dim=param.dim()
			if maxdim<dim:
				maxdim=dim
			for size in param.shape:
				file.write('\t%d'%size)
			file.write(':\n')
			if dim==4:
				for filt in param:
					for kernel in filt:
						for row in kernel:
							for val in row:
								#file.write('\t'+val.item().hex())
								file.write(' %22.18f,'%val.item())
							file.write('\n')
						file.write('\n')
					file.write('\n')
				file.write('\n')
			elif dim==3:
				for matrix in param:
					for row in matrix:
						for val in row:
							#file.write('\t'+val.item().hex())
							file.write(' %22.18f,'%val.item())
						file.write('\n')
					file.write('\n')
				file.write('\n')
			elif dim==2:
				for row in param:
					for val in row:
						#file.write('\t'+val.item().hex())
						file.write(' %22.18f,'%val.item())
					file.write('\n')
				file.write('\n')
			elif dim==1:
				for val in param:
					#file.write('\t'+val.item().hex())
					file.write(' %22.18f,'%val.item())
				file.write('\n')
	return maxdim


def plot_grad_flow(model):#https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/6
	avgrad=0
	nterms=0
	av_grads=[]
	layers=[]
	for name, param in model.named_parameters():
		if(param.requires_grad) and ('bias' not in name):
			if name.endswith('.weight'):
				name=name[:-7]
			layers.append(name)
			val=param.grad.cpu().abs().mean()
			if not math.isfinite(val):
				print('%s mean abs grad = %f'%(name, val))
			else:
				avgrad+=val
				nterms+=1
			if plot_grad==2:
				val=math.log10(val)
			av_grads.append(val)
	if nterms:
		avgrad/=nterms
	plt.clf()
	plt.plot(av_grads, alpha=0.3, color='b')
	plt.hlines(0, 0, len(av_grads)+1, linewidth=1, color='k' )
	plt.xticks(range(0, len(av_grads), 1), layers, rotation='vertical')
	plt.xlim(xmin=0, xmax=len(av_grads))
	plt.xlabel('Layers')
	plt.ylabel('Av. gradient')
	#plt.ylabel('Log10(Av. gradient)')
	plt.title('Gradient flow (av abs grad = %f)'%avgrad)
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


model=Codec()
if resume:
	load_model(model, modelname+'.pth.tar')
if use_cuda:
	model=model.cuda()
#if torch.__version__ >= version.parse('2.0.0'):#doesn't work on Windows yet
#	model=torch.compile(model, mode='reduce-overhead')

if justexportweights:
	filename=modelname+'-'+time.strftime('%Y%m%d-%H%M%S')+'.txt'
	print('Exporting weights as '+filename)
	#exportC35(filename, model)
	exportweights(filename, model)
	#print('Maxdim: %d'%maxdim)
	print('Done.')
	exit(0)

if model_summary:
	print(model)
	summary(model, (3, 64, 64))#
learned_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('%s has %d parameters'%(modelname, learned_params))


#dataset average dimensions:
#		samples		width		height		mindim
#kodak		24		88		72
#CLIC-small	303		512		330		const
#AWM-small	606		399.533003	293.800330
#AWM		601		3196.264024	2350.40264
#flicker-2W	20745						102
#caltech256	30608

dataset_train=GenericDataLoader(path_train, 0)

if path_val is None:
	dataset_val=None
else:
	dataset_val=GenericDataLoader(path_val, 2)

dataset_test=GenericDataLoader(path_test, 1)

train_size=dataset_train.__len__()
test_size=dataset_test.__len__()
train_loader=DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)#try num_workers=16
test_loader=DataLoader(dataset_test, batch_size=1)#different resolutions at 1:1 can't be stacked

if use_optim=='sgd':
	optimizer=optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)
elif use_optim=='adam':
	optimizer=optim.Adam(params=model.parameters(), lr=lr, eps=0.0001, weight_decay=weight_decay)#https://discuss.pytorch.org/t/nan-after-50-epochs/75835/4
elif use_optim=='adadelta':
	optimizer=optim.Adadelta(params=model.parameters(), lr=lr, eps=0.0001, weight_decay=weight_decay)
elif use_optim=='rmsprop':
	optimizer=optim.RMSprop(params=model.parameters(), lr=lr, eps=0.0001, weight_decay=weight_decay)

if reduce_lr_on_plateau:
	scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer)
if use_cuda:
	scaler=torch.cuda.amp.GradScaler()




def calc_csize(p0, bits):
	p0=torch.clamp(p0, 1/0x100, 0xFF/0x100)
	csize=-(bits<0).float()*torch.log2(p0)-(bits>0).float()*torch.log2(1-p0)
	return torch.sum(csize)

def calc_loss(x):
	x=x.to(device)

	x*=2
	x-=1
	x=color_transform_YCmCb(x)
	x=torch.fmod(x+1, 2)-1		#[-1, 1]

	return model(x)



if detect_anomalies:
	torch.autograd.set_detect_anomaly(True)
start=time.time()
min_loss=-1
min_train_loss=-1
nbadepochs=0
nbatches=(train_size+batch_size-1)//batch_size
#inv_nbatches=batch_size//train_size#X
p0=get_params(model)
distance_prev=0


if dataset_val is not None:#initial validation
	model.epoch_start()
	with torch.no_grad():
		nval=len(dataset_val)
		for k in range(nval):	#validation EPOCH loop
			x, fname=dataset_val[k]
			x=x[None, :, :, :]
			loss, msg=calc_loss(x)

			print('%6d/%6d =%6.2f%%  %s'%(k+1, nval, 100*(k+1)/nval, msg), end='\r')
		val_loss, val_msg=model.epoch_end()
	print('E%4d [%10f,%10f]  V %s  elapsed %10f '%(0, 0, 0, val_msg, (time.time()-start)/60))

for epoch in range(epochs):		#TRAIN loop
	it=0
	progress=0

	model.epoch_start()
	for x, fname in train_loader:		#train EPOCH loop
		if use_cuda:
			with torch.cuda.amp.autocast(dtype=torch.float16):#https://pytorch.org/docs/master/notes/amp_examples.html
				loss, msg=calc_loss(x)		#1 compute the objective function f o r w a r d
		else:
			loss, msg=calc_loss(x)

		it+=1
		if not math.isfinite(loss.item()):
			print('\nLoss=%f. SKIPPING BATCH %d.'%(loss.item(), it))
			continue

		model.zero_grad()			#2 cleaning the gradients

		if use_cuda:
			scaler.scale(loss).backward()	#3 accumulate the partial derivatives of L wrt params

			if clip_grad:			#4 clip gradient to avoid nan	https://discuss.pytorch.org/t/gradient-clipping-with-torch-cuda-amp/88359/2
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

			scaler.step(optimizer)		#5 step in the opposite direction of the gradient
			scaler.update()
		else:
			loss.backward()
			if clip_grad:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
			optimizer.step()

		progress+=x.shape[0]

		print('%6d/%6d =%6.2f%%  %s'%(progress, train_size, 100*progress/train_size, msg), end='\r')
	train_loss, train_msg=model.epoch_end()

	if dataset_val is not None:
		model.epoch_start()
		with torch.no_grad():#validation
			nval=len(dataset_val)
			for k in range(nval):	#validation EPOCH loop
				x, fname=dataset_val[k]
				x=x[None, :, :, :]
				loss, msg=calc_loss(x)

				print('%6d/%6d =%6.2f%%  %s'%(k+1, nval, 100*(k+1)/nval, msg), end='\r')
		val_loss, val_msg=model.epoch_end()
		if reduce_lr_on_plateau:
			scheduler.step(val_loss)
	else:
		val_loss=train_loss
		val_msg=train_msg

	record=''
	if reduce_lr_on_plateau:
		record+='  LR %8lf'%optimizer.param_groups[0]['lr']
	to_save=not save_records
	bad_epoch=1

	if min_train_loss<0 or min_train_loss>train_loss:
		if min_train_loss>train_loss:
			to_save=1
			bad_epoch=0
			nbadepochs=0
			record+=' T%10f%%'%(100.*(train_loss-min_train_loss)/min_train_loss)
		min_train_loss=train_loss

	if min_loss<0 or min_loss>val_loss:
		if min_loss>val_loss:
			to_save=1
			bad_epoch=0
			nbadepochs=0
			record+=' V%10f%%'%(100.*(val_loss-min_loss)/min_loss)
		min_loss=val_loss
	if to_save:
		torch.save(model.state_dict(), modelname+'.pth.tar')

	if use_cuda and epoch==0:
		global_free, total=torch.cuda.mem_get_info(device)
		record+=' GPU %f/%f MB'%((total-global_free)/(1024*1024), total/(1024*1024))

	if plot_grad:
		plot_grad_flow(model)

	t2=time.time()
	print('\t\t\t\t', end='\r')

	pk=get_params(model)
	distance_current=torch.norm(pk-p0).item()
	distance_delta=distance_current-distance_prev
	distance_prev=distance_current

	nbadepochs+=bad_epoch
	#if nbadepochs>=10 or nbadepochs>=2 and abs(distance_delta)<0.0001:#X
	#	nbadepochs=0
	#	record+=' +noise'
	#	with torch.no_grad():
	#		for param in model.parameters():
	#			param.add_(torch.randn(param.size(), device=param.device)*0.01)

	print('E%4d [%10f,%10f]  T %s  V %s  elapsed%11f %15s %s'%(epoch+1, distance_current, distance_delta, train_msg, val_msg, (t2-start)/60, str(timedelta(seconds=t2-start)), record))

end=time.time()
print('Train elapsed: '+str(timedelta(seconds=end-start)))

if epochs:
	if save_records:
		load_model(model, modelname+'.pth.tar')
	torch.save(model.state_dict(), '%s-%s-%s.pth.tar'%(modelname, time.strftime('%Y%m%d_%H-%M-%S'), val_msg))

'''
cr_kodak_jxl=[
	2.550446894,#01
	2.829095618,#02
	3.574647580,#03
	2.834874555,#04
	2.349924501,#05
	2.775185264,#06
	3.290739940,#07
	2.307917148,#08
	3.077120521,#09
	3.007434148,#10
	2.840198392,#11
	3.296984874,#12
	2.124886969,#13
	2.525499095,#14
	3.096595354,#15
	3.203023688,#16
	3.026150665,#17
	2.279407332,#18
	2.670735849,#19
	3.602693664,#20
	2.652199622,#21
	2.468714358,#22
	3.072680150,#23
	2.659542377 #24
]
cr_kodak_png=[
	1.601692326,#01
	1.908830978,#02
	2.345746966,#03
	1.850625635,#04
	1.501569481,#05
	1.905858062,#06
	2.082998718,#07
	1.496122871,#08
	2.023760549,#09
	1.987736388,#10
	1.899523850,#11
	2.221458917,#12
	1.433852916,#13
	1.704198636,#14
	1.925698111,#15
	2.208057322,#16
	1.959294311,#17
	1.510535286,#18
	1.756798456,#19
	2.395409189,#20
	1.851732436,#21
	1.680482072,#22
	2.115596238,#23
	1.669950467 #24
]
'''

test_idx=0
t_filt=0
model.epoch_start()
for x, fname in test_loader:		#TEST loop
	with torch.no_grad():
		t1=time.time()
		loss, msg=calc_loss(x)
		t2=time.time()

		print('T%4d  %s  elapsed %9f sec'%(test_idx+1, msg, t2-t1))

		#if test_idx+1==21:#save preview
		#	#sample=torch.cat((r, r2), dim=3)
		#
		#	#size_mask*=10/255
		#	#tr, bl, br=torch.split(size_mask, 3, dim=1)
		#	#tl=torch.zeros(tr.shape[0], tr.shape[1], tr.shape[2], bl.shape[3], dtype=tr.dtype, device=tr.device)
		#	#top=torch.cat((tl, tr), dim=3)
		#	#bot=torch.cat((bl, br), dim=3)
		#	#sample=torch.cat((top, bot), dim=2)
		#
		#	stuff=torch.cat((luma.cpu(), co.cpu(), co_res.cpu(), cb.cpu(), cb_res.cpu()), dim=3)
		#	stuff=torch.cat((stuff.cpu(), stuff.cpu(), stuff.cpu()), dim=1)
		#	sample=torch.cat((x.cpu(), stuff*0.5+0.5), dim=3)
		#
		#	#sample=torch.cat((x2+0.5, x2-mean+0.5, mean+0.5, size_mask*10/255), dim=3)
		#
		#	#size_mask=size_mask.repeat([1, 3, 1, 1])
		#	#top=torch.cat((x2+0.5, mean+0.5, lgconf*10/255), dim=3)
		#	#bot=torch.cat((alpha, x2-mean+0.5, size_mask*10/255), dim=3)
		#	#sample=torch.cat((top, bot), dim=2)
		#	#sample=torch.cat((x2+0.5, mean+0.5, lgconf*10/255, alpha, size_mask*10/255), dim=3)
		#
		#	fn='%s-%s-%d'%(modelname, time.strftime('%Y%m%d-%H%M%S'), test_idx+1)
		#	save_tensor_as_grid(sample, 1, 'results/'+fn+'-rmse%f.PNG'%loss)
		#	#save_tensor_as_grid(sample, 1, 'results/'+fn+'-cr%f.PNG'%current_ratio)
		t_filt+=t2-t1
		test_idx+=1
if test_idx:
	loss, msg=model.epoch_end()
	print('Total  %s  Elapsed %f sec'%(msg, t_filt))
model.checkpoint_msg()
print(time.strftime('%Y-%m-%d %H:%M:%S'))




