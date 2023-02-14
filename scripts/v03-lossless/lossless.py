#2023-02-01We
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
import matplotlib.pyplot as plt
from torchsummary import summary

#name		params		sec/epoch	RMSE@100
#M01-K4		 338648		6.8260512	0.450431
#M02-K6		 806232		4.6558596	0.494994590053
#M03-K2		1006104		5.2097142	0.515215273677
#M04-K6		2101344		6.2042316	0.439545200406
#M05-K7		2190416		6.7232346	0.463457267770

#M06-K6		 517472		6.5869116	0.458793339580

#M07-conf	 515258		5.6286504	0.050271125164

#name			params		sec/epoch	train@100		best test
#M08-pred		 514889		 5.0347398	0.047380689846		0.033638808876@1700
#M09-large		 639689		 6.6802074	0.280827334676		0.172939360142
#M10-M08-bn-drop	 518933		 6.567411	0.398609741133		0.208983749151
#M11-M08-blockbn	 516409		 5.8487718	0.102909903307		0.073111273348@100
#M12-M08-drop		 514889		 5.6818374	0.310803648632		0.035476181656@100
#M13-shallow		 772489		 5.0270928	0.045058444869		0.027763973922@100	ratio=1.3720283857037

#M14-row		 262019		 5.1864354	0.066710988249		ratio=1.001061384226, pred=62.63% @400

#M15-att-ewm		 374595		26.0312046	0.079244662372		r=1.003878769661 p=61.62% @100 x 0.0001
#M15-att-mm		 374595		 6.79136@32	0.080589812309		r=1.107674644810 p=56.53%	//1:50.863130@128 //31.67 mins@256?
#M16-att-mm-neck	 291555		26.67 mins@224?




## config ##
import model15 as currentmodel
modelname='M15'
pretrained=1	# when changing model design, rename old saved model, then assign pretrained=0 for first train
save_records=0

epochs=100
lr=0.001	#always start with high rate
batch_size=24	# <=24, increase batch size instead of decreasing learning rate
train_block=64

clip_grad=1	# enable if got nan
use_adam=1	# disable if got nan
use_flickr_dataset=0

model_summary=0
debug_model=1




use_cuda=0
device_name='cpu'
if torch.cuda.is_available() and torch.cuda.device_count()>0:
	use_cuda=1
	device_name='cuda:0'
else:
	use_cuda=0
print('%s, LR=%f, Batch=%d, Records=%d'%(device_name, lr, batch_size, save_records))
device=torch.device(device_name)

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
			ave_grads.append(param.grad.cpu().abs().mean())
	plt.plot(ave_grads, alpha=0.3, color='b')
	plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color='k' )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation='vertical')
	plt.xlim(xmin=0, xmax=len(ave_grads))
	plt.xlabel('Layers')
	plt.ylabel('average gradient')
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

model=currentmodel.Predictor()
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


start=time.time()
min_loss=-1
rmse=0
nbatches=(train_size+batch_size-1)//batch_size
p0=get_params(model)
distance_prev=0
for epoch in range(epochs):
	progress=0
	for x in train_loader:#TRAIN loop
		if use_cuda:
			with torch.cuda.amp.autocast(dtype=torch.float16):#https://pytorch.org/docs/master/notes/amp_examples.html
				pred, truth=model(x)
				L=loss_func(pred, truth)	#1 compute the objective function forward
		else:
			pred, truth=model(x)
			L=loss_func(pred, truth)

		if not math.isfinite(L.item()):
			print('Loss=%f. ABORTING.\t\t'%L.item())
			exit(0)

		current_rmse=math.sqrt(L.item())

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

		progress+=x.size(0)
		rmse+=current_rmse

		print('%d/%d = %5.2f%%  RMSE %16.12f\t\t'%(progress, train_size, 100*progress/train_size, current_rmse), end='\r')
	rmse/=nbatches

	to_save=not save_records
	record=''
	if min_loss<0 or min_loss>rmse:
		to_save=1
		if min_loss>rmse:
			record=' %10f%%'%(100.*(rmse-min_loss)/min_loss)
		min_loss=rmse
	if to_save:
		torch.save(model.state_dict(), modelname+'.pth.tar')

	memusage=''
	if use_cuda:
		global_free, total=torch.cuda.mem_get_info(device)
		memusage=' %d'%(total-global_free)

	if debug_model:
		plot_grad_flow(model)

	t2=time.time()
	print('\t\t\t\t', end='\r')

	pk=get_params(model)
	distance_current=torch.norm(pk-p0).item()
	distance_delta=distance_current-distance_prev
	distance_prev=distance_current

	print('Epoch %3d [%10f,%10f]  RMSE %16.12f  elapsed %10f '%(epoch+1, distance_current, distance_delta, rmse, (t2-start)/60), end='')
	print(str(timedelta(seconds=t2-start))+memusage+record)

end=time.time()
print('Train elapsed: '+str(timedelta(seconds=end-start)))

if epochs:
	if save_records:
		load_model(model, modelname+'.pth.tar')
	torch.save(model.state_dict(), '%s-%s-rmse%.9f.pth.tar'%(modelname, time.strftime('%Y%m%d-%H%M%S'), rmse))


import pyentropy
test_idx=0
uncbits, compbits, predbits, elapsed=0, 0, 0, 0
t_filt=0
t_enc=0
for x in test_loader:#TEST loop
	with torch.no_grad():
		t1=time.time()

		pred, truth=model(x)

		#pred=x[:, :, :model.rfield, :]
		#for ky in range(model.rfield, x.shape[2]):
		#	temp, _=model.predict(x, ky)
		#	pred=torch.cat((pred, temp), dim=2)
		t2=time.time()

		result=truth-pred
		diff=(result*255).cpu().numpy().astype(np.uint8)#subtract x

		#x=x[:, :, model.rfield:, :]#remove header
		#pred=pred[:, :, model.rfield:, :]
		#code=((x-pred)*255).int().cpu().numpy()#subtract x
		#code=code.astype(np.uint8)
		#t3=time.time()

		for kb in range(x.shape[0]):
			temp_diff=diff[kb]
			temp_diff=temp_diff.reshape(temp_diff.size)
			temp_uncbits, temp_compbits, temp_predbits, temp_elapsed=pyentropy.testencode(temp_diff, 3, 8, 1)
			#print('uncbits %d  compbits %d  predbits %d  elapsed %d'%(temp_uncbits, temp_compbits, temp_predbits, temp_elapsed))#
			print('Test %2d comp %8d ratio %16.12f pred %.2f%%'%(test_idx+1, temp_compbits//8, temp_uncbits/temp_compbits, 100*temp_predbits/temp_uncbits))
			uncbits+=temp_uncbits
			compbits+=temp_compbits
			predbits+=temp_predbits
			elapsed+=temp_elapsed
		t4=time.time()

		if test_idx+1==21:
			result+=0.5
			fn='results/%s-%s-%d-%f.PNG'%(modelname, time.strftime('%Y%m%d-%H%M%S'), test_idx+1, temp_uncbits/temp_compbits)
			save_tensor_as_grid(result, 1, fn)

		t_filt+=t2-t1
		t_enc+=t4-t2
		test_idx+=1

print('Average comp %8d ratio %16.12f pred %.2f%% filt %f enc %f, %d cycles'%(compbits//8, uncbits/compbits, 100*temp_predbits/temp_uncbits, t_filt, t_enc, elapsed))

#print('sanity check...')
#with torch.no_grad():
#	pred=np.zeros(shape=[3*512*768], dtype=np.uint8)
#	t1=time.time()
#	uncbits, compbits, predbits, elapsed=pyentropy.testencode(pred, 3, 8, 0)
#	#csize, predicted=abac.test_encode(pred, 8)
#	t2=time.time()
#	print('ABAC ZEROS: comp %d ratio %16.12f pred %6.2f%% enc %d %f'%(compbits//8, pred.size*8/compbits, 100*predbits/(pred.size*8), elapsed, t2-t1))
#	#print('ABAC ZEROS: comp %8d ratio %16.12f pred %6.2f%% enc %f'%(csize, pred.size/csize, 100*predicted/(pred.size*8), t2-t1))




