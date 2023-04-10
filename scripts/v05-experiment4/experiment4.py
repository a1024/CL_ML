#2023-02-18Sa
import os
import glob
from PIL import Image
import math
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
#from packaging import version

import json
#import random
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
#C11-3c			 902976		 9.3268104	15.434808229835		PSNR 17.23  BPP 0.3260  @200	PSNR 16.93 BPP 0.2145 @1400 grainy banding (train PSNR 27.49 @1400)
#C12-8c			 999820		10.4657478	12.186279430929		PSNR 17.01  BPP 0.4786  @200 grainy banding (train PSNR 28.16 @400)
#C13-halfBPP		 997272		12.8666733	14.731614360013		PSNR 16.13  BPP 0.4480  @200 blurry, checkerboard
#C14-simple		 849816		13.6398		17.329631543928		PSNR 16.28  BPP 0.4501  @400 blurry, banding

#name			params		sec/epoch	train@20		best test
#C15-pred45		 351570		240.80016	CR<0.8739 stuck
#C16-pred15		  76070		 81.809004	CR<0.8739 stuck		pred has image
#C17-W192D21		 436454		109.49262	CR 1.109513600708 SGD	CR 0.999114900390@60 horizontal noise	train CR 1.306131966123@80		//212582 stuck
#C18-W80D45		 212214		201.5923424	CR 1.196688348439	CR 0.851057301155@200			train CR 1.376555901783@200

#name			params		sec/epoch	train@100		best test
#C19-causal64x16	 523142		  4.96311	CR 1.302800046032@100	CR 0.471393422187@500, 0.957793689137@10F, CONFIDENCE IS INVERTED	train CR 1.455082490775@500	200 plain conv > 300 with shortcuts
#C20-cv2-NON-CAUSAL X	  67206		  4.677681	CR 1.543508560776@100	CR 0.787544201094@100
#C21-cv3-NON-CAUSAL X	 562962		  5.135544	CR 1.112669424226@100	CR 1.117531873724@200 good sky bad rocks, 0.939852510542@1000, 0.658796771769@1050	train CR 1.9748 test 0.6588 @1050 NON-CAUSAL
#C22-causalv4		 314776		  5.4297054	CR 1.124200884514@100	CR 1.076068913096@100 high contrast pred	train 1.6754 test 0.4906 @250 overfit
#C23-simpledeep		  //71766 overfit after 8~41  //679110 overfit after 35  //346758 overfit after 83  //4806 overfit after 56  //41862 overfit after 71  //115974 overfit after 20  //227142 overfit after 25
#C24-simpleshallow	   //1206 overfit //3558 overfit after 164  //1357062 overfit after 43
#C25-W16D16-diff-0.5	  34134		  5.061819	CR 1.414708764189@100	CR 1.354303@800		2D differentiation
#C26-W64D16-diff-0	 523590		  5.585556	CR 1.659290826384@100	CR 1.522747085930@100		//CR 2.095254748172@100	CR 1.907240800971@100 X wrong size estimation as distribution was [0, 1]
#C27-W32D32		 281766		  3.826524	CR 1.769571716621@100	CR 1.335184939811@200
#C27-W96D32		2504166		 13.0522272	CR 1.774571292465@100	CR 1.390918725199@400
#C28-perpixel		 277708
#C28-perpixel-W32D08	  71276		 17.2978632 	CR 2.085821703291@100	CR 2.024069@198		1:1 training	//14.4233424 min/E @caltech	1.935536698225@739 mean [0, 1]
#C29-pp-W24D16		  92244		 25.3873584	CR 2.365603678686@100	CR 2.042271328465@100

#squeeze wavelet
#name			params		sec/epoch		peak train CR	peak val CR	peak test CR
#C30-W24D16	X	  78246		  4.8561966								//CR 1.821132290670@100	CR 1.670257791180@100 X  squeeze was broken
#C31-W24D16-a,x,y,xy	 228798		102.348048@CLIC@1	1.941487@98	1.739492@59	1.723313@100	//CR 1.987160219959@100	CR 2.290226@728  2.228110@196 X  squeeze was broken
#C31-W24D16-no-XYB	 228798		103.204945@CLIC@1			1.585646@14, NAN, OVERFIT
#C31-W24D16-diff-av	 226854		103.593176@CLIC@1			1.684674@14, OVERFIT	1.664745@14,wdecay=0.001 OVERFIT

#C32-W16D08		  46422		330.480444@train@32			1.569486@4  wdecay=0.01  1.425418@8  1.492217@17		test 1.429201@20
#C31-dropout2d		 226854		
#C32-W16D08

#C33-W32D08	X	 140040		 94.598373@CLIC@1	2.633249@98	2.344271@96	2.291659@100		//124416//410916
#C33-W32D08	X	 140040		 96.726323@CLIC@1	3.657146@175	3.093322@173	3.130962@175	XGZ

#name			params		sec/epoch		peak train CR	peak val CR	peak test CR
#C34-W32D08-per-row	 117702		 75.766336@CLIC@1	2.522278@113	2.136998@92	2.183298@113
#		CLIC30 validation dataset
#C35-W32D08-per-px	 107640		 97.671703@CLIC@1	2.537100@180	2.450299@146	2.199998@180			//2.119924@24 kodim21
#C36-W64D08-multi-Gauss
#		range bug fixed

#name			params		sec/epoch		peak train CR	peak val CR	peak test CR
#C35-W32D08-per-px-erf	 107640		105.39082@CLIC@1	3.156985@54	3.070384@54	2.885509@50




## config ##
import codec35 as currentcodec
modelname='C35'
pretrained=1	# assign pretrained=0 when training first time
save_records=0

epochs=25
lr=0.0001		#always start with high learning rate 0.001
batch_size=1		# <=24, increase batch size instead of decreasing learning rate
train_block=0		#256: batch_size=8
cache_rebuild=0		#set to 1 if train_block was changed
#use_dataset='awm'	#'flickr' / 'clic' / 'awm' / 'kodak' / 'caltech' / 'imagenet'

clip_grad=1		# enable if got nan
use_SGD=0		# enable if got nan
model_summary=0
plot_grad=0		# 0 disabled   1 plot grad   2 plot log10 grad
weight_decay=0.003	# increase if overfit
use_dropout=0

g_rate=8
#g_rate_factor=0.001

#path_train='E:/ML/datasets-train'	# caltech256 + flickr + imagenet1000
#path_train='E:/ML/datasets-train/dataset-caltech256'
path_train='E:/ML/dataset-CLIC'	# best at 1:1
#path_train='E:/ML/dataset-AWM'
#path_train='E:/ML/dataset-CLIC-small'
#path_train='E:/ML/dataset-AWM-small'
#path_train='E:/ML/dataset-CLIC30'	#30 samples
#path_train='E:/ML/dataset-natural'

#path_val=None
path_val='E:/ML/dataset-CLIC30'

path_test='E:/ML/dataset-kodak'
#path_test='E:/ML/dataset-CLIC30'

justexportweights=0




use_cuda=0
device_name='cpu'
if torch.cuda.is_available() and torch.cuda.device_count()>0:
	use_cuda=1
	device_name='cuda:0'
else:
	use_cuda=0
print('Started on %s, %s, LR=%f, Batch=%d, Records=%d, SGD=%d, dataset=\'%s\', pretrained=%d, wd=%f'%(time.strftime('%Y-%m-%d %H:%M:%S'), device_name, lr, batch_size, save_records, use_SGD, path_train, pretrained, weight_decay))
device=torch.device(device_name)

if plot_grad:#https://stackoverflow.com/questions/61397176/how-to-keep-matplotlib-from-stealing-focus
	matplotlib.use('Qt5agg')

def cropTL(image):
	return torchvision.transforms.functional.crop(image, 0, 0, 64, 64)

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

					if (train_block!=0 and width>=train_block and height>=train_block) or (is_test or train_block==0):
					#if (train_block!=0 and width>=train_block and height>=train_block) or (is_test or train_block==0 and width<=512 and height<=512):
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
		global train_block, cache_rebuild
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
		#self.shuffle()

		#for filename in glob.iglob(path, recursive=True):
    		#	print(os.path.abspath(filename), os.stat(filename).st_uid)

		#for name in os.listdir(path):
		#	fullname=os.path.join(path, name)
		#	if os.path.isfile(fullname):
		#		name=name.lower()
		#		if (name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png')) and not name.endswith('bayer.png'):
		#			self.filenames.append(fullname)

		if is_test:
			self.transforms_x=torchvision.transforms.Compose([
				#torchvision.transforms.Lambda(cropTL),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(ensureChannels)
			])
		elif train_block!=0:#train
			self.transforms_x=torchvision.transforms.Compose([
				#torchvision.transforms.Resize(train_block),#
				torchvision.transforms.RandomCrop(train_block),
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

	#def shuffle(self):
	#	self.filenames=random.shuffle(self.filenames)

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, index):
		if index>=len(self.filenames):
			return None
		image_x=Image.open(self.filenames[index])
		image_x=self.transforms_x(image_x)
		#image_x=image_x.to(device)#https://stackoverflow.com/questions/53998282/how-does-the-number-of-workers-parameter-in-pytorch-dataloader-actually-work
		return image_x
	
	def get_size(self, index):
		return os.path.getsize(self.filenames[index])


class CompressorModel(nn.Module):
	def __init__(self):
		super(CompressorModel, self).__init__()
		#self.qnoise=torch.distributions.Uniform(-0.5, 0.5)

		global device
		self.wx=torch.tensor([
			[[[0.5, -0.5]]],
			[[[0.5, -0.5]]],
			[[[0.5, -0.5]]]
		], dtype=torch.float, device=device)
		self.wy=torch.tensor([
			[[[0.5], [-0.5]]],
			[[[0.5], [-0.5]]],
			[[[0.5], [-0.5]]]
		], dtype=torch.float, device=device)
		self.wxy=torch.tensor([
			[[[0.25, -0.25], [-0.25, 0.25]]],
			[[[0.25, -0.25], [-0.25, 0.25]]],
			[[[0.25, -0.25], [-0.25, 0.25]]]
			], dtype=torch.float, device=device)

		self.predx=currentcodec.Codec(use_dropout)
		self.predy=currentcodec.Codec(use_dropout)
		self.predxy=currentcodec.Codec(use_dropout)

	def forward(self, av, diffx, diffy, diffxy):
		x=nn.functional.pad(av, (0, 1, 0, 0))
		x=nn.functional.conv2d(x, self.wx, groups=x.shape[1])
		pred_diffx=self.predx.predict(x, diffx)

		x=nn.functional.pad(av, (0, 0, 0, 1))
		x=nn.functional.conv2d(x, self.wy, groups=x.shape[1])
		#x=torch.cat((x, diffx), dim=1)
		pred_diffy=self.predy.predict(x, diffy)

		x=nn.functional.pad(av, (0, 1, 0, 1))
		x=nn.functional.conv2d(x, self.wxy, groups=x.shape[1])
		#x=torch.cat((x, diffy), dim=1)
		pred_diffxy=self.predxy.predict(x, diffxy)
	
		pred_diffx =torch.split(pred_diffx, 3, dim=1)
		pred_diffy =torch.split(pred_diffy, 3, dim=1)
		pred_diffxy=torch.split(pred_diffxy, 3, dim=1)
		pred=torch.cat((pred_diffx[0], pred_diffy[0], pred_diffxy[0]), dim=1)
		conf=torch.cat((pred_diffx[1], pred_diffy[1], pred_diffxy[1]), dim=1)
		return pred, conf

		#pred0=torch.cat((pred_diffx[0], pred_diffy[0], pred_diffxy[0]), dim=1)
		#pred1=torch.cat((pred_diffx[1], pred_diffy[1], pred_diffxy[1]), dim=1)
		#pred2=torch.cat((pred_diffx[2], pred_diffy[2], pred_diffxy[2]), dim=1)
		#pred3=torch.cat((pred_diffx[3], pred_diffy[3], pred_diffxy[3]), dim=1)
		#conf0=torch.cat((pred_diffx[4], pred_diffy[4], pred_diffxy[4]), dim=1)
		#conf1=torch.cat((pred_diffx[5], pred_diffy[5], pred_diffxy[5]), dim=1)
		#conf2=torch.cat((pred_diffx[6], pred_diffy[6], pred_diffxy[6]), dim=1)
		#conf3=torch.cat((pred_diffx[7], pred_diffy[7], pred_diffxy[7]), dim=1)
		#return pred0, pred1, pred2, pred3, conf0, conf1, conf2, conf3


	#	#av -> diffxy -> diffx -> diffy
	#	x=av
	#	pred_diffxy=self.pred1.predict(x)
	#
	#	x=torch.cat((x, diffxy), dim=1)
	#	pred_diffx=self.pred2.predict(x)
	#
	#	x=torch.cat((x, diffx), dim=1)
	#	pred_diffy=self.pred3.predict(x)
	#
	#	pred_diffx =torch.split(pred_diffx, 3, dim=1)
	#	pred_diffy =torch.split(pred_diffy, 3, dim=1)
	#	pred_diffxy=torch.split(pred_diffxy, 3, dim=1)
	#	pred=torch.cat((pred_diffx[0], pred_diffy[0], pred_diffxy[0]), dim=1)
	#	conf=torch.cat((pred_diffx[1], pred_diffy[1], pred_diffxy[1]), dim=1)
	#	return pred, conf


	#	#av -> diffx -> diffy -> diffxy
	#	x=nn.functional.pad(av, (0, 1, 0, 0))
	#	x=nn.functional.conv2d(x, self.wx, groups=x.shape[1])
	#	pred_diffx=self.predx.predict(x)
	#
	#	x=nn.functional.pad(av, (0, 0, 0, 1))
	#	x=nn.functional.conv2d(x, self.wy, groups=x.shape[1])
	#	#x=torch.cat((x, diffx), dim=1)
	#	pred_diffy=self.predy.predict(x)
	#
	#	x=nn.functional.pad(av, (0, 1, 0, 1))
	#	x=nn.functional.conv2d(x, self.wxy, groups=x.shape[1])
	#	#x=torch.cat((x, diffy), dim=1)
	#	pred_diffxy=self.predxy.predict(x)
	#
	#	pred_diffx =torch.split(pred_diffx, 3, dim=1)
	#	pred_diffy =torch.split(pred_diffy, 3, dim=1)
	#	pred_diffxy=torch.split(pred_diffxy, 3, dim=1)
	#	pred=torch.cat((pred_diffx[0], pred_diffy[0], pred_diffxy[0]), dim=1)
	#	conf=torch.cat((pred_diffx[1], pred_diffy[1], pred_diffxy[1]), dim=1)
	#	return pred, conf


		#return self.codec.predict(x)

		#pred, sdev, truth=self.codec.predict(x)
		#sdev=torch.abs(sdev)+0.0001
		#return pred, sdev, truth

		#csize, pred, sdev, truth=self.codec.predict(x)
		#sdev=torch.abs(sdev)+0.0001
		#return csize, pred, sdev, truth

	#	global g_rate
	#	nlevels=1<<g_rate
	#	invamp=1/(nlevels-1)
	#
	#	y=self.codec.encode(x)
	#
	#	if type(y) is list:
	#		yhat=[]
	#		for idx in range(len(y)):
	#			y[idx]=torch.mul(y[idx], nlevels-1)#variable amplitude
	#			y[idx]=torch.add(y[idx], self.qnoise.sample(y[idx].shape).to(device))	# add quantization noise and clamp [0, 1]
	#			y[idx]=torch.clamp(y[idx], min=0, max=nlevels-1)
	#			yhat.append(y[idx])
	#			y[idx]=torch.mul(y[idx], invamp)
	#	else:
	#		yhat=torch.mul(y, nlevels-1)
	#		yhat=torch.add(yhat, self.qnoise.sample(yhat.shape).to(device))
	#		yhat=torch.clamp(yhat, min=0, max=nlevels-1)
	#		y=torch.mul(yhat, invamp)
	#
	#	#cbits=self.codec.predict(yhat)
	#
	#	xhat=self.codec.decode(y)
	#
	#	#return xhat, cbits
	#	return xhat, yhat

	#def test(self, x):
	#	global g_rate
	#	nlevels=1<<g_rate
	#	invamp=1/(nlevels-1)
	#
	#	y=self.codec.encode(x)
	#
	#	if type(y) is list:
	#		yhat=[]
	#		for idx in range(len(y)):
	#			y[idx]=torch.mul(y[idx], nlevels-1)#variable amplitude
	#			y[idx]=torch.round(y[idx])
	#			y[idx]=torch.clamp(y[idx], min=0, max=nlevels-1)
	#			yhat.append(y[idx])
	#			y[idx]=torch.mul(y[idx], invamp)
	#	else:
	#		yhat=torch.mul(y, nlevels-1)
	#		yhat=torch.torch.round(yhat)
	#		yhat=torch.clamp(yhat, min=0, max=nlevels-1)
	#		y=torch.mul(yhat, invamp)
	#
	#	#cbits=self.codec.predict(yhat)
	#
	#	xhat=self.codec.decode(y)
	#
	#	return xhat, yhat#, cbits*0.125


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
								file.write('\t'+str(val.item()))
							file.write('\n')
						file.write('\n')
					file.write('\n')
				file.write('\n')
			elif dim==3:
				for matrix in param:
					for row in matrix:
						for val in row:
							file.write('\t'+str(val.item()))
						file.write('\n')
					file.write('\n')
				file.write('\n')
			elif dim==2:
				for row in param:
					for val in row:
						file.write('\t'+str(val.item()))
					file.write('\n')
				file.write('\n')
			elif dim==1:
				for val in param:
					file.write('\t'+str(val.item()))
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


model=CompressorModel()
if pretrained:
	load_model(model, modelname+'.pth.tar')
if use_cuda:
	model=model.cuda()
#if torch.__version__ >= version.parse('2.0.0'):#doesn't work on Windows yet
#	model=torch.compile(model, mode='reduce-overhead')

if justexportweights:
	filename=modelname+'-'+time.strftime('%Y%m%d-%H%M%S')+'.txt'
	print('Exporting weights as '+filename)
	maxdim=exportweights(filename, model)
	print('Maxdim: %d'%maxdim)
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

dataset_train=GenericDataLoader(path_train, False)

#if use_dataset=='flickr':
#	dataset_train=GenericDataLoader('E:/ML/dataset-flicker_2W', False)
#elif use_dataset=='clic':
#	dataset_train=GenericDataLoader('E:/ML/dataset-CLIC-small', False)
#elif use_dataset=='awm':
#	#dataset_train=GenericDataLoader('E:/ML/dataset-AWM', False)
#	dataset_train=GenericDataLoader('E:/ML/dataset-AWM-small', False)
#elif use_dataset=='kodak':
#	dataset_train=GenericDataLoader('E:/ML/dataset-kodak-small', False)
#elif use_dataset=='caltech':
#	dataset_train=GenericDataLoader('E:/ML/dataset-caltech256', False)
##elif use_dataset=='imagenet':
##	dataset_train=torchvision.datasets.ImageNet('E:/ML/dataset-imagenet')

if path_val is None:
	dataset_val=None
else:
	dataset_val=GenericDataLoader(path_val, True)

dataset_test=GenericDataLoader(path_test, True)
#dataset_test=GenericDataLoader('E:/ML/dataset-kodak-small', True)

train_size=dataset_train.__len__()
test_size=dataset_test.__len__()
train_loader=DataLoader(dataset_train, batch_size=batch_size, shuffle=True)#try num_workers=16
test_loader=DataLoader(dataset_test, batch_size=1)#different resolutions at 1:1 can't be stacked

if use_SGD:
	optimizer=optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)
else:
	optimizer=optim.Adam(params=model.parameters(), lr=lr, eps=0.0001, weight_decay=weight_decay)#https://discuss.pytorch.org/t/nan-after-50-epochs/75835/4
if use_cuda:
	scaler=torch.cuda.amp.GradScaler()
loss_func=nn.MSELoss()




def color_space_transform(x):#3 channels [0, 1] -> 12 channels [-1, 1]
	#RYZ
	#color_weight=torch.tensor([[[[2]], [[0]], [[0]]],  [[[-1]], [[1]], [[0]]],  [[[-1]], [[0]], [[1]]]], dtype=torch.float, device=x.device)#RYZ
	#color_bias=torch.tensor([-1, 0, 0], dtype=torch.float, device=x.device)

	#XGZ
	color_weight=torch.tensor([[[[1]], [[-1]], [[0]]],  [[[0]], [[2]], [[0]]],  [[[0]], [[-1]], [[1]]]], dtype=torch.float, device=x.device)#XGZ
	color_bias=torch.tensor([0, -1, 0], dtype=torch.float, device=x.device)

	#XYB
	#color_weight=torch.tensor([[[[1]], [[-1]], [[0]]],  [[[1]], [[1]], [[0]]],  [[[0]], [[0]], [[2]]]], dtype=torch.float, device=x.device)#XYB
	#color_bias=torch.tensor([0, -1, -1], dtype=torch.float, device=x.device)

	squeeze_weight=torch.tensor([
		[[[0.25, 0.25], [0.25, 0.25]]],		#average (top-left)
		[[[0.25, -0.25], [0.25, -0.25]]],	#x-derivative (top-right)
		[[[0.25, 0.25], [-0.25, -0.25]]],	#y-derivative (bottom-left)
		[[[0.25, -0.25], [-0.25, 0.25]]],	#2D derivative (bottom-right)

		[[[0.25, 0.25], [0.25, 0.25]]],
		[[[0.25, -0.25], [0.25, -0.25]]],
		[[[0.25, 0.25], [-0.25, -0.25]]],
		[[[0.25, -0.25], [-0.25, 0.25]]],

		[[[0.25, 0.25], [0.25, 0.25]]],
		[[[0.25, -0.25], [0.25, -0.25]]],
		[[[0.25, 0.25], [-0.25, -0.25]]],
		[[[0.25, -0.25], [-0.25, 0.25]]]
	], dtype=torch.float, device=x.device)

	x=nn.functional.conv2d(x, color_weight, color_bias)
	padR=x.shape[3]&1
	padB=x.shape[2]&1
	if padR or padB:
		x=nn.functional.pad(x, (0, padR, 0, padB))
	x=nn.functional.conv2d(x, squeeze_weight, stride=2, groups=x.shape[1])
	return x

def differentiate_xy(x):
	weight=torch.tensor([[
		[[1, -1],
		 [-1, 1]]
	]], dtype=torch.float, device=x.device).repeat(3, 1, 1, 1)#weight is [3, 1, 2, 2]
	#bias=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float, device=x.device)#bias is [3]	X  no bias
	x=nn.functional.pad(x, (1, 0, 1, 0))#(left=1, right=0, top=1, bottom=0)
	x=nn.functional.conv2d(x, weight, groups=x.shape[1])
	return x

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

#def truncated_normal(x, mean, conf):#[0, 1]
#	invsqrt2        = 0.7071067811865475
#	two_over_sqrtpi = 1.1283791670955126/(1<<g_rate)	#two_over_256sqrtpi = 0.004407731121466846
#
#	return two_over_sqrtpi*torch.exp(-0.5*torch.square((x-mean)*conf))*conf*invsqrt2 / (torch.erf((1-mean)*conf*invsqrt2)-torch.erf((-mean)*conf*invsqrt2))

	##f(x) = (2/sqrt(pi)) (exp(-0.5sq((x-mean)*conf))*(conf/sqrt2))  /  ( erf((1-mean)*(conf/sqrt2)) - erf(-mean*(conf/sqrt2)) )
	#x-=mean	#gradient error
	#x*=conf
	#conf*=invsqrt2
	#x=torch.exp(-0.5*torch.square(x))*conf
	#mean*=-conf
	#den=torch.erf(conf+mean)-torch.erf(mean)
	##vmin=torch.min(torch.abs(den))
	##if vmin==0:
	##	idx=torch.argmin(den)
	##	print('min den %f '%vmin, end='')
	##	print(idx)
	#x/=den
	#x*=two_over_256sqrtpi	#multiply f(x) by rectangle width = 1/256
	#return x

def calf_phi3(sym, mean0, mean1, mean2, mean3, conf0, conf1, conf2, conf3):
	ret=torch.erf((sym-mean0)*conf0)+torch.erf((sym-mean1)*conf1)+torch.erf((sym-mean2)*conf2)+torch.erf((sym-mean3)*conf3)
	return ret

def calc_rate3(x, mean0, mean1, mean2, mean3, lgconf0, lgconf1, lgconf2, lgconf3):
	m0=torch.clamp(mean0, -1, 1)
	m1=torch.clamp(mean1, -1, 1)
	m2=torch.clamp(mean2, -1, 1)
	m3=torch.clamp(mean3, -1, 1)
	c0=torch.pow(2, torch.clamp(torch.abs(lgconf0), min=None, max=9))
	c1=torch.pow(2, torch.clamp(torch.abs(lgconf1), min=None, max=9))
	c2=torch.pow(2, torch.clamp(torch.abs(lgconf2), min=None, max=9))
	c3=torch.pow(2, torch.clamp(torch.abs(lgconf3), min=None, max=9))
	p=(calf_phi3(x+1/256, m0, m1, m2, m3, c0, c1, c2, c3)-calf_phi3(x, m0, m1, m2, m3, c0, c1, c2, c3)+1/(256*4096))/(calf_phi3(1, m0, m1, m2, m3, c0, c1, c2, c3)-calf_phi3(-1, m0, m1, m2, m3, c0, c1, c2, c3)+1/(256*4096)+2/4096)
	#p=(torch.erf((x+1/256-mean)*conf)-torch.erf((x-mean)*conf)+1/(256*4096))/(torch.erf((1-mean)*conf)-torch.erf((-1-mean)*conf)+2/4096)
	bitsize=-torch.log2(p)
	return bitsize


#def calc_phi(sym, mean, conf):
#	x=(sym-mean)*conf
#	ret=torch.erf(x)
#	#ret=torch.erf(x)+sym*(1/4096)
#	#ret=torch.sgn(x)*torch.sqrt(torch.abs(x))#X
#	return ret

def calc_rate2(x, mean, lgconf):
	mean=torch.clamp(mean, -1, 1)
	lgconf=torch.clamp(torch.abs(lgconf), min=None, max=9)
	conf=torch.pow(2, lgconf)
	p=(torch.erf((x+1/256-mean)*conf)-torch.erf((x-mean)*conf)+1/(256*4096))/(torch.erf((1-mean)*conf)-torch.erf((-1-mean)*conf)+2/4096)
	#p=(calc_phi(x+1/256, mean, conf)-calc_phi(x, mean, conf)+1/(256*4096))/(calc_phi(1, mean, conf)-calc_phi(-1, mean, conf)+2/4096)
	bitsize=-torch.log2(p)
	return bitsize


#	mean=torch.clamp(mean, -1, 1)
#
#	#conf=torch.clamp(torch.abs(conf), min=None, max=512)
#	lgconf=torch.clamp(torch.abs(lgconf), min=None, max=9)
#
#	#dmin=torch.abs(x-mean).detach()*0.4365
#	#dmin=torch.max(torch.abs(x-mean))
#	#dmin=torch.clamp(dmin, min=0.0001, max=None)
#	#sdev=torch.clamp(torch.abs(sdev), min=dmin, max=None)
#	#sdev=torch.abs(sdev)+1/256
#
#	#m=conf
#	m=torch.pow(2, lgconf)
#	#m=1/sdev
#	c=-mean*m
#
#	num=torch.erfc(c-m)-torch.erfc(c+m)
#	den=torch.erfc(m*x+c)-torch.erfc(m*(x+1/256)+c)
#	#num=torch.erf(c+m)-torch.erf(c-m)
#	#den=torch.erf(m*(x+(1/256))+c)-torch.erf(m*x+c)
#	#den=torch.clamp(den, min=0.0001, max=None)
#	bitsize=torch.log2(num/den)

	#bitsize=torch.nan_to_num(bitsize, 1000, 1000, 1000)#gradient is zero

	#num_finite=torch.sum(torch.isfinite(bitsize)).item()
	#if num_finite!=bitsize.nelement():
	#	print('\nnum_finite %d, nelement %d'%(num_finite, bitsize.nelement()))
	#	assert(0)

	#bitsize=torch.log2(torch.erf(c+m)-torch.erf(c-m))-torch.log2(torch.erf(m*(x+(1/256))+c)-torch.erf(m*x+c))
	
	#p=(torch.erf(m*(x+(1/256))+c)-torch.erf(m*x+c))/(torch.erf(c+m)-torch.erf(c-m))
	#bitsize=-torch.log2(p)
#	return bitsize

def calc_rate(x, mean, lg_conf):#Zipf's law: bitsize = g_rate - log2(f(x))		f(x) = truncated normal distribution
	offset   = 0.5*(math.log2(math.pi)-1)
	gain     = 0.5/math.log(2)
	invsqrt2 = 1/math.sqrt(2)
	conf=torch.pow(2, lg_conf)
	x2=(x-mean)*conf
	c2=conf*invsqrt2
	end  =( 1-mean)*c2
	start=(-1-mean)*c2
	bitsize=g_rate+offset-lg_conf+gain*torch.square(x2)+torch.log2(torch.erfc(start)-torch.erfc(end))#https://math.stackexchange.com/questions/3508801/accurate-computation-of-log-error-function-difference
	#bitsize=g_rate+offset-lg2_conf+gain*torch.square(x2)+torch.log2(torch.erf(end)-torch.erf(start))
	flags=torch.isfinite(bitsize)
	num_finite=int(torch.count_nonzero(flags).item())
	if num_finite!=bitsize.nelement():
		print('\nnum_finite %d, nelement %d'%(num_finite, bitsize.nelement()))
		assert(0)
	#bitsize=torch.nan_to_num(bitsize, nan=100, posinf=100, neginf=100)
	bitsize=torch.clamp(bitsize, 0, 100)#sensible range
	return bitsize

def calc_loss(x):
	x=x.to(device)
	#print('x [%f, %f]'%(torch.min(x).item(), torch.max(x).item()))

	x=color_space_transform(x)
	#print('T [%f, %f]'%(torch.min(x).item(), torch.max(x).item()))

	t=torch.split(x, 1, dim=1)
	av    =torch.cat((t[0], t[4], t[ 8]), dim=1)#top-left
	diffx =torch.cat((t[1], t[5], t[ 9]), dim=1)#top-right
	diffy =torch.cat((t[2], t[6], t[10]), dim=1)#bottom-left
	diffxy=torch.cat((t[3], t[7], t[11]), dim=1)#bottom-right

	#mean0, mean1, mean2, mean3, lgconf0, lgconf1, lgconf2, lgconf3=model(av, diffx, diffy, diffxy)
	mean, lgconf=model(av, diffx, diffy, diffxy)
	truth=torch.cat((diffx, diffy, diffxy), dim=1)

	#size_mask=calc_rate3(truth, mean0, mean1, mean2, mean3, lgconf0, lgconf1, lgconf2, lgconf3)

	size_mask=calc_rate2(truth, mean, lgconf)

	#mean=torch.clamp(mean, -1, 1)
	#lgconf=torch.clamp(torch.abs(lgconf), min=None, max=8)
	#size_mask=calc_rate(truth, mean, lgconf)

	bpp=torch.sum(size_mask)*(1/(truth.nelement()))
	return bpp, size_mask

	#bitsize=torch.sum(size_mask)*(1/(truth.nelement()<<3))
	#ratio=1/bitsize.item()
	#bitsize=torch.sum(size_mask)
	#ratio=truth.nelement()*g_rate/bitsize.item()
	#return bitsize, ratio, mean, lgconf, av, truth, size_mask


#	x=differentiate_xy(x)
#	#x-=0.5	#[-0.5, 0.5]
#	#x*=2	#[-1, 1]
#
#	mean, lgconf, truth=model(x)
#
#	mean=torch.clamp(mean, -1, 1)
#
#	lgconf=torch.clamp(torch.abs(lgconf), min=None, max=8)
#	#lgconf=torch.clamp(lgconf+4, 0, 8)
#	#conf=torch.clamp(conf, 1, 1<<g_rate)
#
#	size_mask=calc_rate(truth, mean, lgconf)
#
#	bitsize=torch.sum(size_mask)
#	ratio=truth.nelement()*g_rate/bitsize.item()
#
#	#mse_rate=loss_func(torch.zeros(bitsize.shape, device=bitsize.device), bitsize)
#	#csize=torch.sum(bitsize).item()/g_rate
#	#bpp=csize*(8/truth.nelement())
#
#	#if bpp==0:
#	#	print('BPP %f'%bpp)
#	return bitsize, ratio, mean, lgconf, truth, size_mask

	#mean, sdev, truth=model(x)
	#prob=torch.div(torch.exp(-0.5*torch.square(torch.div(truth-mean, sdev))), 2.506628274631*sdev)#normal distribution
	#prob*=1/256#rectandle width
	#rate=-torch.log2(prob)#Zipf's law
	#rate=torch.nan_to_num(rate, nan=0, posinf=0, neginf=0)
	#csize=torch.sum(rate)
	#loss=loss_func(torch.zeros(csize.shape, device=csize.device), csize)
	#bpp=csize.item()*(8/truth.nelement())
	#return loss, bpp, mean, sdev, truth

	#csize, pred, sdev, truth=model(x)
	#return loss_func(torch.zeros(csize.shape, device=csize.device), csize), csize.item()*(8/truth.nelement()), pred, sdev, truth
	#return loss_func(torch.zeros(truth.shape, device=truth.device), (truth-pred)/sdev), csize.item()*(8/truth.nelement()), pred, sdev, truth

	#global g_rate
	#xhat, cbits=model(x)
	#E=cbits*(g_rate/(8*x.nelement()))

	#xhat, yhat=model(x)
	#if type(yhat) is list:
	#	E=torch.zeros(1, device=x.device)
	#	for yk in yhat:
	#		if g_rate==1:
	#			E+=calc_pairwiseentropy_differentiable(yk)*yk.nelement()
	#		else:
	#			E+=calc_entropy_differentiable(yk)*yk.nelement()
	#else:
	#	if g_rate==1:
	#		E=calc_pairwiseentropy_differentiable(yhat)*yhat.nelement()
	#	else:
	#		E=calc_entropy_differentiable(yhat)*yhat.nelement()
	#E*=g_rate/(x.nelement()*8)
	#
	#D=loss_func(x, xhat)
	#
	#if g_rate_factor:
	#	L=D+g_rate_factor*E
	#else:
	#	L=D
	#
	#return L, 255*math.sqrt(D.item()), 1/E.item()


#if use_cuda:
#	global_free, total=torch.cuda.mem_get_info(device)
#	print('Using %f/%f MB of GPU memory'%((total-global_free)/(1024*1024), total/(1024*1024)))

#torch.autograd.set_detect_anomaly(True)
start=time.time()
min_loss=-1
nbatches=(train_size+batch_size-1)//batch_size
#inv_nbatches=batch_size//train_size#X
p0=get_params(model)
distance_prev=0
#rmse=0
bpp=0
for epoch in range(epochs):
	it=0
	progress=0
	#rmse=0
	bpp=0
	for x in train_loader:#TRAIN loop
		if use_cuda:
			with torch.cuda.amp.autocast(dtype=torch.float16):#https://pytorch.org/docs/master/notes/amp_examples.html
				L, size_mask=calc_loss(x)		#1 compute the objective function forward
				#L, current_ratio, pred, sdev, av, truth, size_mask=calc_loss(x)
		else:
			L, size_mask=calc_loss(x)

		it+=1
		if not math.isfinite(L.item()):
			print('Loss=%f. SKIPPING BATCH %d.'%(L.item(), it))
			continue

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
		current_bpp=L.item()
		bpp+=current_bpp	# BPP can be averaged, but not CR
		#rmse+=current_rmse
		#ratio+=current_ratio

		print('%5d/%5d = %5.2f%%  CR %16.12f BPP %15.12f size %10d /%10d'%(progress, train_size, 100*progress/train_size, 8/current_bpp, current_bpp, math.ceil(L.item()*size_mask.shape[1]*size_mask.shape[2]*size_mask.shape[3]/8), size_mask.nelement()//x.shape[0]), end='\r')
		#print('%d/%d = %5.2f%%  RMSE %16.12f BPP %14.12f\t\t'%(progress, train_size, 100*progress/train_size, current_rmse, 8/current_ratio), end='\r')
	#rmse/=nbatches
	bpp/=nbatches
	ratio=8/bpp

	if dataset_val is None:
		val_bpp=bpp
	else:
		val_bpp=0
		with torch.no_grad():#validation
			nval=len(dataset_val)
			for k in range(nval):
				x=dataset_val[k]
				x=x[None, :, :, :]
				val_L, size_mask=calc_loss(x)
				val_bpp+=val_L.item()
				print('%5d/%5d = %5.2f%%  CR %16.12f BPP %15.12f size %10d /%10d'%(k+1, nval, 100*(k+1)/nval, 8/val_L.item(), val_L.item(), math.ceil(val_L.item()*size_mask.shape[1]*size_mask.shape[2]*size_mask.shape[3]), size_mask.nelement()//x.shape[0]), end='\r')
			val_bpp/=nval

			#x=dataset_test[21-1]
			#x=x[None, :, :, :]
			#v_L, v_ratio, v_pred, v_sdev, v_av, v_truth, size_mask=calc_loss(x)

	to_save=not save_records
	record=''
	if min_loss<0 or min_loss>val_bpp:
		to_save=1
		if min_loss>val_bpp:
			record=' %10f%%'%(100.*(val_bpp-min_loss)/min_loss)
		min_loss=val_bpp
	if to_save:
		torch.save(model.state_dict(), modelname+'.pth.tar')

	if epoch==0:
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

	print('Epoch %3d [%10f,%10f]  CR %16.12f BPP %13.9f  val CR %10f  elapsed %10f '%(epoch+1, distance_current, distance_delta, ratio, bpp, 8/val_bpp, (t2-start)/60), end='')
	#psnr=20*math.log10(255/rmse)
	#print('Epoch %3d [%10f,%10f]  RMSE %16.12f PSNR %13.9f BPP %12.9f  elapsed %10f '%(epoch+1, distance_current, distance_delta, rmse, psnr, 8/ratio, (t2-start)/60), end='')
	print(str(timedelta(seconds=t2-start))+record)

end=time.time()
print('Train elapsed: '+str(timedelta(seconds=end-start)))

if epochs:
	if save_records:
		load_model(model, modelname+'.pth.tar')
	torch.save(model.state_dict(), '%s-%s-cr%f.pth.tar'%(modelname, time.strftime('%Y%m%d-%H%M%S'), 8/bpp))
	#torch.save(model.state_dict(), '%s-%s-psnr%f-bpp%f.pth.tar'%(modelname, time.strftime('%Y%m%d-%H%M%S'), 20*math.log10(255/rmse), 8/ratio))

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
test_idx=0
#rmse=0
bpp=0
t_filt=0
for x in test_loader:#TEST loop
	with torch.no_grad():
		t1=time.time()
		current_bpp, size_mask=calc_loss(x)
		t2=time.time()

		current_ratio=8/current_bpp.item()

		#minconf=torch.min(conf).item()
		#maxconf=torch.max(conf).item()
		#conf*=0.125	#[0, 8] -> [0, 1]
		vs_jxl=current_ratio/cr_kodak_jxl[test_idx]
		vs_png=current_ratio/cr_kodak_png[test_idx]
		#fmt_bytesize=dataset_test.get_size(test_idx)
		#fmt_ratio=x.nelement()/fmt_bytesize
		#current_bpp=8/current_ratio
		print('Test %2d  CR %16.12f BPP %13.9f  vs.jxl %8.6f vs.png %8.6f  elapsed %9f sec'%(test_idx+1, current_ratio, current_bpp, vs_jxl, vs_png, t2-t1))
		#print('Test %2d  CR %16.12f BPP %13.9f  vs.jxl %8.6f vs.png %8.6f  conf [%f, %f]  elapsed %9f sec'%(test_idx+1, current_ratio, current_bpp, vs_jxl, vs_png, minconf, maxconf, t2-t1))
		#current_psnr=20*math.log10(255/current_rmse)
		#print('Test %2d  rmse %16.12f psnr %13.9f  lossy %7d BPP %16.12f  lossless %7d BPP %16.12f  elapsed %9f'%(test_idx+1, current_rmse, current_psnr, current_csize0, current_csize0*8/current_usize, current_csize1, current_csize1*8/current_usize, t2-t1))

		if test_idx+1==21:#save preview
			#diffx, diffy, diffxy=torch.split(truth-mean+0.5, 3, dim=1)
			#s1=torch.cat((torch.cat((av, diffx), dim=3), torch.cat((diffy, diffxy), dim=3)), dim=2)#difference between original & prediction
			size_mask*=10/255
			#vmin=torch.min(size_mask).item()
			#vmax=torch.max(size_mask).item()
			#if vmin<vmax:
			#	size_mask=(size_mask-vmin)/(vmax-vmin)
			topright, bottomleft, bottomright=torch.split(size_mask, 3, dim=1)
			topleft=torch.zeros(topright.shape, dtype=topright.dtype, device=topright.device)
			s2=torch.cat((torch.cat((topleft, topright), dim=3), torch.cat((bottomleft, bottomright), dim=3)), dim=2)#per-pixel bit-cost
			sample=s2
			#sample=torch.cat((s1, s2), dim=3)
		
			fn='%s-%s-%d'%(modelname, time.strftime('%Y%m%d-%H%M%S'), test_idx+1)
			save_tensor_as_grid(sample, 1, 'results/'+fn+'-cr%f.PNG'%current_ratio)

		#if test_idx+1==21:
		#	sample=torch.cat((truth+0.5, mean+0.5, conf, truth-mean+0.5, size_mask), dim=3)
		#	fn='%s-%s-%d'%(modelname, time.strftime('%Y%m%d-%H%M%S'), test_idx+1)
		#	save_tensor_as_grid(sample, 1, 'results/'+fn+'-bpp%f-conf%f.PNG'%(current_bpp, maxconf))
		bpp+=current_bpp.item()
		#usize+=truth.nelement()#current_usize
		#csize+=truth.nelement()/current_ratio#current_csize
		t_filt+=t2-t1
		test_idx+=1

if test_idx:
	bpp/=test_idx
	ratio=8/bpp
	#ratio=usize/csize
	print('Average  CR %16.12f BPP %13.9f  vs.jxl %8.6f vs.png %8.6f  filt %f sec'%(ratio, 8/ratio, ratio/2.78408057362, ratio/1.8390925735, t_filt/test_idx))
	#rmse/=test_idx
	#print('Average rmse %16.12f psnr %13.9f  size %7d lossy %7d BPP %13.9f  lossless %7d BPP %13.9f  filt %f sec'%(rmse, 20*math.log10(255/rmse), usize/test_idx, csize0/test_idx, csize0*8/usize, csize1/test_idx, csize1*8/usize, t_filt/test_idx))
print('Finished on '+time.strftime('%Y-%m-%d %H:%M:%S'))




