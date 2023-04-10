#2023-04-10Mo
import os
from PIL import Image
import math
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision

import json
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
#C35-W32D08-per-px-erf	 107640		105.390820@CLIC@1	3.156985@54	3.070384@54	2.885509@50
#		range bug fix 2

#name			params		sec/epoch		peak train CR	peak val CR	peak test CR
#C35-W32D08-per-px-erf	 107640		108.658572@CLIC@1	1.724101#75	1.692290@75	1.618866@75
#C37-W32D08-perpx-plain	  30000		114.883141@CLIC@1	3.155952@75	2.994918@74	2.879331@75




## config ##
import codec37 as currentcodec
modelname='C37'
pretrained=1	# assign pretrained=0 when training first time
save_records=0

epochs=25
lr=0.0001		#always start with high learning rate 0.001
batch_size=1		# <=24, increase batch size instead of decreasing learning rate
train_block=0		#256: batch_size=8
cache_rebuild=0		#set to 1 if train_block was changed

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
path_train='E:/ML/dataset-CLIC'		# best at 1:1
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


def color_transform(x):#3 channels [0, 1] -> 12 channels [-1, 1]
	#RYZ
	#color_weight=torch.tensor([[[[2]], [[0]], [[0]]],  [[[-1]], [[1]], [[0]]],  [[[-1]], [[0]], [[1]]]], dtype=torch.float, device=x.device)#RYZ
	#color_bias=torch.tensor([-1, 0, 0], dtype=torch.float, device=x.device)

	#XGZ
	color_weight=torch.tensor([[[[1]], [[-1]], [[0]]],  [[[0]], [[2]], [[0]]],  [[[0]], [[-1]], [[1]]]], dtype=torch.float, device=x.device)#XGZ
	color_bias=torch.tensor([0, -1, 0], dtype=torch.float, device=x.device)

	#XYB
	#color_weight=torch.tensor([[[[1]], [[-1]], [[0]]],  [[[1]], [[1]], [[0]]],  [[[0]], [[0]], [[2]]]], dtype=torch.float, device=x.device)#XYB
	#color_bias=torch.tensor([0, -1, -1], dtype=torch.float, device=x.device)

	#squeeze_weight=torch.tensor([
	#	[[[0.25, 0.25], [0.25, 0.25]]],		#average (top-left)
	#	[[[0.25, -0.25], [0.25, -0.25]]],	#x-derivative (top-right)
	#	[[[0.25, 0.25], [-0.25, -0.25]]],	#y-derivative (bottom-left)
	#	[[[0.25, -0.25], [-0.25, 0.25]]],	#2D derivative (bottom-right)
	#
	#	[[[0.25, 0.25], [0.25, 0.25]]],
	#	[[[0.25, -0.25], [0.25, -0.25]]],
	#	[[[0.25, 0.25], [-0.25, -0.25]]],
	#	[[[0.25, -0.25], [-0.25, 0.25]]],
	#
	#	[[[0.25, 0.25], [0.25, 0.25]]],
	#	[[[0.25, -0.25], [0.25, -0.25]]],
	#	[[[0.25, 0.25], [-0.25, -0.25]]],
	#	[[[0.25, -0.25], [-0.25, 0.25]]]
	#], dtype=torch.float, device=x.device)

	x=nn.functional.conv2d(x, color_weight, color_bias)
	#padR=x.shape[3]&1
	#padB=x.shape[2]&1
	#if padR or padB:
	#	x=nn.functional.pad(x, (0, padR, 0, padB))
	#x=nn.functional.conv2d(x, squeeze_weight, stride=2, groups=x.shape[1])
	#x*=256
	return x

def differentiate_xy(x):
	weight=torch.tensor([
		[[[0.25, -0.25], [-0.25, 0.25]]],
		[[[0.25, -0.25], [-0.25, 0.25]]],
		[[[0.25, -0.25], [-0.25, 0.25]]]
	], dtype=x.dtype, device=x.device)#[3, 1, 2, 2]

	x=nn.functional.pad(x, (0, 1, 0, 1))#(left=0, right=1, top=0, bottom=1)
	x=nn.functional.conv2d(x, weight, groups=x.shape[1])
	return x


def calc_rate2b(x, mean, conf, nlevels):
	p=(torch.erf((x+1/nlevels-mean)*conf)-torch.erf((x-mean)*conf)+1/(nlevels*4096))/(torch.erf((1-mean)*conf)-torch.erf((-1-mean)*conf)+2/4096)
	bitsize=-torch.log2(p)

	#num_finite=num_finite=torch.sum(torch.isfinite(bitsize)).item()
	#if num_finite!=bitsize.nelement():
	#	print('\nnum_finite %d, nelement %d'%(num_finite, bitsize.nelement()))
	#	assert(0)

	return bitsize

def calc_rate2(sym, mean, conf, nlevels):
	symx,  symg,  symz =torch.split(sym, 1, dim=1)
	meanx, meang, meanz=torch.split(mean, 1, dim=1)
	confx, confg, confz=torch.split(conf, 1, dim=1)
	return calc_rate2b(symx, meanx, confx, nlevels<<1)+calc_rate2b(symg, meang, confg, nlevels)+calc_rate2b(symz, meanz, confz, nlevels<<1)

class CompressorModel(nn.Module):
	def __init__(self):
		super(CompressorModel, self).__init__()
		global device

		self.pred=currentcodec.Codec(use_dropout)

		#self.wx=torch.tensor([
		#	[[[0.5, -0.5]]],
		#	[[[0.5, -0.5]]],
		#	[[[0.5, -0.5]]]
		#], dtype=torch.float, device=device)
		#self.wy=torch.tensor([
		#	[[[0.5], [-0.5]]],
		#	[[[0.5], [-0.5]]],
		#	[[[0.5], [-0.5]]]
		#], dtype=torch.float, device=device)
		#self.wxy=torch.tensor([
		#	[[[0.25, -0.25], [-0.25, 0.25]]],
		#	[[[0.25, -0.25], [-0.25, 0.25]]],
		#	[[[0.25, -0.25], [-0.25, 0.25]]]
		#	], dtype=torch.float, device=device)
		#
		#self.predx=currentcodec.Codec(use_dropout)
		#self.predy=currentcodec.Codec(use_dropout)
		#self.predxy=currentcodec.Codec(use_dropout)

	def forward(self, x):
		x=x.to(device)

		x=color_transform(x)
		x=differentiate_xy(x)

		result=self.pred.predict(x)
		mean, lgconf=torch.split(result, 3, dim=1)

		mean=torch.clamp(mean, -1, 1)
		lgconf=torch.clamp(torch.abs(lgconf), min=None, max=11)
		conf=torch.pow(2, lgconf)
		size_mask=calc_rate2(x, mean, conf, 256)

		bpp=torch.sum(size_mask)*(1/x.nelement())
		return bpp, size_mask


	#	x=x.to(device)
	#
	#	x*=2
	#	x-=1
	#
	#	result=self.pred.predict(x)
	#	mean, lgconf=torch.split(result, 3, dim=1)
	#
	#	mean=torch.clamp(mean, -1, 1)
	#	lgconf=torch.clamp(torch.abs(lgconf), min=None, max=11)
	#	conf=torch.pow(2, lgconf)
	#	size_mask=calc_rate2b(x, mean, conf, 256)
	#
	#	bpp=torch.sum(size_mask)*(1/x.nelement())
	#	return bpp, size_mask


	#	x=nn.functional.pad(av, (0, 1, 0, 0))
	#	x=nn.functional.conv2d(x, self.wx, groups=x.shape[1])
	#	#diffx=torch.zeros(diffx.shape, dtype=diffx.dtype, device=diffx.device)#
	#	pred_diffx=self.predx.predict(x, diffx)
	#
	#	x=nn.functional.pad(av, (0, 0, 0, 1))
	#	x=nn.functional.conv2d(x, self.wy, groups=x.shape[1])
	#	#x=torch.cat((x, diffx), dim=1)
	#	#diffy=torch.zeros(diffy.shape, dtype=diffy.dtype, device=diffy.device)#
	#	pred_diffy=self.predy.predict(x, diffy)
	#
	#	x=nn.functional.pad(av, (0, 1, 0, 1))
	#	x=nn.functional.conv2d(x, self.wxy, groups=x.shape[1])
	#	#x=torch.cat((x, diffy), dim=1)
	#	#diffxy=torch.zeros(diffxy.shape, dtype=diffxy.dtype, device=diffxy.device)#
	#	pred_diffxy=self.predxy.predict(x, diffxy)
	#
	#	pred_diffx =torch.split(pred_diffx, 3, dim=1)
	#	pred_diffy =torch.split(pred_diffy, 3, dim=1)
	#	pred_diffxy=torch.split(pred_diffxy, 3, dim=1)
	#	return pred_diffx[0], pred_diffy[0], pred_diffxy[0], pred_diffx[1], pred_diffy[1], pred_diffxy[1]


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


def write4D(file, name, param):
	file.write('%s\t%d\t%d\t%d\t%d:\n'%(name, param.shape[0], param.shape[1], param.shape[2], param.shape[3]))
	for filt in param:
		for kernel in filt:
			for row in kernel:
				for val in row:
					file.write('\t'+val.item().hex())
					#file.write('\t%#X'%int(val.item()*0x100000000))
				file.write('\n')
			file.write('\n')
		file.write('\n')
	file.write('\n')

def write1D(file, name, param):
	file.write('%s\t%d:\n'%(name, param.shape[0]))
	for val in param:
		file.write('\t'+val.item().hex())
		#file.write('\t%#X'%int(val.item()*0x100000000))
	file.write('\n')

def catmul(top, bot, bn):
	x=torch.cat((bot, torch.zeros(bot.shape[0], bot.shape[1], bot.shape[2], top.shape[3]-bot.shape[3], dtype=bot.dtype, device=bot.device)), dim=3)
	x=torch.cat((top, x), dim=2)
	x=torch.split(x, 1, dim=0)
	b=torch.split(bn, 1, dim=0)
	y=[]
	for k in range(len(x)):
		y.append(x[k]*b[k])
	x=torch.cat(y, dim=0)
	return x

def addmuladd(a, b, m, c):
	return (a+b)*m+c

def exportC35(filename, model: CompressorModel):
	with torch.no_grad():
		with open(filename, 'w') as file:
			#write4D(file, 'predx.conv01.weight', model.predx.conv01.weight)
			#write1D(file, 'predx.conv01.bias',   model.predx.conv01.bias)
			#write4D(file, 'predx.conv02.weight', model.predx.conv02.weight)
			#write1D(file, 'predx.conv02.bias',   model.predx.conv02.bias)

			write4D(file, 'pred.c01.weight',  catmul(model.pred.ct01.weight, model.pred.cl01.weight, model.pred.b01.weight))
			write1D(file, 'pred.c01.bias', addmuladd(model.pred.ct01.bias,   model.pred.cl01.bias,   model.pred.b01.weight, model.pred.b01.bias))
			write4D(file, 'pred.c02.weight',  catmul(model.pred.ct02.weight, model.pred.cl02.weight, model.pred.b02.weight))
			write1D(file, 'pred.c02.bias', addmuladd(model.pred.ct02.bias,   model.pred.cl02.bias,   model.pred.b02.weight, model.pred.b02.bias))
			write4D(file, 'pred.c03.weight',  catmul(model.pred.ct03.weight, model.pred.cl03.weight, model.pred.b03.weight))
			write1D(file, 'pred.c03.bias', addmuladd(model.pred.ct03.bias,   model.pred.cl03.bias,   model.pred.b03.weight, model.pred.b03.bias))
			write4D(file, 'pred.c04.weight',  catmul(model.pred.ct04.weight, model.pred.cl04.weight, model.pred.b04.weight))
			write1D(file, 'pred.c04.bias', addmuladd(model.pred.ct04.bias,   model.pred.cl04.bias,   model.pred.b04.weight, model.pred.b04.bias))
			write4D(file, 'pred.c05.weight',  catmul(model.pred.ct05.weight, model.pred.cl05.weight, model.pred.b05.weight))
			write1D(file, 'pred.c05.bias', addmuladd(model.pred.ct05.bias,   model.pred.cl05.bias,   model.pred.b05.weight, model.pred.b05.bias))
			write4D(file, 'pred.c06.weight',  catmul(model.pred.ct06.weight, model.pred.cl06.weight, model.pred.b06.weight))
			write1D(file, 'pred.c06.bias', addmuladd(model.pred.ct06.bias,   model.pred.cl06.bias,   model.pred.b06.weight, model.pred.b06.bias))
			write4D(file, 'pred.c07.weight',  catmul(model.pred.ct07.weight, model.pred.cl07.weight, model.pred.b07.weight))
			write1D(file, 'pred.c07.bias', addmuladd(model.pred.ct07.bias,   model.pred.cl07.bias,   model.pred.b07.weight, model.pred.b07.bias))
			write4D(file, 'pred.c08.weight',  catmul(model.pred.ct08.weight, model.pred.cl08.weight, model.pred.b08.weight))
			write1D(file, 'pred.c08.bias', addmuladd(model.pred.ct08.bias,   model.pred.cl08.bias,   model.pred.b08.weight, model.pred.b08.bias))



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
	exportC35(filename, model)
	#exportweights(filename, model)
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

dataset_train=GenericDataLoader(path_train, False)

if path_val is None:
	dataset_val=None
else:
	dataset_val=GenericDataLoader(path_val, True)

dataset_test=GenericDataLoader(path_test, True)

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




#torch.autograd.set_detect_anomaly(True)
start=time.time()
min_loss=-1
nbatches=(train_size+batch_size-1)//batch_size
#inv_nbatches=batch_size//train_size#X
p0=get_params(model)
distance_prev=0
bpp=0
for epoch in range(epochs):
	it=0
	progress=0
	bpp=0
	for x in train_loader:#TRAIN loop
		if use_cuda:
			with torch.cuda.amp.autocast(dtype=torch.float16):#https://pytorch.org/docs/master/notes/amp_examples.html
				L, size_mask=model(x)		#1 compute the objective function forward
		else:
			L, size_mask=model(x)

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

		ubytes=x.shape[1]*x.shape[2]*x.shape[3]
		print('%5d/%5d =%6.2f%%  CR %16.12f BPP %15.12f size %10d /%10d'%(progress, train_size, 100*progress/train_size, 8/current_bpp, current_bpp, math.ceil(L.item()*ubytes/8), ubytes), end='\r')
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
				val_L, size_mask=model(x)
				val_bpp+=val_L.item()
				ubytes=x.shape[1]*x.shape[2]*x.shape[3]
				print('%5d/%5d =%6.2f%%  CR %16.12f BPP %15.12f size %10d /%10d'%(k+1, nval, 100*(k+1)/nval, 8/val_L.item(), val_L.item(), math.ceil(val_L.item()*ubytes/8), ubytes), end='\r')
			val_bpp/=nval

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
	print(str(timedelta(seconds=t2-start))+record)

end=time.time()
print('Train elapsed: '+str(timedelta(seconds=end-start)))

if epochs:
	if save_records:
		load_model(model, modelname+'.pth.tar')
	torch.save(model.state_dict(), '%s-%s-cr%f.pth.tar'%(modelname, time.strftime('%Y%m%d-%H%M%S'), 8/bpp))

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
bpp=0
t_filt=0
for x in test_loader:#TEST loop
	with torch.no_grad():
		t1=time.time()
		current_bpp, size_mask=model(x)
		t2=time.time()

		current_ratio=8/current_bpp.item()

		vs_jxl=current_ratio/cr_kodak_jxl[test_idx]
		vs_png=current_ratio/cr_kodak_png[test_idx]
		print('Test %2d  CR %16.12f BPP %13.9f  vs.jxl %8.6f vs.png %8.6f  elapsed %9f sec'%(test_idx+1, current_ratio, current_bpp, vs_jxl, vs_png, t2-t1))

		if test_idx+1==21:#save preview
			size_mask*=10/255
			sample=size_mask
		
			fn='%s-%s-%d'%(modelname, time.strftime('%Y%m%d-%H%M%S'), test_idx+1)
			save_tensor_as_grid(sample, 1, 'results/'+fn+'-cr%f.PNG'%current_ratio)

		bpp+=current_bpp.item()
		t_filt+=t2-t1
		test_idx+=1

if test_idx:
	bpp/=test_idx
	ratio=8/bpp
	print('Average  CR %16.12f BPP %13.9f  vs.jxl %8.6f vs.png %8.6f  filt %f sec'%(ratio, 8/ratio, ratio/2.78408057362, ratio/1.8390925735, t_filt/test_idx))
print('Finished on '+time.strftime('%Y-%m-%d %H:%M:%S'))




