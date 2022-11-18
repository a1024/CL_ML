from ctypes import sizeof
import torch
from torchvision import models
from torchsummary import summary

from pprint import pprint
import struct


resnet==models.resnet18(pretrained=True)
#resnet=torch.load('E:/ML/resnet18-f37072fd.pth')
summary(resnet, (3, 224, 224))

def save_values4(name):
	c1=resnet[name]
	#print(c1.size())

	nfilters=c1.size(0)
	nch=c1.size(1)
	h=c1.size(2)
	w=c1.size(3)
	outfile=open(name+'.txt', 'w')
	outfile.write(str(nfilters)+' '+str(nch)+' '+str(h)+' '+str(w)+'\n')
	for k in range(nfilters):
		outfile.write('filt'+str(k)+'\n')
		filt=c1.data[k]
		for kc in range(nch):
			channel=filt.data[kc]
			outfile.write('chan'+str(kc)+'\n')
			for ky in range(h):
				row=channel.data[ky]
				for kx in range(w):
					b=struct.pack('>f', row.data[kx])
					outfile.write(b.hex())
				outfile.write('\n')
			outfile.write('\n')
		outfile.write('\n')
	outfile.close()
	
def save_values1(name):
	c1=resnet[name]
	#print(c1.size())

	outfile=open(name+'.txt', 'w')
	nvals=c1.size(0)
	for ky in range(nvals):
		b=struct.pack('>f', c1.data[ky])
		outfile.write(b.hex()+'\n')
	outfile.close()

def save_values_fc(name):
	c1=resnet[name]
	#print(c1.size())

	outfile=open(name+'.txt', 'w')
	noutputs=c1.size(0)
	ninputs=c1.size(1)
	outfile.write(str(noutputs)+' '+str(ninputs)+'\n')
	for ky in range(noutputs):
		node=c1.data[ky]
		for kx in range(ninputs):
			b=struct.pack('>f', node.data[kx])
			outfile.write(b.hex())
		outfile.write('\n')
	outfile.close()


save_values4('conv1.weight')
save_values1('bn1.running_mean')
save_values1('bn1.running_var')
save_values1('bn1.weight')
save_values1('bn1.bias')
save_values4('layer1.0.conv1.weight')
save_values1('layer1.0.bn1.running_mean')
save_values1('layer1.0.bn1.running_var')
save_values1('layer1.0.bn1.weight')
save_values1('layer1.0.bn1.bias')
save_values4('layer1.0.conv2.weight')
save_values1('layer1.0.bn2.running_mean')
save_values1('layer1.0.bn2.running_var')
save_values1('layer1.0.bn2.weight')
save_values1('layer1.0.bn2.bias')
save_values4('layer1.1.conv1.weight')
save_values1('layer1.1.bn1.running_mean')
save_values1('layer1.1.bn1.running_var')
save_values1('layer1.1.bn1.weight')
save_values1('layer1.1.bn1.bias')
save_values4('layer1.1.conv2.weight')
save_values1('layer1.1.bn2.running_mean')
save_values1('layer1.1.bn2.running_var')
save_values1('layer1.1.bn2.weight')
save_values1('layer1.1.bn2.bias')
save_values4('layer2.0.conv1.weight')
save_values1('layer2.0.bn1.running_mean')
save_values1('layer2.0.bn1.running_var')
save_values1('layer2.0.bn1.weight')
save_values1('layer2.0.bn1.bias')
save_values4('layer2.0.conv2.weight')
save_values1('layer2.0.bn2.running_mean')
save_values1('layer2.0.bn2.running_var')
save_values1('layer2.0.bn2.weight')
save_values1('layer2.0.bn2.bias')
save_values4('layer2.0.downsample.0.weight')
save_values1('layer2.0.downsample.1.running_mean')
save_values1('layer2.0.downsample.1.running_var')
save_values1('layer2.0.downsample.1.weight')
save_values1('layer2.0.downsample.1.bias')
save_values4('layer2.1.conv1.weight')
save_values1('layer2.1.bn1.running_mean')
save_values1('layer2.1.bn1.running_var')
save_values1('layer2.1.bn1.weight')
save_values1('layer2.1.bn1.bias')
save_values4('layer2.1.conv2.weight')
save_values1('layer2.1.bn2.running_mean')
save_values1('layer2.1.bn2.running_var')
save_values1('layer2.1.bn2.weight')
save_values1('layer2.1.bn2.bias')
save_values4('layer3.0.conv1.weight')
save_values1('layer3.0.bn1.running_mean')
save_values1('layer3.0.bn1.running_var')
save_values1('layer3.0.bn1.weight')
save_values1('layer3.0.bn1.bias')
save_values4('layer3.0.conv2.weight')
save_values1('layer3.0.bn2.running_mean')
save_values1('layer3.0.bn2.running_var')
save_values1('layer3.0.bn2.weight')
save_values1('layer3.0.bn2.bias')
save_values4('layer3.0.downsample.0.weight')
save_values1('layer3.0.downsample.1.running_mean')
save_values1('layer3.0.downsample.1.running_var')
save_values1('layer3.0.downsample.1.weight')
save_values1('layer3.0.downsample.1.bias')
save_values4('layer3.1.conv1.weight')
save_values1('layer3.1.bn1.running_mean')
save_values1('layer3.1.bn1.running_var')
save_values1('layer3.1.bn1.weight')
save_values1('layer3.1.bn1.bias')
save_values4('layer3.1.conv2.weight')
save_values1('layer3.1.bn2.running_mean')
save_values1('layer3.1.bn2.running_var')
save_values1('layer3.1.bn2.weight')
save_values1('layer3.1.bn2.bias')
save_values4('layer4.0.conv1.weight')
save_values1('layer4.0.bn1.running_mean')
save_values1('layer4.0.bn1.running_var')
save_values1('layer4.0.bn1.weight')
save_values1('layer4.0.bn1.bias')
save_values4('layer4.0.conv2.weight')
save_values1('layer4.0.bn2.running_mean')
save_values1('layer4.0.bn2.running_var')
save_values1('layer4.0.bn2.weight')
save_values1('layer4.0.bn2.bias')
save_values4('layer4.0.downsample.0.weight')
save_values1('layer4.0.downsample.1.running_mean')
save_values1('layer4.0.downsample.1.running_var')
save_values1('layer4.0.downsample.1.weight')
save_values1('layer4.0.downsample.1.bias')
save_values4('layer4.1.conv1.weight')
save_values1('layer4.1.bn1.running_mean')
save_values1('layer4.1.bn1.running_var')
save_values1('layer4.1.bn1.weight')
save_values1('layer4.1.bn1.bias')
save_values4('layer4.1.conv2.weight')
save_values1('layer4.1.bn2.running_mean')
save_values1('layer4.1.bn2.running_var')
save_values1('layer4.1.bn2.weight')
save_values1('layer4.1.bn2.bias')
save_values_fc('fc.weight')
save_values1('fc.bias')
