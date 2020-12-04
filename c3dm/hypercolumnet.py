# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as Fu
import torchvision
from torchvision import models
from visdom import Visdom
import numpy as np
from tools.utils import auto_init_args

import torchvision

import collections

class HyperColumNet(nn.Module):
	def __init__( self, 
				  trunk_arch='resnet50', 
				  n_upsample=2, 
				  hc_layers=[1,2,3,4], 
				  hcdim=512, 
				  pose_confidence=True,
				  depth_offset=0., 
				  smooth=False, 
				  encode_input_keypoints = False,
				  kp_encoding_sig=1., 
				  dimout=1,
				  dimout_glob = 0,
				  dimout_glob_alpha = 0,
				  n_keypoints=12, 
				  architecture='hypercolumns',
				  dilate_start=2, 
				  glob_inst_norm=False,
				  final_std=0.01,
				  final_bias=-1.,
				  glob_activation=True,
				  pretrained=True ):
		super().__init__()

		auto_init_args(self)

		trunk = getattr(torchvision.models,trunk_arch)(pretrained=pretrained)
		# nfc = trunk.fc.in_features

		self.layer0 = torch.nn.Sequential( trunk.conv1,
										   trunk.bn1,
										   trunk.relu,
										   trunk.maxpool )

		if self.architecture=='hypercolumns':

			for l in [1, 2, 3, 4]:
				lname = 'layer%d'%l
				setattr(self, lname, getattr(trunk,lname))

			for hcl in hc_layers:
				lname = 'hc_layer%d'%hcl
				indim = getattr(trunk,'layer%d'%hcl)[-1].conv1.in_channels
				
				# if ((self.dimout_glob + self.dimout_glob_alpha) > 0 \
				# 	and hcl==hc_layers[-1]):
				# 	if not self.smooth:
				# 		glob_layers = [ torch.nn.Conv2d(indim, indim,1,bias=True,padding=0),
				# 						torch.nn.ReLU(),
				# 						nn.Conv2d(indim, self.dimout_glob+self.dimout_glob_alpha, \
				# 						1, bias=True, padding=0) ]
				# 		if self.glob_activation:
				# 			glob_layers.insert(1, \
				# 				torch.nn.InstanceNorm2d(indim) if self.glob_inst_norm \
				# 					else torch.nn.BatchNorm2d(indim))
				# 	else:
				# 		glob_layers  = [ nn.Conv2d(indim, self.dimout_glob+self.dimout_glob_alpha, \
				# 						 1, bias=True, padding=0) ]
				# 	self.final_glob = torch.nn.Sequential(*glob_layers )

				if self.encode_input_keypoints:
					indim += self.n_keypoints
				
				if not self.smooth:
					layer_ = torch.nn.Sequential( \
								torch.nn.Conv2d(indim, hcdim, 3, bias=True, padding=1),
								torch.nn.BatchNorm2d(hcdim),
								torch.nn.ReLU(),
								torch.nn.Conv2d(hcdim, hcdim, 3, bias=True, padding=1),
								)
				else:
					layer_ = torch.nn.Sequential( \
								torch.nn.Conv2d(indim, hcdim, 3, bias=True, padding=1),
								)
				setattr(self, lname, layer_)

			if not self.smooth:
				up_layers = [ torch.nn.Conv2d(hcdim,hcdim,3,bias=True,padding=1),
							torch.nn.BatchNorm2d(hcdim),
							torch.nn.ReLU(),
							nn.Conv2d(hcdim, dimout, 3, bias=True, padding=1) ]
			else:
				up_layers = [ nn.Conv2d(hcdim, dimout, 3, bias=True, padding=1) ]

			llayer = up_layers[-1]
			llayer.weight.data = \
				llayer.weight.data.normal_(0., self.final_std)
			if self.final_bias > -1.:
				llayer.bias.data = \
					llayer.bias.data.fill_(self.final_bias)
			print('hcnet: final bias = %1.2e, final std=%1.2e' % \
						(llayer.bias.data.mean(),
						 llayer.weight.data.std())
						)
			self.final = torch.nn.Sequential(*up_layers)
		

		elif self.architecture=='dilated':

			if self.dimout_glob > 0:
				raise NotImplementedError('not done yet')

			# for l in [1, 2, 3, 4]:
			# 	lname = 'layer%d'%l
			# 	setattr(self, lname, getattr(trunk,lname))

			if self.encode_input_keypoints:
				c1 = self.layer0[0]
				wsz = list(c1.weight.data.shape)
				wsz[1] = self.n_keypoints
				c1_add = c1.weight.data.new_zeros( wsz ).normal_(0.,0.0001)
				c1.weight.data = torch.cat( (c1.weight.data, c1_add), dim=1 )
				c1.in_channels += self.n_keypoints

			layers = [self.layer0]

			li = 0
			for l in [1,2,3,4]:
				lname = 'layer%d'%l
				m = getattr(trunk,lname)
				if l >= self.dilate_start:
					for mm in m.modules():
						if type(mm) == torch.nn.Conv2d:
							mm.stride = (1,1)
							if mm.kernel_size==(3,3):
								dil = (li+2)**2
								mm.dilation = ( dil, dil )
								mm.padding  = ( dil, dil )
					li += 1
				layers.append(m)
				# setattr(self, lname, m)

			for m in layers[-1][-1].modules():
				if hasattr(m, 'out_channels'):
					lastdim = m.out_channels

			if True: # deconv for final layer (2x higher resol)
				layers.append( torch.nn.ConvTranspose2d( \
					lastdim, dimout, kernel_size=3, \
					stride=2, output_padding=1, padding=1, bias=True) )
			else: # classic conv
				layers.append( torch.nn.Conv2d( \
					lastdim, dimout, kernel_size=3, \
					stride=1, padding=1, bias=True) )
			layers[-1].weight.data = \
				layers[-1].weight.data.normal_(0., self.final_std)
			
			self.trunk = torch.nn.Sequential(*layers )

		self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
		self.std = torch.FloatTensor([0.229, 0.224, 0.225])

	def get_last_layer_numchannels(self):
		return getattr(self,'layer4')[-1].conv1.in_channels

	def norm_image(self, x):
		mean = self.mean[None,:,None,None].type_as(x)
		std  = self.std[None,:,None,None].type_as(x)
		return (x - mean) / std

	def gkernel( self, sz, rel_scale, mu, sig ):
		g = torch.linspace( 0.5, sz-0.5, sz ).type_as(mu)
		g = ( (-(g[None,None,:] - mu[:,:,None]*rel_scale)**2) / \
			  (sig * rel_scale) ).exp()
		return g
		
	def make_kp_encoding(self, kp_loc_vis, im_size, grid_size):
		rel_scale = [g/i for g,i in zip(grid_size, im_size)]
		g_x = self.gkernel( grid_size[1], rel_scale[1], kp_loc_vis[:,0,:], 
							self.kp_encoding_sig )
		g_y = self.gkernel( grid_size[0], rel_scale[0], kp_loc_vis[:,1,:], 
							self.kp_encoding_sig )
		g = g_y[:,:,:,None] * g_x[:,:,None,:]
		g *= kp_loc_vis[:,2,:,None, None]
		return g

	def run_hc(self, images, kp_loc_vis=None, only_glob=False, skip_norm_image=False):

		if skip_norm_image:
			x  = self.layer0(images)
		else:
			x  = self.layer0(self.norm_image(images))

		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		x4 = self.layer4(x3)

		x4_avg = x4.mean((2,3), keepdim=True)  # TODO: keepdim=False

		if only_glob:
			return _, x4_avg

		# if (self.dimout_glob + self.dimout_glob_alpha) > 0:
		# 	out_glob = self.final_glob(x4_avg)
		# 	if only_glob:
		# 		return out_glob
		# else:
		# 	assert not only_glob

		xs = [x1, x2, x3, x4]

		if self.encode_input_keypoints:
			# append kp_encoding to all xs
			kp_encoding = self.make_kp_encoding( \
				kp_loc_vis, images.shape[2:], x.shape[2:] )
			for i in range(len(xs)):
				kp_up_ = Fu.interpolate( kp_encoding, size=xs[i].shape[2:], 
										 mode='bilinear' )
				xs[i] = torch.cat( (xs[i], kp_up_), dim=1 )

		hc = 0.

		upsize = None
		for hcl in self.hc_layers:
			if upsize==None:
				upsize = xs[hcl-1].shape[2:]
			lname = 'hc_layer%d'%hcl
			f = getattr(self, lname)(xs[hcl-1])
			fup = Fu.interpolate(f,size=upsize,mode='bilinear')
			hc = hc + fup * (1./len(self.hc_layers))

		out = self.final(hc)

		return out, x4_avg
		# if (self.dimout_glob+self.dimout_glob_alpha) > 0:
		# 	return out, out_glob
		# else:
		# 	return out, None

	def run_dil(self, images, kp_loc_vis=None, only_glob=False, skip_norm_image=False):

		assert not only_glob, 'not yet implemented'

		if skip_norm_image:
			l1in = images
		else:
			l1in = self.norm_image(images)

		if self.encode_input_keypoints:
			kp_encoding = self.make_kp_encoding( \
				kp_loc_vis, images.shape[2:], images.shape[2:] )
			l1in = torch.cat( (l1in, kp_encoding), dim=1 )

		return self.trunk(l1in)

	def forward(self, images, kp_loc_vis=None, only_glob=False, skip_norm_image=False):

		if self.architecture=='dilated':
			out = self.run_dil(images, kp_loc_vis=kp_loc_vis, only_glob=only_glob, skip_norm_image=skip_norm_image)
		elif self.architecture=='hypercolumns':
			out = self.run_hc(images, kp_loc_vis=kp_loc_vis, only_glob=only_glob, skip_norm_image=skip_norm_image)
		else:
			raise ValueError()

		return out


# taken from FCRN_pytorch on github
class FasterUpProj(nn.Module):
	# Faster UpProj decorder using pixelshuffle

	class faster_upconv(nn.Module):

		def __init__(self, in_channel):
			super(FasterUpProj.faster_upconv, self).__init__()

			self.conv1_ = nn.Sequential(collections.OrderedDict([
				('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=3)),
				('bn1', nn.BatchNorm2d(in_channel // 2)),
			]))

			self.conv2_ = nn.Sequential(collections.OrderedDict([
				('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(2, 3))),
				('bn1', nn.BatchNorm2d(in_channel // 2)),
			]))

			self.conv3_ = nn.Sequential(collections.OrderedDict([
				('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=(3, 2))),
				('bn1', nn.BatchNorm2d(in_channel // 2)),
			]))

			self.conv4_ = nn.Sequential(collections.OrderedDict([
				('conv1', nn.Conv2d(in_channel, in_channel // 2, kernel_size=2)),
				('bn1', nn.BatchNorm2d(in_channel // 2)),
			]))

			self.ps = nn.PixelShuffle(2)
			self.relu = nn.ReLU(inplace=True)

		def forward(self, x):
			# print('Upmodule x size = ', x.size())
			x1 = self.conv1_(nn.functional.pad(x, (1, 1, 1, 1)))
			x2 = self.conv2_(nn.functional.pad(x, (1, 1, 0, 1)))
			x3 = self.conv3_(nn.functional.pad(x, (0, 1, 1, 1)))
			x4 = self.conv4_(nn.functional.pad(x, (0, 1, 0, 1)))
			# print(x1.size(), x2.size(), x3.size(), x4.size())

			x = torch.cat((x1, x2, x3, x4), dim=1)

			x = self.ps(x)
			return x

	class FasterUpProjModule(nn.Module):
		def __init__(self, in_channels, smooth=False):
			super(FasterUpProj.FasterUpProjModule, self).__init__()
			out_channels = in_channels // 2
			self.smooth = smooth
			
			self.upper_branch = nn.Sequential(collections.OrderedDict([
				('faster_upconv', FasterUpProj.faster_upconv(in_channels)),
				('relu', nn.ReLU(inplace=True)),
				('conv', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
				('batchnorm', nn.BatchNorm2d(out_channels)),
			]))
			
			if self.smooth:
				self.bottom_branch = None
			else:			
				self.bottom_branch = FasterUpProj.faster_upconv(in_channels)
			
			self.relu = nn.ReLU(inplace=True)

		def forward(self, x):
			x1 = self.upper_branch(x)
			if self.smooth:
				x2 = Fu.interpolate(x[:,:x1.shape[1],:,:],size=x1.shape[2:],mode='bilinear')
			else:	
				x2 = self.bottom_branch(x)
			x = x1 + x2
			x = self.relu(x)
			return x

	def __init__(self, in_channel, n_layers=2, smooth=False, dimout=2):
		super(FasterUpProj, self).__init__()
		layers = []
		for l in range(n_layers):
			indim = in_channel // int(2**l)
			layers.append(self.FasterUpProjModule(indim,smooth=smooth))

		last = nn.Conv2d(indim//2, dimout, 3, padding=1)
		layers.append( last )
		self.trunk = nn.Sequential(*layers)

	def forward(self,x):
		return self.trunk(x)














# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def upconv(in_planes, out_planes, stride=2, groups=1, dilation=1):
#     """up convolution"""
# 	kernel_size = 2*(stride-1)+1
# 	pad = int((kernel_size-1)/2)
# 	return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride=stride, \
# 								padding=pad, output_padding=pad, groups=groups)
	

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# class UpBottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, upfactor=2, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(UpBottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
# 		self.conv2 = upconv(width, width, upfactor, groups)
# 		self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.scale = scale
#         self.stride = stride

#     def forward(self, x):
		
# 		identity = x
		
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         out += identity
# 		identity = Fu.interpolate(x,size=out.shape[2:],mode='bilinear')

#         out = self.relu(out)

#         return out