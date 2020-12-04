# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
Fu = F
from torch.autograd import Variable
import numpy as np
from math import exp

from tools.functions import avg_l2_dist, avg_l2_huber, image_meshgrid, huber, logexploss

if torch.cuda.is_available():
	T = torch.cuda
else:
	T = torch

def total_variation_loss(image):
	# shift one pixel and get difference (for both x and y direction)
	loss = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]) + \
		   torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
	return loss


class GaussianLayer(nn.Module):
	def __init__(self, sigma=1., separated=False):
		super(GaussianLayer, self).__init__()
		self.separated = separated

		filter_size = int(2*np.ceil(sigma)+1)
		generated_filters = gaussian(filter_size, sigma).reshape([1,filter_size])

		if self.separated:
			self.gaussian_filter_horizontal = nn.Conv2d(\
				in_channels=1, out_channels=1, \
				kernel_size=(1,filter_size), padding=(0,filter_size//2),bias=False)
			self.gaussian_filter_horizontal.weight.data.copy_(\
				generated_filters)
			self.gaussian_filter_vertical = nn.Conv2d(\
				in_channels=1, out_channels=1, \
				kernel_size=(filter_size,1), padding=(filter_size//2,0), bias=False)
			self.gaussian_filter_vertical.weight.data.copy_(\
				generated_filters.t())
		else:
			filter_full = generated_filters * generated_filters.t()
			self.gaussian_filter = nn.Conv2d(\
				in_channels=1, out_channels=1, \
				kernel_size=(filter_size,filter_size), 
				padding=(filter_size//2,filter_size//2),bias=False)
			self.gaussian_filter.weight.data = filter_full[None, None]
		
		# do not back prop!!!
		for prm in self.parameters():
			prm.requires_grad = False

	def forward(self, img):
		ba, dim, he, wi = img.shape
		img = torch.cat((img, img.new_ones(ba,1,he,wi)), dim=1)
		img = img.view(ba*(dim+1), 1, he, wi)
		if self.separated:
			imgb = self.gaussian_filter_horizontal(img)
			imgb = self.gaussian_filter_vertical(imgb)
		else:
			imgb = self.gaussian_filter(img)
		imgb = imgb.view(ba, dim+1, he, wi)
		imgb = imgb[ :, :dim, :, : ] / \
			torch.clamp(imgb[ :, dim:dim+1, :, : ], 0.001)
		return imgb


class TVLoss(nn.Module):
	def __init__(self):
		super(TVLoss, self).__init__()
		sobel_filter = torch.FloatTensor([[1, 0, -1],
										 [2, 0, -2],
										 [1, 0, -1]])
		sobel_filter = sobel_filter / sobel_filter.abs().sum()
		self.sobel_filter_horizontal = nn.Conv2d(
			in_channels=1, out_channels=1, bias=False,
			kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
		self.sobel_filter_horizontal.weight.data.copy_(sobel_filter)
		self.sobel_filter_vertical = nn.Conv2d(
			in_channels=1, out_channels=1, bias = False,
			kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
		self.sobel_filter_vertical.weight.data.copy_(sobel_filter.t())

		# do not back prop!!!
		for prm in self.parameters():
			prm.requires_grad = False
		
	def forward(self, im, masks=None):
		
		tv = self.sobel_filter_horizontal(im).abs() + \
			 self.sobel_filter_vertical(im).abs()

		if masks is not None:
			masks = Fu.interpolate(masks, tv.shape[2:], mode='nearest')
			tv = tv * masks
		
		return tv.mean()


class LapFilter(nn.Module):
	def __init__(self, size=5):
		super(LapFilter, self).__init__()

		# use gauss layer to setup the circular 2D filter (hacky)
		gauss = GaussianLayer(sigma=size, separated=False)

		flt = gauss.gaussian_filter.weight
		thr = flt[0, 0, flt.shape[2]//2, 0]
		flt = (flt >= thr).float()
		flt = flt / torch.clamp(flt.sum(), 1e-4)

		self.circ_filter = nn.Conv2d( 
				in_channels=1, 
				out_channels=1, 
				bias=False,
				kernel_size=size,
				padding=size
			)
		self.circ_filter.weight.data = flt.clone()

		# do not back prop!!!
		for prm in self.parameters():
			prm.requires_grad = False
		
	def forward(self, img, masks=None):
		
		ba, dim, he, wi = img.shape

		if (masks is not None) and (masks.shape[2:]!=img.shape[2:]):
			masks = Fu.interpolate(masks, (he, wi), mode='nearest')
		else:
			masks = img.new_ones(ba, 1, he, wi)
		
		imgf = img * masks
		imgf = torch.cat((imgf, masks), dim=1)
		imgf = imgf.view(ba*(dim+1), 1, he, wi)
		imgf = self.circ_filter(imgf)
		imgf = imgf.view(ba, dim+1, he, wi)
		imgf = imgf[ :, :dim, :, : ] / \
			torch.clamp(imgf[ :, dim:dim+1, :, : ], 0.001)
		
		return imgf

class LapLoss(nn.Module):
	def __init__(self, size=5):
		super(LapLoss, self).__init__()

		self.lapfilter = LapFilter(size=size)

		# do not back prop!!!
		for prm in self.parameters():
			prm.requires_grad = False
		
	def forward(self, img, masks=None):
		
		if masks is not None:
			masks = Fu.interpolate(masks, size=img.shape[2:], mode='nearest')
		else:
			masks = img[:,0:1,:,:] * 0. + 1.

		imgf = self.lapfilter(img, masks=masks)

		diff = (((img - imgf)*masks)**2).sum(dim=(1,2,3))
		diff = diff / torch.clamp(masks.sum(dim=(1,2,3)), 1e-4)

		return diff.mean(), imgf


## Perceptual VGG19 loss
class PerceptualVGG19(nn.Module):
	def __init__(self, feature_layers, use_normalization=True,
				 path=None, input_from_tanh=True, flatten=True,
				 ):
		super(PerceptualVGG19, self).__init__()
		if path != '' and path is not None:
			print('Loading pretrained model')
			model = models.vgg19(pretrained=False)
			model.load_state_dict(torch.load(path))
		else:
			model = models.vgg19(pretrained=True)
		model.float()
		model.eval()

		self.model = model
		self.feature_layers = feature_layers
		self.input_from_tanh = input_from_tanh
		self.flatten = flatten
		

		self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
		self.mean_tensor = None

		self.std = torch.FloatTensor([0.229, 0.224, 0.225])
		self.std_tensor = None

		self.use_normalization = use_normalization

		if torch.cuda.is_available():
			self.mean = self.mean.cuda()
			self.std = self.std.cuda()

		for param in self.parameters():
			param.requires_grad = False

	def normalize(self, x):
		if not self.use_normalization:
			return x

		if self.mean_tensor is None:
			self.mean_tensor = self.mean.view(1, 3, 1, 1)
			self.std_tensor = self.std.view(1, 3, 1, 1)

		if self.input_from_tanh:
			x = (x + 1) / 2
		
		return (x - self.mean_tensor) / self.std_tensor

	def run(self, x, resize):
		features = []
		masks = []
		h = x

		for f in range(max(self.feature_layers) + 1):
			h = self.model.features[f](h)
			if f in self.feature_layers:
				not_normed_features = h.clone()
				if resize:
					features.append(not_normed_features.view(h.size(0),-1))
				else:
					features.append(not_normed_features)

		if resize:
			features = torch.cat(features, dim=1)

		return masks, features

	def forward(self, x, resize=True):
		h = self.normalize(x)
		return self.run(h, resize)

class AppearanceLoss(nn.modules.Module):
	def __init__(self, n_l1_scales=4, sigma_coeff=1., huber_thr=0.03, border = 0.1):
		super(AppearanceLoss, self).__init__()
		self.n_l1_scales = n_l1_scales
		self.sigma_coeff = sigma_coeff
		self.huber_thr = huber_thr
		self.border=border
		self.perception_loss_module = PerceptualVGG19( feature_layers=[0, 5, 10, 15], 
													   use_normalization=True,
													   input_from_tanh=False,
													   flatten=False )
		self.perception_loss_module = self.perception_loss_module.cuda()

	def grayscale_transform(self, x):
		return x.mean(1,keepdim=True)

	def forward(self, input, target, sig=None, mask=None):
		# input/target an image between [0,1]
		input_rgb  = input 
		gt_tgt_rgb = target

		image_size = list(input.shape[2:])

		# mask both input and target borders
		border_in_pix = int(self.border * np.array(input.shape[2:]).mean())
		brd_mask = input.new_zeros(input.shape)
		brd_mask[:,:,border_in_pix:-border_in_pix,border_in_pix:-border_in_pix] = 1.
		
		if mask is not None:
			brd_mask *= mask
		
		input_rgb  = input_rgb * brd_mask
		gt_tgt_rgb = gt_tgt_rgb * brd_mask

		# make sure we got the right input
		assert gt_tgt_rgb.min() >= -0.001
		assert gt_tgt_rgb.max() <=  1.001

		# VGG
		_, fake_features = self.perception_loss_module(input_rgb,  resize=False)
		_, tgt_features  = self.perception_loss_module(gt_tgt_rgb, resize=False)
		loss_vgg = 0.	
		sig_vgg = sig 
		for fake, tgt in zip(fake_features,tgt_features):
			# vgg_df = huber(((fake-tgt)**2).mean(1,keepdim=True),scaling=self.huber_thr)
			vgg_df = huber(((fake-tgt)**2),scaling=self.huber_thr).mean(1,keepdim=True)
			if sig_vgg is not None:
				# first smooth the sigmas
				# g_sigma = sum(sig_vgg.shape[i]/fake.shape[i] for i in (2,3))*0.5
				# if g_sigma > 1.:
				# 	sig_vgg = gauss_filter(sig_vgg, g_sigma)
				sig_vgg = Fu.interpolate(sig_vgg, size=fake.shape[2:],mode='bilinear')
				
				loss_vgg = loss_vgg + \
					Fu.interpolate( \
						logexploss(vgg_df, sig_vgg, \
								   coeff=self.sigma_coeff, accum=False),
								   size=image_size )
			else:
				loss_vgg  = loss_vgg + Fu.interpolate(vgg_df, size=image_size)
				# loss_vgg = loss_vgg + vgg_df #.mean((1,2,3))

		# RGB L1 ... multiscale
		loss_rgb = 0.
		sig_rgb = sig
		for scale in range(self.n_l1_scales):
			if scale > 0:	
				input_rgb = Fu.interpolate(input_rgb, scale_factor=0.5, mode='bilinear')
				gt_tgt_rgb= Fu.interpolate(gt_tgt_rgb, scale_factor=0.5, mode='bilinear')
				if sig_rgb is not None:
					sig_rgb = Fu.interpolate(sig_rgb, scale_factor=0.5, mode='bilinear')
			rgb_diff = huber(((input_rgb-gt_tgt_rgb)**2),scaling=self.huber_thr).mean(1,keepdim=True)

			if sig is not None:
				loss_rgb  = loss_rgb  + Fu.interpolate(logexploss(rgb_diff,  sig_rgb, 
							coeff=self.sigma_coeff, accum=False), size=image_size)
			else:
				loss_rgb  = loss_rgb  + Fu.interpolate(rgb_diff, size=image_size)

		return loss_vgg, loss_rgb, 0


def multiscale_loss(pred, gt, n_scales=4, scaling=0.01, 
					downscale=0.5, per_dim_loss=False, loss_fun=None,
					grid=None):
	# basis rendering loss
	size = pred.shape[2:] 
	loss = 0.

	# get the gauss filter
	sig = 2 * (1/downscale) / 6.0 # as in scipy
	g_filter = GaussianLayer(sigma=sig, separated=True).to(pred.device)

	for scl in range(n_scales):
		if scl==0:
			gt_ = gt; p_ = pred; grid_ = grid
		else:
			p_ = g_filter(p_)
			p_ = Fu.interpolate(p_, scale_factor=downscale, mode='bilinear')
			gt_ = g_filter(gt_)
			gt_ = Fu.interpolate(gt_, scale_factor=downscale, mode='bilinear')
			if grid is not None:
				grid_ = g_filter(grid_)
				grid_ = Fu.interpolate(grid_, scale_factor=downscale, mode='bilinear')

		if grid is not None:
			gt_sample = Fu.grid_sample(gt_, grid_.permute(0, 2, 3, 1))
		else:
			gt_sample = gt_

		if loss_fun is None:
			if per_dim_loss:
				h = huber((p_ - gt_sample)**2, scaling=scaling).mean(dim=1, keepdim=True)
			else:
				h = huber(((p_ - gt_sample)**2).mean(dim=1, keepdim=True), scaling=scaling)
		else:
			h = loss_fun(p_, gt_sample)
		loss = loss + Fu.interpolate(h, size=size, mode='bilinear')
	return loss * (1 / n_scales)

