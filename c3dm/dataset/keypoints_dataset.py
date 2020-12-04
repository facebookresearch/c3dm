# Copyright (c) Facebook, Inc. and its affiliates.

import torch

import os
import sys
import json
import copy
import glob

import pickle, gzip

import numpy as np
import torch

from PIL import Image

from torch.utils import data
from tools.utils import NumpySeedFix, auto_init_args

class KeypointsDataset(data.Dataset):
	"""
	This is a generalized class suitable for storing object keypoint annotations
	
	The input jsonfile needs to be a list of dictionaries 
	(one dictionary per pose annotation) of the form:
	
	{ 
		# REQUIRED FIELDS #
		"kp_loc" : 2 x N list of keypoints
		"kp_vis" : 1 x N list of 1/0 boolean indicators
		# OPTIONAL FIELDS #
		"file_name": name of file from image_root
		"kp_loc_3d": 3 x N list of 3D keypoint locations in camera coords
	}
	
	"""
	def __init__( self, 
				  jsonfile=None, 
				  train=True, 
				  limit_to=0, 
				  limit_seq_to=-1,
				  rand_sample=0,
				  image_root=None,
				  mask_root=None,
				  depth_root=None,
				  refresh_db=False,
				  min_visible=0,
				  subsample=1,
				  load_images=True,
				  load_depths=True,
				  load_masks=True,
				  image_height=9*20*2,
				  image_width=9*20*2,
				  dilate_masks=5,
				  max_frame_diff = -1.,
				  max_angle_diff = 4.,
				  kp_conf_thr = 0.,
				  nrsfm_model_outputs = None,
				  box_crop_context=1.,
				  box_crop=False,
				):
		
		auto_init_args(self)

		self.load_db_file()

		has_classes = 'class_mask' in self.db[0]
		if has_classes: 
			self.class_db = self.get_class_db()
		else:
			self.class_db = None

		self.get_transposed_db()

	def get_transposed_db(self):

		print('getting transposed db ...')

		self.dbT = {}
		self.dbT['unqseq'] = sorted(list(set([e['seq'] for e in self.db])))
		self.dbT['seq_dict'] = {}
		self.dbT['seq'] = [e['seq'] for e in self.db]

		dict_seq = {s:i for i,s in enumerate(self.dbT['seq'])}
		for i in range(len(self.db)):
			# seq_ = self.dbT['unqseq'].index(self.db[i]['seq'])
			seq = dict_seq[self.db[i]['seq']]
			# assert seq_==seq
			# print('%d==%d' % (seq_,seq))
			if seq not in self.dbT['seq_dict']:
				self.dbT['seq_dict'][seq] = []
			self.dbT['seq_dict'][seq].append(i)

	def load_db_file(self):

		print("loading data from %s" % self.jsonfile)

		ext = self.jsonfile.split('.')[-1]
		if ext=='json':
			with open(self.jsonfile,'r') as data_file:    
				db = json.load(data_file)  
		elif ext=='pkl':
			with open(self.jsonfile,'rb') as data_file:
				db = pickle.load(data_file)
		elif ext=='pgz':
			with gzip.GzipFile(self.jsonfile, 'r') as data_file:
				db = pickle.load(data_file)
		else:
			raise ValueError('bad extension %s' % ext)

		if 'seq' not in db[0]:
			print('no sequence numbers! => filling with unique seq per image')
			for ei, e in enumerate(db): 
				e['seq_name'] = str(ei)
				e['seq'] = ei
			unqseq = list(range(len(db)))
		else:
			unqseq = sorted(list(set([e['seq'] for e in db])))
			for e in db:
				e['seq_name'] = copy.deepcopy(e['seq'])
				e['seq'] = unqseq.index(e['seq'])
		
		print("data train=%d , n frames = %d, n seq = %d" % \
			(self.train, len(db), len(unqseq)))

		self.db = db

		self.restrict_images()

	def get_class_db(self):
		print('parsing class db ...')
		masks     = np.stack([np.array(e['class_mask']) for e in self.db])
		unq_masks = np.unique(masks, axis=0)
		n_cls     = unq_masks.shape[0]

		class_db = {tuple(m.tolist()):[] for m in unq_masks}
		for ei,e in enumerate(self.db):
			class_db[tuple(e['class_mask'])].append(ei)
		class_db = list(class_db.values())

		for eis in class_db: # sanity check
			cls_array = np.stack([self.db[ei]['class_mask'] for ei in eis])
			assert ((cls_array - cls_array[0:1,:])**2).sum()<=1e-6

		return class_db

	def restrict_images(self):

		print( "limitting dataset to seqs: " + str(self.limit_seq_to) )
		if type(self.limit_seq_to) in (tuple,list):
			if len(self.limit_seq_to) > 1 or self.limit_seq_to[0] >= 0:
				self.db = [f for f in self.db if f['seq'] in self.limit_seq_to ]            
		elif type(self.limit_seq_to)==int:
			if self.limit_seq_to > 0:
				self.db = [f for f in self.db if f['seq'] < self.limit_seq_to ]
		else:
			assert False, "bad seq limit type"

		if self.limit_to > 0:
			tgtnum = min( self.limit_to, len(self.db) )
			prm = list(range(len(self.db)))[0:tgtnum]
			# with NumpySeedFix(): 
			# 	prm = np.random.permutation( \
			# 				len(self.db))[0:tgtnum]
			print( "limitting dataset to %d samples" % tgtnum )
			self.db = [self.db[i] for i in prm]
	
		if self.subsample > 1:
			orig_len = len(self.db)
			self.db = [self.db[i] for i in range(0, len(self.db), self.subsample)]
			print('db subsampled %d -> %d' % (orig_len, len(self.db)))

		if self.kp_conf_thr > 0. and 'kp_conf' in self.db[0]:
			for e in self.db:
				v = torch.FloatTensor(e['kp_vis'])
				c = torch.FloatTensor(e['kp_conf'])
				e['kp_vis'] = (c > self.kp_conf_thr).float().tolist()
		
		if self.min_visible > 0:
			len_orig = len(self.db)
			self.db = [ e for e in self.db \
				if (torch.FloatTensor(e['kp_vis'])>0).float().sum()>self.min_visible]
			print('kept %3.1f %% entries' % (100.*len(self.db)/float(len_orig)) )
			assert len(self.db) > 10

	def resize_image(self, image, mode='bilinear'):
		image_size = [self.image_height, self.image_width]
		minscale = min(image_size[i] / image.shape[i+1] for i in [0, 1])
		imre = torch.nn.functional.interpolate( \
			image[None], scale_factor=minscale, mode=mode)[0]
		imre_ = torch.zeros(image.shape[0],image_size[0],image_size[1])
		imre_[:,0:imre.shape[1],0:imre.shape[2]] = imre
		return imre_, minscale

	def load_image(self, entry):
		im = np.array(Image.open(entry['image_path']).convert('RGB'))
		im = im.transpose((2,0,1))
		im = im.astype(np.float32) / 255.
		return im

	def crop_around_box(self, entry, box_context=1.):
		bbox = entry['bbox'].clone() # [xmin, ymin, w, h]

		# increase box size
		c = box_context
		bbox[0] -= bbox[2]*c/2
		bbox[1] -= bbox[3]*c/2
		bbox[2] += bbox[2]*c
		bbox[3] += bbox[3]*c
		bbox = bbox.long()

		# assert bbox[2] >= 2, 'weird box!'
		# assert bbox[3] >= 2, 'weird box!'
		bbox[2:4] = torch.clamp(bbox[2:4], 2)
		entry['orig_image_size'] = bbox[[3,2]].float()
		
		bbox[2:4] += bbox[0:2]+1 # convert to [xmin, ymin, xmax, ymax]
		for k in ['images', 'masks', 'depths']:
			if getattr(self, 'load_'+k) and k in entry:
				crop_tensor = entry[k]
				bbox[[0,2]] = torch.clamp(bbox[[0,2]], 0., crop_tensor.shape[2])
				bbox[[1,3]] = torch.clamp(bbox[[1,3]], 0., crop_tensor.shape[1])
				crop_tensor = crop_tensor[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]			
				assert all(c>0 for c in crop_tensor.shape), 'squashed image'
				entry[k] = crop_tensor
		entry['kp_loc'] = entry['kp_loc'] - bbox[0:2,None].float()
		return entry


	def __len__(self):
		if self.rand_sample > 0:
			return self.rand_sample
		else:
			return len(self.db)

	def __getitem__(self, index):

		assert index < len(self.db), \
			'index %d out of range (%d)' % (index, len(self.db))

		entry = copy.deepcopy(self.db[index])

		if self.image_root is not None and 'image_path' in entry:
			entry['image_path'] = os.path.join(self.image_root,entry['image_path'])
		if self.mask_root is not None and 'mask_path' in entry:
			entry['mask_path'] = os.path.join(self.mask_root,entry['mask_path'])
		if self.depth_root is not None and 'depth_path' in entry:
			entry['depth_path'] = os.path.join(self.depth_root,entry['depth_path'])

		if self.load_images:
			entry['images'] = self.load_image(entry)
			entry['orig_image_size'] = list(entry['images'].shape[1:])

		if self.load_depths:
			entry['depths'] = load_depth(entry)

		if self.load_masks:
			entry['masks'] = load_mask(entry)
			if entry['masks'] is None:
				entry['masks'] = np.zeros(entry['images'].shape[1:3] \
										  )[None].astype(np.float32)
			else:
				# assert entry['masks'].shape[1:3]==entry['images'].shape[1:3]
				if self.load_images and \
				   entry['masks'].shape[1:3] != entry['images'].shape[1:3]:
					# print(entry['mask_path'])
					# print(entry['image_path'])
					# import pdb; pdb.set_trace()
					print('bad mask size!!!!')
					# print(entry['image_path'])
					# print(entry['mask_path'])
					# import pdb; pdb.set_trace()
					entry['masks'] = np.zeros(entry['images'].shape[1:3] \
										  )[None].astype(np.float32)

		# convert to torch Tensors where possible
		for fld in ( 'kp_loc', 'kp_vis', 'kp_loc_3d', 
					'class_mask', 'kp_defined', 'images', 
					'orig_image_size', 'masks', 'K', 'depths', 'bbox',
					'kp_conf', 'R', 'T'):
			if fld in entry:
				entry[fld] = torch.FloatTensor(entry[fld])

		# first crop if needed, then resize
		if self.box_crop and self.load_images:
			entry = self.crop_around_box(entry, self.box_crop_context)

		if 'sfm_model' not in entry:
			entry['sfm_model'] = '<NO_MODEL>'
		
		entry['K_orig'] = entry['K'].clone()

		if self.load_images:
			# resize image
			entry['images'], scale = self.resize_image(entry['images'], 
													   mode='bilinear')	
			for fld in ('kp_loc', 'kp_loc_3d', 'K'):
				if fld in entry:
					entry[fld] *= scale
				if fld=='K':
					entry[fld][2,2] = 1. 
		else:
			scale = 1.

		if self.load_masks:
			entry['masks'], _ = self.resize_image(entry['masks'], 
												  mode='nearest')			
			if self.dilate_masks > 0:
				#print('mask dilation')
				entry['masks'] = torch.nn.functional.max_pool2d(
									entry['masks'],
									self.dilate_masks*2+1, 
									stride=1, 
									padding=self.dilate_masks )
			elif self.dilate_masks < 0:
				imask_dil = torch.nn.functional.max_pool2d(
									1-entry['masks'],
									abs(self.dilate_masks)*2+1, 
									stride=1, 
									padding=abs(self.dilate_masks) )
				entry['masks'] = torch.clamp(entry['masks'] - imask_dil, 0.)
				
		if self.load_depths:
			entry['depths'], _ = self.resize_image(entry['depths'], 
												   mode='nearest')
			entry['depths'] *= scale

		if 'p3d_info' in entry:  # filter the kp out of bbox
			bbox = torch.FloatTensor(entry['p3d_info']['bbox'])
			bbox_vis, bbox_err = bbox_kp_visibility( \
									bbox, entry['kp_loc'], entry['kp_vis'])
			entry['kp_vis'] = entry['kp_vis'] * bbox_vis.float()

		# mask out invisible
		entry['kp_loc'] = entry['kp_loc'] * entry['kp_vis'][None]

		return entry


def bbox_kp_visibility(bbox, keypoints, vis):
	bx,by,bw,bh = bbox
	x = keypoints[0]; y = keypoints[1]
	ctx_ = 0.1
	in_box = (x>=bx-ctx_*bw) * (x<=bx+bw*(1+ctx_)) * \
			 (y>=by-ctx_*bh) * (y<=by+bh*(1+ctx_))
	
	in_box = in_box * (vis==1)

	err = torch.stack( [ (bx-ctx_*bw)-x,
							x-(bx+bw*(1+ctx_)),
							(by-ctx_*bh)-y,
							y-(by+bh*(1+ctx_)) ] )
	err = torch.relu(err) * vis[None].float()
	err = torch.stack( ( torch.max( err[0],err[1] ),
							torch.max( err[2],err[3] ) ) ).max(dim=1)[0]
	
	return in_box, err


def read_colmap_depth(path):
	with open(path, "rb") as fid:
		width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
												usecols=(0, 1, 2), dtype=int)
		fid.seek(0)
		num_delimiter = 0
		byte = fid.read(1)
		while True:
			if byte == b"&":
				num_delimiter += 1
				if num_delimiter >= 3:
					break
			byte = fid.read(1)
		array = np.fromfile(fid, np.float32)
	array = array.reshape((width, height, channels), order="F")
	return np.transpose(array, (1, 0, 2)).squeeze()


def load_depth(entry):
	if entry['depth_path'].endswith('<NO_DEPTH>'):
		# we dont have depth
		d = np.ones(entry['images'].shape[1:]).astype(float)[None]
	else:
		ext = os.path.splitext(entry['depth_path'])[-1]
		if ext=='.bin':  # colmap binary format
			d = read_colmap_depth(entry['depth_path'])
			# clamp the values
			min_depth, max_depth = np.percentile(d, [1, 95])
			d[d < min_depth] = min_depth
			d[d > max_depth] = max_depth
			d = d.astype(np.float32)[None]
		elif ext=='.png':  # ldos depth
			postfixl = len('081276300.rgb.jpg')
			dpath_corrected = glob.glob(entry['depth_path'][0:-postfixl]+'*')
			assert len(dpath_corrected)==1
			d = np.array(Image.open(dpath_corrected[0])).astype(float)[None]
			d /= 1000. # to meters
		elif ext=='.tiff':  # sparse colmap depth
			d = np.array(Image.open(entry['depth_path'])).astype(float)[None]
		else:
			raise ValueError('unsupported depth ext "%s"' % ext)
	
	return d


def load_mask(entry):
	# fix for birds
	if not os.path.isfile(entry['mask_path']):
		for ext in ('.png', '.jpg'):
			new_path = os.path.splitext(entry['mask_path'])[0] + ext
			if os.path.isfile(new_path):
				entry['mask_path'] = new_path
	
	if not os.path.isfile(entry['mask_path']):
		print('no mask!')
		print(entry['mask_path'])
		mask = None
	else:
		mask = np.array(Image.open(entry['mask_path']))
		mask = mask.astype(np.float32)[None]
	return mask