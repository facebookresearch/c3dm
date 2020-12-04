# Copyright (c) Facebook, Inc. and its affiliates.

import os
import copy
import io
import gzip
import urllib.request

from dataset.dataset_configs import (
	IMAGE_ROOTS, MASK_ROOTS, DEPTH_ROOTS, DATASET_ROOT, DATASET_CFG,
	IMAGE_URLS, MASK_URLS, DEPTH_URLS
)
from dataset.keypoints_dataset import KeypointsDataset

from tools import utils

def dataset_zoo( dataset_name='freicars',
				 sets_to_load = ['train','val'],
				 force_download = False,
				 test_on_trainset=False,
				 TRAIN= { 'rand_sample': 6000,
						  'limit_to': -1,
						  'limit_seq_to': [-1],
						  'subsample': 1,
						  'dilate_masks': 5,
						  },
				 VAL  = { 'rand_sample': 1000,
						  'limit_to': -1,
						  'limit_seq_to': -1,  
						  'subsample': 1,
						  'dilate_masks': 0,
						  },
				 TEST = { 'rand_sample': -1,
						  'limit_seq_to': -1,
						  'limit_to': -1,
						  'subsample': 1, 
						  'dilate_masks': 0,
						}, 
		 		 **kwargs ):

	main_root = DATASET_ROOT
	ext = '.json'
	json_train = os.path.join( main_root, dataset_name + '_train' + ext )
	json_val   = os.path.join( main_root, dataset_name + '_val' + ext )

	image_root_train, image_root_val = get_train_val_roots(dataset_name, IMAGE_ROOTS, IMAGE_URLS)
	mask_root_train, mask_root_val = get_train_val_roots(dataset_name, MASK_ROOTS, MASK_URLS)
	depth_root_train, depth_root_val = get_train_val_roots(dataset_name, DEPTH_ROOTS, DEPTH_URLS)

	# auto-download dataset file if doesnt exist
	for json_file in (json_train, json_val):
		if not os.path.isfile(json_file) or force_download:
			download_dataset_json(json_file)

	dataset_train = None
	dataset_val   = None
	dataset_test  = None

	if dataset_name in DATASET_CFG:
		dataset_cfg = copy.deepcopy(DATASET_CFG[dataset_name])
	else:
		dataset_cfg = copy.deepcopy(DATASET_CFG['default'])
	TRAIN, VAL, TEST = [ copy.deepcopy(set_) for set_ in (TRAIN, VAL, TEST) ]
	for set_ in (TRAIN, VAL, TEST):
		set_.update(dataset_cfg)
		print(set_)

	if 'train' in sets_to_load:
		dataset_train = KeypointsDataset(\
			image_root=image_root_train,
			mask_root=mask_root_train,
			depth_root=depth_root_train,
			jsonfile=json_train, train=True, **TRAIN)
	if 'val' in sets_to_load:
		if dataset_name in ('celeba_ff',):
			TEST['box_crop'] = True
			VAL['box_crop'] = True
		if test_on_trainset:
			image_root_val, json_val = image_root_train, json_train
		dataset_val   = KeypointsDataset(\
			image_root=image_root_val,
			mask_root=mask_root_val,
			depth_root=depth_root_val,
			jsonfile=json_val, train=False, **VAL)
		dataset_test  = KeypointsDataset(\
			image_root=image_root_val,
			mask_root=mask_root_val,
			depth_root=depth_root_val,
			jsonfile=json_val, train=False, **TEST)

	return 	dataset_train, dataset_val, dataset_test

def get_train_val_roots(dataset_name, image_roots, urls):
	if dataset_name not in image_roots:
		return None, None

	image_roots = copy.copy(image_roots[dataset_name])
	if len(image_roots) == 2:
		return image_roots
	elif len(image_roots) == 1:
		return image_roots[0], image_roots[0]
	else:
		raise ValueError('Wrong image roots format.')

	for subset_idx, images_dir in enumerate(image_roots[dataset_name]):
		if not os.path.exists(images_dir):
			if dataset_name not in urls:
				raise ValueError(
					f"Images for {dataset_name} not found in {images_dir}. "
					"Please download manually."
				)
			url = urls[dataset_name][subset_idx]
			print('Downloading images to %s from %s' % (images_dir, url))
			utils.untar_to_dir(url, images_dir)


def download_dataset_json(json_file):
	from dataset.dataset_configs import DATASET_URL
	
	json_dir  = '/'.join(json_file.split('/')[0:-1])
	json_name = json_file.split('/')[-1].split('.')[0]
	os.makedirs(json_dir, exist_ok=True)

	url = DATASET_URL[json_name]
	print('downloading dataset json %s from %s' % (json_name, url))
	
	response = urllib.request.urlopen(url)
	compressed_file = io.BytesIO(response.read())
	decompressed_file = gzip.GzipFile(fileobj=compressed_file)

	try:
		with open(json_file, 'wb') as outfile:
			outfile.write(decompressed_file.read())
	except:
		if os.path.isfile(json_file):
			os.remove(json_file)
	
	# can be zipped
	# print('checking dataset')
	# with open(json_file,'r') as f:
	# 	dt = json.load(f)
	# assert dt['dataset']==json_name
