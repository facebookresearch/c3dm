# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np
import copy
from model import load_nrsfm_model
from tools.cache_preds import cache_preds

def run_c3dpo_model_on_dset(dset, nrsfm_exp_dir):

	print('caching c3dpo outputs')

	# make a dataset copy without any random sampling 
	# and image/mask/depth loading
	dset_copy = copy.deepcopy(dset)
	dset_copy.load_images = False
	dset_copy.load_masks = False
	dset_copy.load_depths = False
	dset_copy.rand_sample = -1

	nrsfm_model, nrsfm_cfg = load_nrsfm_model(nrsfm_exp_dir, get_cfg=True)
	nrsfm_model.cuda()
	nrsfm_model.eval()

	loader = torch.utils.data.DataLoader( \
					dset_copy,
					num_workers=0, 
					pin_memory=True,
					batch_size=nrsfm_cfg.batch_size )

	cache_vars = ('phi', 'image_path')

	cache = cache_preds(nrsfm_model, loader,
		cache_vars=cache_vars, cat=True)

	dset.nrsfm_model_outputs = cache
