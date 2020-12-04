# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np

import torch
from tools.utils import Timer
from torch.utils.data.sampler import Sampler
from torch._six import int_classes as _int_classes


class SceneBatchSampler(Sampler):
	def __init__(self, sampler, batch_size, drop_last, \
					   train=True, strategy='uniform_viewpoints'):
		if not isinstance(sampler, Sampler):
			raise ValueError("sampler should be an instance of "
							 "torch.utils.data.Sampler, but got sampler={}"
							 .format(sampler))
		if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
				batch_size <= 0:
			raise ValueError("batch_size should be a positive integeral value, "
							 "but got batch_size={}".format(batch_size))
		if not isinstance(drop_last, bool):
			raise ValueError("drop_last should be a boolean value, but got "
							 "drop_last={}".format(drop_last))
		
		assert strategy == 'uniform_viewpoints' 

		self.sampler                  = sampler
		self.batch_size               = batch_size
		self.drop_last                = drop_last
		self.strategy 				  = strategy
		self.train                    = train
		self.restrict_seq		      = None
		
		
	def __iter__(self):
		batch = []
		for idx,_ in enumerate(self.sampler):
			ii = idx % self.batch_size
			if ii==0:
				sample_fun = {
					'uniform_viewpoints': self.sample_batch_vp_diff,
				}[self.strategy]
				with Timer(name='batch_sample', quiet=True):
					batch, seq = sample_fun(idx)
			if ii==(self.batch_size-1):
				yield batch
				batch = []

	def _get_dataset_yaws(self):
		dataset       = self.sampler.data_source
		rots = dataset.nrsfm_model_outputs['phi']['R']
		pr_axes = rots[:, -1, :]
		up = torch.svd(pr_axes)[2][:, -1]
		x = torch.cross(up, torch.tensor([0., 0., 1.]))
		x = x / x.norm()
		y = torch.cross(x, up)
		y = y / y.norm()
		x_c = torch.matmul(pr_axes, x)
		y_c = torch.matmul(pr_axes, y)
		yaw = torch.atan2(x_c, y_c)

		return yaw

	
	def sample_batch_vp_diff(self, idx):
		dataset       = self.sampler.data_source

		# get the cached log rots
		assert (
			hasattr(dataset, 'nrsfm_model_outputs') and 
				dataset.nrsfm_model_outputs is not None
		), 'make sure to set cfg.annotate_with_c3dpo_outputs=True'

		yaws = self._get_dataset_yaws()
		hist, edges = np.histogram(yaws, bins=16)
		bins = (yaws.cpu().data.numpy().reshape(-1, 1) > edges[1:]).sum(axis=1)
		weights = 1. / hist[bins]
		weights /= weights.sum()
		pivot = np.random.choice(np.arange(len(dataset.db)), p=weights)

		seq = dataset.dbT['seq'][pivot]

		rots = dataset.nrsfm_model_outputs['phi']['R']

		seqs = rots.new_tensor(dataset.dbT['seq'], dtype=torch.int64)
		# convert bool array to indices
		okdata = (seqs != seqs[pivot]).nonzero().view(-1).tolist()

		for o in okdata:
			assert o <= len(dataset.db), \
				'%d out of range (%d)!' % (o, len(dataset.db))


		if len(okdata) >= (self.batch_size-1):
			replace = False
		else:
			replace = True
			if len(okdata)==0:
				print('no samples!!')
				okdata = list(range(len(dataset.db)))
		if weights is not None:  # cross with okdata:
			weights = weights[okdata] / weights[okdata].sum()
		sample = np.random.choice(okdata, \
			self.batch_size-1, replace=replace, p=weights).tolist()
		sample.insert(0, pivot)

		for si, s in enumerate(sample):
			assert s < len(dataset.db), \
				'%d out of range (%d)!' % (s, len(dataset.db))

		return sample, seq

	def __len__(self):
		if self.drop_last:
			return len(self.sampler) // self.batch_size
		else:
			return (len(self.sampler) + self.batch_size - 1) // self.batch_size
