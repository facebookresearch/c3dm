#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy, os, sys, time
import itertools as itt
import yaml

# torch imports
import numpy as np
import torch
from dataset.batch_samplers import SceneBatchSampler
from   dataset.dataset_zoo import dataset_zoo
from   dataset.eval_zoo import eval_zoo
from dataset.c3dpo_annotate import run_c3dpo_model_on_dset

from model import Model

from config import set_config_from_file, set_config, \
					get_arg_parser, dump_config, get_default_args, auto_init_args

from tools.attr_dict import nested_attr_dict
from tools.utils import get_net_input, pprint_dict
from tools import utils
from tools.stats import Stats
from tools.vis_utils import get_visdom_env
from tools.model_io import find_last_checkpoint, purge_epoch, \
						   load_model, get_checkpoint, save_model
from tools.cache_preds import cache_preds


def init_model(cfg,force_load=False,clear_stats=False,add_log_vars=None):

	# get the model
	model = Model(**cfg.MODEL)

	# obtain the network outputs that should be logged
	if hasattr(model,'log_vars'):
		log_vars = copy.deepcopy(model.log_vars)
	else:
		log_vars = ['objective']
	if add_log_vars:
		log_vars.extend(copy.deepcopy(add_log_vars))

	visdom_env_charts = get_visdom_env(cfg) + "_charts"

	# init stats struct
	stats = Stats( log_vars, visdom_env=visdom_env_charts, \
				   verbose=False, visdom_server=cfg.visdom_server, \
				   visdom_port=cfg.visdom_port )    

	model_path = None
	if cfg.resume_epoch > 0:
		model_path = get_checkpoint(cfg.exp_dir,cfg.resume_epoch)
	elif cfg.resume_epoch == -1: # find the last checkpoint
		model_path = find_last_checkpoint(cfg.exp_dir)
	
	optimizer_state = None

	if model_path is None and force_load:
		from dataset.dataset_configs import C3DM_URLS
		url = C3DM_URLS[cfg.DATASET.dataset_name]
		print('Downloading C3DM model %s from %s' % (cfg.DATASET.dataset_name, url))
		utils.untar_to_dir(url, cfg.exp_dir)
		model_path = find_last_checkpoint(cfg.exp_dir)

	if model_path is not None:
		print( "found previous model %s" % model_path )
		if force_load or cfg.resume:
			print( "   -> resuming" )
			model_state_dict, stats_load, optimizer_state = load_model(model_path)
			if not clear_stats: 
				if stats_load is None:
					print("   -> bad stats! -> clearing")
				else:
					stats = stats_load
			else:
				print("   -> clearing stats")
			try:
				model.load_state_dict(model_state_dict, strict=True)
			except RuntimeError as e:
				print('!!!!! cant load state dict in strict mode:')
				print(e)
				print('loading in non-strict mode ...')
				model.load_state_dict(model_state_dict, strict=False)
			model.log_vars = log_vars
		else:
			print( "   -> but not resuming -> starting from scratch" )
	elif force_load:
		print('!! CANNOT RESUME FROM A CHECKPOINT !!')

	# update in case it got lost during load:
	stats.visdom_env    = visdom_env_charts
	stats.visdom_server = cfg.visdom_server
	stats.visdom_port   = cfg.visdom_port
	#stats.plot_file = os.path.join(cfg.exp_dir,'train_stats.pdf')
	stats.synchronize_logged_vars(log_vars)

	return model, stats, optimizer_state

def init_optimizer(model,optimizer_state,
					PARAM_GROUPS=(),
					freeze_bn=False,
					breed='sgd',
					weight_decay=0.0005,
					lr_policy='multistep',
					lr=0.001,
					gamma=0.1,
					momentum=0.9,
					betas=(0.9,0.999),
					milestones=[100,],
					max_epochs=300,
					):    

	# init the optimizer
	if hasattr(model,'_get_param_groups'): # use the model function
		p_groups = model._get_param_groups(lr,wd=weight_decay)
	else:
		allprm = [prm for prm in model.parameters() if prm.requires_grad]
		p_groups = [{'params': allprm, 'lr': lr}]
	
	if breed=='sgd':
		optimizer = torch.optim.SGD( p_groups, lr=lr, \
							   momentum=momentum, \
							   weight_decay=weight_decay )

	elif breed=='adagrad':
		optimizer = torch.optim.Adagrad( p_groups, lr=lr, \
							   weight_decay=weight_decay )

	elif breed=='adam':
		optimizer = torch.optim.Adam( p_groups, lr=lr, \
							   betas=betas, \
							   weight_decay=weight_decay )
	
	else:
		raise ValueError("no such solver type %s" % breed)
	print("  -> solver type = %s" % breed)

	if lr_policy=='multistep':
		scheduler = torch.optim.lr_scheduler.MultiStepLR( \
					optimizer, milestones=milestones, gamma=gamma)
	else:
		raise ValueError("no such lr policy %s" % lr_policy)    

	# add the max epochs here!
	scheduler.max_epochs = max_epochs

	if optimizer_state is not None:
		print("  -> setting loaded optimizer state")        
		optimizer.load_state_dict(optimizer_state)
		optimizer.param_groups[0]['momentum'] = momentum
		optimizer.param_groups[0]['dampening'] = 0.0

	optimizer.zero_grad()
	return optimizer, scheduler

def run_training(cfg):
	# run the training loops
	
	# make the exp dir
	os.makedirs(cfg.exp_dir,exist_ok=True)

	# set the seed
	np.random.seed(cfg.seed)

	# dump the exp config to the exp dir
	dump_config(cfg)

	# setup datasets
	dset_train, dset_val, dset_test = dataset_zoo(**cfg.DATASET)

	 # init loaders
	if cfg.batch_sampler=='default':
		trainloader = torch.utils.data.DataLoader( dset_train, 
							num_workers=cfg.num_workers, pin_memory=True,
							batch_size=cfg.batch_size, shuffle=False )
	elif cfg.batch_sampler=='sequence':
		trainloader = torch.utils.data.DataLoader( dset_train, 
							num_workers=cfg.num_workers, pin_memory=True,
							batch_sampler=SceneBatchSampler(
									torch.utils.data.SequentialSampler(dset_train),
									cfg.batch_size,
									True,
		) )
	else:
		raise BaseException()

	if dset_val is not None:
		if cfg.batch_sampler=='default':
			valloader = torch.utils.data.DataLoader( dset_val, 
								num_workers=cfg.num_workers, pin_memory=True,
								batch_size=cfg.batch_size, shuffle=False )
		elif cfg.batch_sampler=='sequence':
			valloader = torch.utils.data.DataLoader( dset_val, 
								num_workers=cfg.num_workers, pin_memory=True,
								batch_sampler=SceneBatchSampler( \
									 torch.utils.data.SequentialSampler(dset_val),
									 cfg.batch_size,
									 True,
			) )
		else:
			raise BaseException()
	else:
		valloader = None

	# test loaders
	if dset_test is not None:
		testloader = torch.utils.data.DataLoader(dset_test, 
				num_workers=cfg.num_workers, pin_memory=True,
				batch_size=cfg.batch_size, shuffle=False,
				)
		_,_,eval_vars = eval_zoo(cfg.DATASET.dataset_name)
	else:
		testloader = None
		eval_vars = None


	# init the model    
	model, stats, optimizer_state = init_model(cfg,add_log_vars=eval_vars)
	start_epoch = stats.epoch + 1

	# annotate dataset with c3dpo outputs
	if cfg.annotate_with_c3dpo_outputs:
		for dset in dset_train, dset_val, dset_test:
			if dset is not None:
				run_c3dpo_model_on_dset(dset, cfg.MODEL.nrsfm_exp_path)

	# move model to gpu
	model.cuda(0)

	# init the optimizer
	optimizer, scheduler = init_optimizer(\
		model, optimizer_state=optimizer_state, **cfg.SOLVER)
	
	# loop through epochs
	scheduler.last_epoch = start_epoch
	for epoch in range(start_epoch, cfg.SOLVER.max_epochs):
		with stats: # automatic new_epoch and plotting of stats at every epoch start
			
			print("scheduler lr = %1.2e" % float(scheduler.get_lr()[-1]))
			
			# train loop
			trainvalidate(model, stats, epoch, trainloader, optimizer, False, \
										visdom_env_root=get_visdom_env(cfg), **cfg )
			
			# val loop
			if valloader is not None:
				trainvalidate(model, stats, epoch, valloader,   optimizer, True,  \
											visdom_env_root=get_visdom_env(cfg), **cfg  )

			# eval loop (optional)
			if testloader is not None:
				if cfg.eval_interval >= 0:                
					if cfg.eval_interval == 0 or \
						((epoch % cfg.eval_interval)==0 and epoch > 0):
						torch.cuda.empty_cache() # we have memory heavy eval ...
						with torch.no_grad():
							run_eval(cfg,model,stats,testloader)

			assert stats.epoch==epoch, "inconsistent stats!"

			# delete previous models if required
			if cfg.store_checkpoints_purge > 0 and cfg.store_checkpoints:
				for prev_epoch in range(epoch-cfg.store_checkpoints_purge):
					period = cfg.store_checkpoints_purge_except_every
					if (period > 0 and prev_epoch % period == period - 1):
						continue
					purge_epoch(cfg.exp_dir,prev_epoch)

			# save model
			if cfg.store_checkpoints:
				outfile = get_checkpoint(cfg.exp_dir,epoch)        
				save_model(model,stats,outfile,optimizer=optimizer)

			scheduler.step()


def run_evaluation(cfg):
	np.random.seed(cfg.seed)

	# setup datasets
	dset_train, dset_val, dset_test = dataset_zoo(**cfg.DATASET)

	# test loaders
	testloader = torch.utils.data.DataLoader(
		dset_test, 
		num_workers=cfg.num_workers, pin_memory=True,
		batch_size=cfg.batch_size, shuffle=False,
	)
	_, _, eval_vars = eval_zoo(cfg.DATASET.dataset_name)

	# init the model    
	model, _, _ = init_model(cfg, force_load=True, add_log_vars=eval_vars)
	model.cuda(0)
	model.eval()

	# init the optimizer
	#optimizer, scheduler = init_optimizer(model, optimizer_state=optimizer_state, **cfg.SOLVER)

	# val loop
	#trainvalidate(model, stats, 0, valloader,   optimizer, True,
	#			  visdom_env_root=get_visdom_env(cfg), **cfg  )
			
	with torch.no_grad():
		run_eval(cfg, model, None, testloader)
		

def trainvalidate(  model,
					stats,
					epoch,
					loader,
					optimizer,
					validation,
					bp_var='objective',
					metric_print_interval=5,
					visualize_interval=0, 
					visdom_env_root='trainvalidate',
					**kwargs ):

	if validation:
		model.eval()
		trainmode = 'val'
	else:
		model.train()
		trainmode = 'train'

	t_start = time.time()

	# clear the visualisations on the first run in the epoch
	clear_visualisations = True

	# get the visdom env name
	visdom_env_imgs = visdom_env_root + "_images_" + trainmode

	#loader = itt.islice(loader, 1)
	n_batches = len(loader)
	for it, batch in enumerate(loader):

		last_iter = it==n_batches-1        

		# move to gpu where possible
		net_input = get_net_input(batch)
		
		# add epoch to the set of inputs
		net_input['epoch_now'] = int(epoch)

		if (not validation):
			optimizer.zero_grad()
			preds = model(**net_input)
		else:
			with torch.no_grad():
				preds = model(**net_input)

		# make sure we dont overwrite something
		assert not any( k in preds for k in net_input.keys() )    
		preds.update(net_input) # merge everything into one big dict

		# update the stats logger
		stats.update(preds,time_start=t_start,stat_set=trainmode)
		assert stats.it[trainmode]==it, "inconsistent stat iteration number!"

		# print textual status update
		if (it%metric_print_interval)==0 or last_iter:
			stats.print(stat_set=trainmode,max_it=n_batches)

		# optimizer step
		if (not validation):
			loss  = preds[bp_var]    
			loss.backward()
			optimizer.step()

		# visualize results
		if (visualize_interval>0) and (it%visualize_interval)==0:
			model.visualize( visdom_env_imgs, trainmode, \
						preds, stats, clear_env=clear_visualisations )
			clear_visualisations = False


def run_eval(cfg,model,stats,loader):

	if hasattr(model, 'embed_db_eval'):
		from dataset.dataset_configs import FILTER_DB_SETTINGS
		dset_name = cfg['DATASET']['dataset_name']
		if dset_name in FILTER_DB_SETTINGS:
			filter_settings = FILTER_DB_SETTINGS[dset_name]
		else:
			filter_settings = FILTER_DB_SETTINGS['default']
		print('filter settings: %s' % str(filter_settings))
		print('turning embed_db eval on!')
		prev_embed_db_eval = copy.deepcopy(model.embed_db_eval)
		model.embed_db_eval = True
		model.embed_db.filter_db(**filter_settings)

	eval_script, cache_vars, eval_vars = eval_zoo(cfg.DATASET.dataset_name)	
	if True:
		cached_preds = cache_preds(model, loader, stats=stats, 
								   cache_vars=cache_vars)
	else:
		cached_preds = cache_preds(model, loader, stats=stats, 
								   cache_vars=cache_vars, eval_mode=False)
		assert False, 'make sure not to continue beyond here!'
	
	results, _ = eval_script(cached_preds, eval_vars=eval_vars)
	if stats is not None:
		stats.update(results, stat_set='test') #, log_vars=results.keys())
		stats.print(stat_set='test')

	if hasattr(model, 'embed_db_eval'):
		model.embed_db_eval = prev_embed_db_eval



class ExperimentConfig(object):
	def __init__(   self,
					cfg_file=None,
					model_zoo='./data/torch_zoo/',
					exp_name='test',
					exp_idx=0,
					exp_dir='./data/exps/keypoint_densification/default/',
					gpu_idx=0,
					resume=True,
					seed=0,
					resume_epoch=-1,
					eval_interval=1,
					store_checkpoints=True,
					store_checkpoints_purge=1,
					store_checkpoints_purge_except_every=25,
					batch_size=10,
					num_workers=8,
					visdom_env='',
					collect_basis_before_eval=False,
					visdom_server='http://localhost',
					visdom_port=8097,
					metric_print_interval=5,
					visualize_interval=0,
					mode='trainval',
					batch_sampler='sequence',
					annotate_with_c3dpo_outputs=True,
					SOLVER  = get_default_args(init_optimizer),
					DATASET = get_default_args(dataset_zoo),
					MODEL   = get_default_args(Model),
				):

		self.cfg = get_default_args(ExperimentConfig)
		if cfg_file is not None:
			set_config_from_file(self.cfg,cfg_file)
		else:
			auto_init_args(self,tgt='cfg',can_overwrite=True)
		self.cfg = nested_attr_dict(self.cfg)

if __name__ == '__main__':
	
	torch.manual_seed(0)
	np.random.seed(0)
	# init the exp config
	exp = ExperimentConfig()

	set_config_from_file(exp.cfg, sys.argv[1])
	mode = 'train'
	if len(sys.argv) > 2 and sys.argv[2] == '--eval':
		mode = 'eval'

	pprint_dict(exp.cfg)

	#with open('freicars.yaml', 'w') as yaml_file:
	#	yaml.dump(exp.cfg, yaml_file, default_flow_style=False)

	os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = str(exp.cfg.gpu_idx)
	if not exp.cfg.model_zoo is None:
		os.environ["TORCH_MODEL_ZOO"] = exp.cfg.model_zoo
	
	if mode == 'eval':
		run_evaluation(exp.cfg)
	else:
		run_training(exp.cfg)
	

	