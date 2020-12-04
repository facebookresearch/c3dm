# Copyright (c) Facebook, Inc. and its affiliates.

from collections import defaultdict
import json
import os

import numpy as np
import torch
import trimesh
from visdom import Visdom

from dataset.dataset_configs import IMAGE_ROOTS
from dataset.keypoints_dataset import load_depth, load_mask
from tools.eval_functions import eval_depth_pcl, eval_full_pcl, eval_sparse_pcl
from tools.pcl_unproject import depth2pcl
import torch.nn.functional as Fu
from tqdm import tqdm
import pickle
import time

def eval_zoo(dataset_name, include_debug_vars=False):	

	if 'freicars_clickp_filtd' in dataset_name:
		eval_script = eval_freicars
		cache_vars = [ 
			'masks', 'depth_dense', 
			'K_orig', 'image_path',
			'orig_image_size', 
			'depth_path', 'mask_path', 
			'seq_name', 'R', 'T',
			'embed_db_shape_camera_coord',
			'cmr_faces',
			'kp_loc_3d',
			'shape_image_coord_cal',
			'nrsfm_shape_image_coord'
		]
		eval_vars = [  
				'EVAL_depth_scl_perspective_med',
				'EVAL_pcl_scl_perspective_med',
				'EVAL_pcl_corr_scl_perspective_med',
				'EVAL_depth_scl_orthographic_med',
				'EVAL_pcl_scl_orthographic_med',
				'EVAL_pcl_corr_scl_orthographic_med',
				'EVAL_depth_scl_perspective',
				'EVAL_pcl_scl_perspective',
				'EVAL_pcl_corr_scl_perspective',
				'EVAL_depth_scl_orthographic',
				'EVAL_pcl_scl_orthographic',
				'EVAL_pcl_corr_scl_orthographic',
				'EVAL_sparse_pcl',
				'EVAL_sparse_pcl_nrsfm',
			 ]
	elif dataset_name in ('celeba_ff',):
		eval_script = eval_florence
		cache_vars = [ 'masks', 'depth_dense', 
					   'K_orig', 'image_path',
					   'orig_image_size', 
					   'depth_path', 'mask_path', 
					   'seq_name',
					   'images',
					   'embed_db_shape_camera_coord',
					   'shape_image_coord_cal_dense',
					   'cmr_faces',
					   'kp_loc',
					   'mesh_path',
					   'shape_image_coord_best_scale',
					]
		eval_vars = [  \
				'EVAL_pcl_scl_recut_orthographic_flip_med',
				'EVAL_pcl_scl_orthographic_flip_med',
				'EVAL_pcl_orthographic_flip_med',
				'EVAL_pcl_scl_recut_orthographic_med',
				'EVAL_pcl_scl_orthographic_med',
				'EVAL_pcl_orthographic_med',
				'EVAL_pcl_scl_recut_orthographic_flip',
				'EVAL_pcl_scl_orthographic_flip',
				'EVAL_pcl_orthographic_flip',
				'EVAL_pcl_scl_recut_orthographic',
				'EVAL_pcl_scl_orthographic',
				'EVAL_pcl_orthographic',
			]
	
	elif 'pascal3d' in dataset_name:

		eval_script = eval_p3d
		cache_vars = [ 'masks', 'depth_dense', 
					   'image_path', 'R', 'T',
					   'orig_image_size', 
					   'mask_path', 
					   'images',
					   'embed_db_shape_camera_coord',
					   'shape_image_coord_cal_dense',
					   'cmr_faces',
					   'kp_loc',
					   'mesh_path' ]
		eval_vars = [  \
				'EVAL_pcl_scl_detkp',
				'EVAL_pcl_corr_scl_detkp',
				'EVAL_pcl_corr_scl_detkp_med',
			]
	
	else:
		eval_script = eval_dummy
		cache_vars = [ 'images', ]
		eval_vars = [  'EVAL_pcl_dist_scl', ]

	return eval_script, cache_vars, eval_vars


def eval_dummy(cached_preds, eval_vars=None):
	return {'EVAL_pcl_dist_scl': -1.}, None


def load_freicar_gt_pcl():
	print('loading gt freicar point clouds ...')
	# load the gt point clouds
	gt_pcl_dir = '.../vpdr/freicars_sfm/'
	unqseq = ['037', '036', '042', '022', '034']
	gt_pcl_db = {}
	for seq in unqseq:
		fl = os.path.join(gt_pcl_dir, seq + '.pkl')
		with open(fl, 'rb') as f: 
			pcl_data = pickle.load(f)
		pcl_data = torch.FloatTensor(pcl_data)
		pcl_std = pcl_data.std(1).mean()
		gt_pcl_db[seq] = {
			'xyz': pcl_data,
			'scale_correction': 1. / pcl_std,
		}
	return gt_pcl_db

def load_freicar_data(imname, seq_name):
	data_root  = IMAGE_ROOTS['freicars_clickp_filtd'][0]
	depth_name = imname + '.jpg.half.jpg.filtdepth.tiff'
	depth_path = os.path.join(data_root, seq_name, \
		'undistort/stereo/filtered_depth_0.2', depth_name)
	assert 'filtdepth' in depth_path
	mask_name = imname + '.jpg.half.png'
	mask_path = os.path.join(data_root, seq_name, \
		'masks', mask_name)
	depth_gt = torch.FloatTensor(load_depth({'depth_path': depth_path}))
	mask = torch.FloatTensor(load_mask({'mask_path': mask_path}))
	return depth_gt, mask

def load_freicar_gt_pcl_clean(cache):
	print('loading clean gt freicar point clouds ...')
	# load the gt point clouds
	gt_pcl_dir = '.../vpdr/freicars_sfm/'
	unqseq = ['037', '036', '042', '022', '034']
	gt_pcl_db = {}
	for seq in tqdm(unqseq):
		ok = [ (1 if seq==s else 0) for s in cache['seq_name'] ]
		ok = np.where(np.array(ok))[0]
		if len(ok)==0:
			continue
		pcl_seq = []
		for idx in ok:
			orig_sz = cache['orig_image_size'][idx].long().tolist()
			imname = cache['depth_path'][idx].split('/')[-1].split('.')[0]
			depth_gt, mask = load_freicar_data(imname, seq)
			mask = Fu.interpolate(mask[None], size=orig_sz, mode='nearest')[0]
			depth_gt = Fu.interpolate(depth_gt[None], size=orig_sz, mode='nearest')[0]
			mask = mask * (depth_gt > 0.).float()
			ok = torch.nonzero(mask.view(-1)).squeeze()
			if len(ok)==0: continue
			K, R, T = cache['K_orig'][idx], cache['R'][idx], cache['T'][idx]
			pcl = depth2pcl(depth_gt[None], K[None], image_size=orig_sz, projection_type='perspective')[0]
			pcl = pcl.view(3, -1)[:, ok]
			pcl = R.t() @ (pcl - T[:,None])
			pcl_seq.append(pcl)

		pcl_seq = torch.cat(pcl_seq, dim=1)
		if pcl_seq.shape[1] > 30000:
			state = torch.get_rng_state()
			torch.manual_seed(0)
			prm = torch.randperm(pcl_seq.shape[1])[:30000]
			torch.set_rng_state(state)
			pcl_seq = pcl_seq[:, prm]

		pcl_std = pcl_seq.std(1).mean()
		gt_pcl_db[seq] = { 'xyz': pcl_seq, 'scale_correction': 1. / pcl_std }
	
	outdir = './data/vpdr/'
	os.makedirs(outdir, exist_ok=True)
	outfile = os.path.join(outdir, 'freicars_pcl_db_eval.pth')
	
	torch.save(gt_pcl_db, outfile)

	return gt_pcl_db

def load_p3d_meshes(cached_preds, n_sample=30000):
	mesh_db = {}
	root = IMAGE_ROOTS['pascal3d_clickp_all'][0]
	for mesh_path in cached_preds['mesh_path']:
		if mesh_path not in mesh_db:
			vertices, faces = load_off(os.path.join(root,mesh_path))
			if vertices is None:
				continue
			mesh = trimesh.Trimesh( \
						vertices=vertices.tolist(), \
						faces=faces.tolist() )
			pcl = trimesh.sample.sample_surface(mesh, n_sample)
			mesh_db[mesh_path] = torch.from_numpy(pcl[0].T).float()
	return mesh_db

def eval_p3d(cached_preds, eval_vars=None, visualize=False, \
			  dump_dir=None, skip_flip=False):

	nim = len(cached_preds['masks'])

	errs = []


	mesh_db = load_p3d_meshes(cached_preds)

	for imi in tqdm(range(nim)):

		gt_pcl = mesh_db[cached_preds['mesh_path'][imi]]
		gt_pcl_imcoord = (cached_preds['R'][imi] @ gt_pcl + cached_preds['T'][imi][:,None])
		
		# pcl prediction
		pcl_pred = cached_preds['embed_db_shape_camera_coord'][imi,:,:,0].clone()
		
		errs_this_im = {}
		pcl_out_this_im = {}

		for flip in (False, True):

			gt_pcl_test = gt_pcl_imcoord.clone()
			
			if skip_flip and flip:
				pass # use the previous result
			else:
				if flip: gt_pcl_test[2,:] *= -1.

				errs_now_pcl = eval_full_pcl( \
								pcl_pred[None].clone(),
								gt_pcl_test[None].clone() )
			
				pcl_full_err = float(errs_now_pcl['pcl_error'])
				pcl_full_err_align = float(errs_now_pcl['pcl_error_align'])

				errs_now = \
					{ 'EVAL_pcl_scl_detkp':      pcl_full_err,
					  'EVAL_pcl_corr_scl_detkp': pcl_full_err_align }
			
			errs_this_im[flip] = errs_now
			pcl_out_this_im[flip] = errs_now_pcl

		decvar = 'EVAL_pcl_corr_scl_detkp' # decide whether we flip based on this
		
		flip_better = errs_this_im[True][decvar] < errs_this_im[False][decvar]

		# take the better one in case of flipping
		pcl_out_this_im = pcl_out_this_im[flip_better]
		errs_this_im = errs_this_im[flip_better]
	
		if False:
			from tools.vis_utils import get_visdom_connection, \
										visdom_plotly_pointclouds
			viz = get_visdom_connection()
			from PIL import Image
			im = Image.open(cached_preds['image_path'][imi]).convert('RGB')
			im = torch.FloatTensor(np.array(im)).permute(2,0,1)
			viz.image(im, env='pcl_debug', win='im')		
			pcl_gt         = pcl_out_this_im['gt']
			pcl_pred       = pcl_out_this_im['pred']
			pcl_pred_orig  = pcl_out_this_im['pred_orig']
			pcl_pred_align = pcl_out_this_im['pred_align']
			for imii in (0,):
				show = {
					'gt':         pcl_gt[imii].view(3, -1),
					# 'pred':       pcl_pred[imii].view(3, -1),
					'pred_orig':  pcl_pred_orig[imii].view(3, -1),
					'pred_align': pcl_pred_align[imii].view(3, -1),
				}
				visdom_plotly_pointclouds( \
						viz,
						show,
						'pcl_debug',
						title='pcl_debug',
						win='pcl_debug',
						markersize=2,
						height=800,
						width=800,
					)
			import pdb; pdb.set_trace()

		errs.append(errs_this_im)

	results = {}
	for med in (False, True): # dont show the median
		for k in errs[0]:
			res = torch.FloatTensor([float(err[k]) for err in errs])
			res = float(res.median()) if med else float(res.mean())
			results[(k+'_med') if med else k] = res

	print('P3D evaluation results:')
	for k, v in results.items():
		print('%20s: %1.5f' % (k,v) )

	if eval_vars is not None:
		for eval_var in eval_vars:
			assert eval_var in results, \
				'evaluation variable missing! (%s)' % eval_var
		print('eval vars check ok!')

	# if TGT_NIMS==None:
	# 	results = { k+'_DBG':v for k, v in results.items() }	
		

	return results, None

def eval_freicars(
		cached_preds, eval_vars=None, visualize=True, 
		TGT_NIMS=1427, dump_dir=None
	):

	from dataset.dataset_configs import FREIBURG_VAL_IMAGES

	cache_path = './cache/vpdr/freicars_pcl_db_eval.pth'
	if not os.path.isfile(cache_path):
		gt_pcl_db = load_freicar_gt_pcl_clean(cached_preds)
	else:
		gt_pcl_db = torch.load(cache_path)

	nim = len(cached_preds['depth_path'])
	
	if TGT_NIMS is None:
		print('\n\n\n!!!! DEBUG MODE !!!!\n\n\n')

	errs = []

	for imi in tqdm(range(nim)):
		seq_name = cached_preds['seq_name'][imi]
		gt_pcl = gt_pcl_db[seq_name]['xyz']
		gt_pcl_imcoord = (cached_preds['R'][imi] @ gt_pcl + \
						  cached_preds['T'][imi][:,None])
		scale_correction = gt_pcl_db[seq_name]['scale_correction']
		orig_sz = cached_preds[
			'orig_image_size'][imi].type(torch.int32).tolist()

		imname = cached_preds['depth_path'][imi].split('/')[-1].split('.')[0]
		depth_gt, mask = load_freicar_data(imname, seq_name)
		depth_gt = Fu.interpolate(depth_gt[None], size=orig_sz, mode='nearest' )[0]
		mask = Fu.interpolate(mask[None], size=orig_sz, mode='nearest')[0]

		# check we have a correct size
		for s, s_ in zip(orig_sz, depth_gt.shape[1:]): assert s==s_

		depth_pred = cached_preds['depth_dense'][imi].clone()
		minscale = min(depth_pred.shape[i] / orig_sz[i-1] for i in [1, 2])

		newsz = np.ceil(np.array(depth_pred.shape[1:])/minscale).astype(int).tolist()
		depth_pred_up = Fu.interpolate( \
			depth_pred[None], \
			size=newsz, \
			mode='bilinear' )[0]
		depth_pred_up = depth_pred_up[:,:depth_gt.shape[1],:depth_gt.shape[2]]
		depth_pred_up /= minscale

		K = cached_preds['K_orig'][imi:imi+1].clone()

		errs_this_im = {}
		for pred_projection_type in ( 'perspective', 'orthographic'):
			errs_now = eval_depth_pcl(depth_pred_up[None].clone(),
									  depth_gt[None].clone(), 
									  K=K.clone(),
									  pred_projection_type=pred_projection_type,
									  gt_projection_type='perspective',
									  masks=mask[None],
									  lap_thr=0.01)
			pcl_err_corrected =  scale_correction * float(errs_now['dist_pcl'])
		
			errs_this_im.update( \
				{ 'EVAL_depth_scl_'+pred_projection_type: pcl_err_corrected} )

		if True:

			pcl_pred = cached_preds['embed_db_shape_camera_coord'][imi,:,:,0].clone()
			pcl_pred /= minscale # !!!!

			errs_now_pcl = eval_full_pcl( \
										pcl_pred[None].clone(),
										gt_pcl_imcoord[None].clone(), 
										K=K.clone(),
										scale_best=errs_now['scale_best'], )
			pcl_full_err_corrected = \
				scale_correction * float(errs_now_pcl['pcl_error'])
			pcl_full_err_align_corrected = \
				scale_correction * float(errs_now_pcl['pcl_error_align'])

			for pred_projection_type in ('perspective', 'orthographic'):
				errs_this_im.update( \
					{ 'EVAL_pcl_scl_'+pred_projection_type: \
						pcl_full_err_corrected,
					  'EVAL_pcl_corr_scl_'+pred_projection_type: \
						pcl_full_err_align_corrected} )

		errs.append(errs_this_im)

	results = {}
	for med in (True, False):
		for k in errs[0]:
			res = torch.FloatTensor([float(err[k]) for err in errs])
			res = float(res.median()) if med else float(res.mean())
			results[(k+'_med') if med else k] = res

	if True: # eval sparse kps
		gt_kp_loc_3d = cached_preds['kp_loc_3d']
		pred_kp_loc_3d = cached_preds['shape_image_coord_cal']
		nrsfm_kp_loc_3d = cached_preds['nrsfm_shape_image_coord']
		scale_corrs = torch.stack([
			gt_pcl_db[cached_preds['seq_name'][imi]]['scale_correction'] 
			for imi in range(nim)
		])
		results['EVAL_sparse_pcl'] = float(eval_sparse_pcl(
			pred_kp_loc_3d, gt_kp_loc_3d, scale_corrs))
		results['EVAL_sparse_pcl_nrsfm'] = float(eval_sparse_pcl(
			nrsfm_kp_loc_3d, gt_kp_loc_3d, scale_corrs))

	print('Freiburg Cars evaluation results:')
	for k, v in results.items():
		print('%20s: %1.5f' % (k,v) )

	if eval_vars is not None:
		for eval_var in eval_vars:
			assert eval_var in results, \
				'evaluation variable missing! (%s)' % eval_var
		print('eval vars check ok!')

	if TGT_NIMS==None:
		results = { k+'_DBG':v for k, v in results.items() }	
		

	return results, None



def load_off(obj_path):

	if not os.path.isfile(obj_path):
		print('%s does not exist!' % obj_path)
		return None, None

	with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
		lines = f.readlines()
	lines = [ l.strip() for l in lines ]

	nv, nf, _ = [int(x) for x in lines[1].split(' ')]

	entries = lines[2:]

	for vertface in ('v', 'f'):

		if vertface=='v':
			vertices = [ [float(v_) for v_ in v.split(' ')] for v in entries[:nv]]
			vertices = torch.FloatTensor(vertices).float()
			entries = entries[nv:]

		elif vertface=='f':
			faces = [ [int(v_) for v_ in v.split(' ')[1:]] for v in entries]
			faces = torch.LongTensor(faces)
			assert faces.shape[0]==nf
		else:
			raise ValueError()

	return vertices, faces



def load_ff_obj(obj_path):
	with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
		lines = f.readlines()

	lines = [ l.strip() for l in lines ]

	for vertface in ('v', 'f'):

		entries = [ [ v for v in l.split(' ')[1:4] ] \
						for l in lines if l.split(' ')[0]==vertface ]
		
		if vertface=='v':
			entries = [ [float(v_) for v_ in v] for v in entries ]
			entries = torch.FloatTensor(entries)
		elif vertface=='f':
			entries = [ [ int(v_.split('/')[0]) for v_ in v ] \
							for v in entries ]
			entries = torch.LongTensor(entries)
		else:
			raise ValueError()

		if vertface=='v':
			vertices = entries.float()
		else:
			faces = (entries-1).long()

	return vertices, faces



def eval_florence(cached_preds, eval_vars=None, TGT_NIMS=1427, visualize=False):
	from tools.pcl_unproject import depth2pcl
	from tools.eval_functions import eval_pcl_icp

	root = IMAGE_ROOTS['celeba_ff'][1]
	
	nim = len(cached_preds['mesh_path'])

	errs = []

	for imi in tqdm(range(nim)):

		# if imi <= 775:
		# 	continue

		# get the ff mesh
		mesh_path = cached_preds['mesh_path'][imi]
		if len(mesh_path)==0: continue
		mesh_path = os.path.join(root, mesh_path)
		vertices, faces = load_ff_obj(mesh_path)
		mesh_gt = trimesh.Trimesh(
			vertices=vertices.tolist(),
			faces=faces.tolist()
		)

		# get our prediction
		kp_loc     = cached_preds['kp_loc'][imi]
		# image_size = list(cached_preds['images'][imi].shape[1:])
		mask       = cached_preds['masks'][imi]

		if mask.sum()<=1:
			print('Empty mask!!!')	
			continue

		image_size = list(mask.shape[1:])
		# mask       = Fu.interpolate(mask[None], size=image_size)[0]
		pcl_pred   = cached_preds['shape_image_coord_best_scale'][imi]
		pcl_pred   = Fu.interpolate(pcl_pred[None], size=image_size)[0]

		err_now = {}
		for flip in (True, False):
			pcl_pred_now = pcl_pred.clone()
			if flip: pcl_pred_now[2,:] = -pcl_pred_now[2,:]
			# compute icp error
			err = eval_pcl_icp(pcl_pred_now, mesh_gt, mask, kp_loc)
			err = {
				'EVAL_pcl_scl_recut_orthographic': err['dist_pcl_scl_recut'],
				'EVAL_pcl_scl_orthographic':       err['dist_pcl_scl'],
				'EVAL_pcl_orthographic':           err['dist_pcl'],
			}
			if flip: err = {k+'_flip':v for k, v in err.items()}
			err_now.update(err)
			
		errs.append(err_now)

		print('<EVAL_STATE>')
		print(f'IMAGE={imi}')
		print(err_now)
		print('<\EVAL_STATE>')

	results = {}
	for med in (True, False):
		for k in errs[0]:
			res = torch.FloatTensor([float(err[k]) for err in errs])
			if med:
				res = float(res.median())
			else:
				res = float(res.mean())
			results[(k+'_med') if med else k] = res
				
	print('Florence Face evaluation results:')
	for k, v in results.items():
		print('%20s: %1.5f' % (k,v) )

	if eval_vars is not None:
		for eval_var in eval_vars:
			assert eval_var in results, \
				'evaluation variable missing! (%s)' % str(eval_var)
		print('eval vars check ok!')

	return results, None






