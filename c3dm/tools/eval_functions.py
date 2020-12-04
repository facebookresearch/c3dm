import time
import torch
import torch.nn.functional as Fu
import numpy as np
import collections
from tools.functions import safe_sqrt
from tools.pcl_unproject import depth2pcl


def in_hull(p, hull, extendy=False):
	"""
	Test if points in `p` are in `hull`

	`p` should be a `NxK` coordinates of `N` points in `K` dimensions
	`hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
	coordinates of `M` points in `K`dimensions for which Delaunay triangulation
	will be computed
	"""
	from scipy.spatial import Delaunay
	if not isinstance(hull, Delaunay):
		hull = Delaunay(hull, incremental=True)
		if extendy:
			pts = hull.points
			minx = np.min(pts[:,0])
			maxx = np.max(pts[:,0])
			new_pts = [[minx, 0], [maxx, 0]]
			hull.add_points(new_pts)

	return hull.find_simplex(p)>=0


def get_ff_head_mask(pcl_pred, kp_loc):
	axx = np.arange( pcl_pred.shape[2] ) + 0.5
	axy = np.arange( pcl_pred.shape[1] ) + 0.5
	all_pt = np.stack(np.meshgrid(axx, axy))
	all_pt = all_pt.reshape(2, -1)
	kpmask = in_hull(all_pt.T, kp_loc.t().numpy())
	kpmask = kpmask.reshape( list(pcl_pred.shape[1:]) ).astype(float)
	return torch.tensor(kpmask).float()
	

def cut_ff_head(pcl_pred, kp_loc, mask):
	
	if True:
		axx = np.arange( pcl_pred.shape[2] ) + 0.5
		axy = np.arange( pcl_pred.shape[1] ) + 0.5
		all_pt = np.stack(np.meshgrid(axx, axy))
		all_pt = all_pt.reshape(2, -1)
		kpmask = in_hull(all_pt.T, kp_loc.t().numpy())
		# kpmask = kpmask.reshape( list(pcl_pred.shape[1:]) ).astype(float)
		ok = np.where(kpmask.reshape(-1))[0].tolist()
	else:
		chin_pt = kp_loc[:, 16].long()
		nose_pt = kp_loc[:, 54].long()
		chin_3d_pt = pcl_pred[:, chin_pt[1], chin_pt[0]]
		nose_3d_pt = pcl_pred[:, nose_pt[1], nose_pt[0]]
		thr = ((nose_3d_pt - chin_3d_pt)**2).sum().sqrt()
		thr *= 1.01
		df = ((pcl_pred - nose_3d_pt[:,None,None])**2).sum(0, keepdim=True).sqrt()
		df = df * mask + (1-mask) * thr * 1000.
		ok = torch.nonzero(df.view(-1) <= thr).squeeze()
		
	# if True:
	# 	npix = pcl_pred[0].numel()
	# 	nok = np.setdiff1d(np.arange(npix), ok)
	# 	pcl_pred_nok = pcl_pred.view(3,-1)[:, nok].numpy()
	# 	pcl_pred_raw = pcl_pred.view(3,-1).numpy()
	# 	pcl_pred_ok  = pcl_pred.view(3,-1)[:, ok].numpy()
	# 	from tools.vis_utils import get_visdom_connection, \
	# 						        visdom_plotly_pointclouds
	# 	viz = get_visdom_connection()
	# 	visdom_plotly_pointclouds( \
	# 				viz, 
	# 				{ 'pred':     pcl_pred_ok,
	# 				  'pred_nok': pcl_pred_nok,
	# 				  'pred_raw': pcl_pred_raw, }, 
	# 				'ff_debug', 
	# 				title='ff_debug', win='ff_debug_', 
	# 				markersize=2,
	# 				in_subplots=True,
	# 			)
	# 	import pdb; pdb.set_trace()

	pcl_pred = pcl_pred.view(3,-1)[:, ok]

	pcl_pred = apply_pcl_pred_transform(pcl_pred)

	return pcl_pred

def apply_pcl_pred_transform(pcl_pred):
	a = np.pi + np.pi/2. # original
	Rx = [
		[  1.,         0.,  	   0. ],
		[  0.,  np.cos(a), -np.sin(a) ],
		[  0.,  np.sin(a),  np.cos(a) ],
	]
	pcl_pred = torch.FloatTensor(Rx) @ pcl_pred
	return pcl_pred

def get_nose_loc(pcl_gt):
	nose_id = np.argmin(pcl_gt[1,:])
	nose_loc = pcl_gt[:, nose_id:(nose_id+1)]
	return nose_loc

def cut_nose(pcl_gt, thr=100., nose_loc=None):
	if nose_loc is None:
		nose_loc = get_nose_loc(pcl_gt)
	df = pcl_gt - nose_loc
	dst = np.sqrt((df*df).sum(0))
	ok = np.where(dst <= thr)[0]
	pcl_gt = pcl_gt[:, ok]
	return pcl_gt

def cut_ff_nose(pcl_gt, do_rotate=True):
	# 2) 45 deg along x
	# a = np.pi / 4. # original
	a = np.pi / 4. + np.pi / 10.
	Rx = [
		[  1.,         0.,  	   0. ],
		[  0.,  np.cos(a), -np.sin(a) ],
		[  0.,  np.sin(a),  np.cos(a) ],
	]
	if do_rotate:
		pcl_gt = Rx @ pcl_gt
	pcl_gt = cut_nose(pcl_gt)
	return pcl_gt

def re_cut_ff_nose(matrix_scl, pcl_pred, kp_loc, trans_scl, mask, mu, scl):

	ok = torch.nonzero(mask.view(-1) > 0.).squeeze()

	# cut off the hull
	if True:
		axx = np.arange( pcl_pred.shape[2] ) + 0.5
		axy = np.arange( pcl_pred.shape[1] ) + 0.5
		all_pt = np.stack(np.meshgrid(axx, axy))
		all_pt = all_pt.reshape(2, -1)
		kpmask = in_hull(all_pt.T, kp_loc.t().numpy(), extendy=True)
		# kpmask = kpmask.reshape( list(pcl_pred.shape[1:]) ).astype(float)
		okkp = np.where(kpmask.reshape(-1))[0]
		ok = np.intersect1d( okkp, ok.numpy() ).tolist()

	if len(ok)==0:
		print('WARNING: RE-CUT results in empty face!')
		return None

	pcl_pred_ok = pcl_pred.view(3, -1)[:, ok]
	pcl_pred_ok = apply_pcl_pred_transform(pcl_pred_ok)
	pcl_pred_ok -= torch.FloatTensor(mu)
	pcl_pred_ok *= scl
	
	R = torch.FloatTensor(matrix_scl[:3,:3])
	T = torch.FloatTensor(matrix_scl[:3,3:4])

	pcl_pred_ok_t_t = R @ pcl_pred_ok + T

	nose_loc = torch.FloatTensor(get_nose_loc(trans_scl.T))
	
	pcl_pred_recut = cut_nose(pcl_pred_ok_t_t, nose_loc=nose_loc)
	pcl_pred_recut = pcl_pred_recut.numpy()

	return pcl_pred_recut


def eval_pcl_icp(pcl_pred, mesh_gt, mask, kp_loc):
	import trimesh
	
	from tools.utils import Timer

	profile = True # actually this is inverted

	with Timer(quiet=profile):
		# sample points from the surface
		pcl_gt_orig = trimesh.sample.sample_surface(mesh_gt, 60000)[0]

		# cut stuff
		pcl_gt_cut   = cut_ff_nose(pcl_gt_orig.T)
		pcl_pred_cut = cut_ff_head(pcl_pred, kp_loc, mask).numpy()

		# center
		pred_cut_mean = pcl_pred_cut.mean(1)[:, None]
		pcl_pred_cut  = pcl_pred_cut - pred_cut_mean
		pcl_gt_cut    = pcl_gt_cut - pcl_gt_cut.mean(1)[:, None]

		# align stds
		pred_std     = pcl_pred_cut.std(1).mean()
		gt_std       = pcl_gt_cut.std(1).mean()
		pcl_pred_cut = pcl_pred_cut * (gt_std / pred_std)
		
		# matrix, transformed, _ = \
		# 	trimesh.registration.icp( \
		# 		pcl_pred_cut.T, pcl_gt_cut.T, \
		# 		initial=np.identity(4), threshold=1e-5, \
		# 		max_iterations=50, **{'scale': True})

	with Timer(quiet=profile):
		matrix_scl, transformed_scl, _ = \
			trimesh.registration.icp( \
				pcl_pred_cut.T, pcl_gt_cut.T, \
				initial=np.identity(4), threshold=1e-5, \
				max_iterations=30, **{'scale': False})

	with Timer(quiet=profile):
		pcl_pred_recut = re_cut_ff_nose( matrix_scl, pcl_pred, kp_loc, 
										transformed_scl, mask,
										pred_cut_mean, 
										gt_std / pred_std )

	if pcl_pred_recut is None or pcl_pred_recut.size==0:
		print('WARNING: RE-CUT results in empty face!')
		pcl_pred_recut = pcl_pred_cut

	with Timer(quiet=profile):
		matrix_scl_recut, transformed_scl_recut, _ = \
			trimesh.registration.icp( \
				pcl_pred_recut.T, pcl_gt_cut.T, \
				initial=np.identity(4), threshold=1e-5, \
				max_iterations=30, **{'scale': False})

	# if True:
	# 	from tools.vis_utils import get_visdom_connection, \
	# 								visdom_plotly_pointclouds
	# 	viz = get_visdom_connection()
	# 	visdom_plotly_pointclouds( \
	# 			viz,
	# 			{
	# 				'pred': pcl_pred_cut, 
	# 				'pred_align': transformed_scl.T,
	# 				# 'pred_align_scl': transformed.T,
	# 				'pcl_gt': pcl_gt_cut, 
	# 				'pred_recut': pcl_pred_recut,
	# 				'pred_align_recut': transformed_scl_recut.T
	# 			}, 
	# 			'ff_debug', 
	# 			title='ff_debug', 
	# 			win='ff_debug_align', 
	# 			markersize=2, 
	# 			in_subplots=False, 
	# 			height=600, 
	# 			width=600
	# 		)
	# 	time.sleep(1)
	# 	import pdb; pdb.set_trace()

	# pcl distance
	ft = lambda x: torch.FloatTensor(x).t().cuda()
	fl  = lambda x: torch.FloatTensor(x).cuda()

	with Timer(quiet=profile):
		# err           = chamfer(ft(transformed), fl(pcl_gt_cut))
		err_scl       = float(chamfer(ft(transformed_scl), fl(pcl_gt_cut)).detach())
		err_scl_recut = float(chamfer(ft(transformed_scl_recut), fl(pcl_gt_cut)).detach())

	res = collections.OrderedDict( [
		('dist_pcl', err_scl),
		('dist_pcl_scl', err_scl),
		('dist_pcl_scl_recut', err_scl_recut),
		# ('pred_t', ft(transformed)),
		('pred_t_scl', ft(transformed_scl)),
		('gt', fl(pcl_gt_cut)),
	] )

	return res


def eval_depth( pred, gt, crop=5, masks=None, 
				get_best_scale=False):
	
	# chuck out border
	gt   = gt  [ :, :, crop:-crop, crop:-crop ]
	pred = pred[ :, :, crop:-crop, crop:-crop ]

	if masks is not None:
		# mult gt by mask
		masks = masks[:,:,crop:-crop,crop:-crop]
		gt = gt * (masks > 0.).float()

	dmask      = (gt > 0.).float()
	dmask_mass = torch.clamp(dmask.sum((1,2,3)),1e-4)

	if get_best_scale:
		# mult preds by a scalar "scale_best" 
		# 	s.t. we get best possible mse error
		xy = pred * gt ; xx = pred * pred
		if masks is not None:
			xy *= masks ; xx *= masks
		scale_best = xy.mean((1,2,3)) / torch.clamp(xx.mean((1,2,3)), 1e-4)
		pred = pred * scale_best[:, None, None, None]

	df   = gt - pred

	mse_depth = (dmask*(df**2)).sum((1,2,3))  / dmask_mass
	abs_depth = (dmask*df.abs()).sum((1,2,3)) / dmask_mass
		
	res = collections.OrderedDict( [
		('mse_depth', mse_depth),
		('abs_depth', abs_depth),
		] )

	# as in https://arxiv.org/pdf/1606.00373.pdf
	for thr_exp in (1.,2.,3.):
		delta     = (1.25**thr_exp) / 100. # to meters
		lessdelta = (dmask*(df.abs()<=delta).float()).sum((1,2,3)) \
					/ dmask_mass
		res[ 'delta_%d'%int(thr_exp) ] = lessdelta.cpu()

	# delta error for linspaced thresholds
	for delta in np.linspace(0.,2.,21):
		if delta <= 0.: continue
		lessdelta = (dmask*(df.abs()<=delta).float()).sum((1,2,3)) \
					/ dmask_mass
		res[ 'delta_%03d'%int(100*delta) ] = lessdelta.cpu()	

	if get_best_scale:
		res['scale_best'] = scale_best

	return res

def set_mean_depth_to_0(x,mask=None):
	
	x = x.copy()
	if mask is not None:
		x = x * mask[:,None,:]
		mu_depth = (x.sum(2)/mask.sum(1)[:,None])[:,2]
	else:
		mu_depth = x.mean(2)[:,2]

	x[:,2,:] = x[:,2,:] - mu_depth[:,None]

	if mask is not None:
		x = x * mask[:,None,:]

	return x

def get_edm(pts,pts2=None):

	dtype = pts.data.type()
	ba, dim, N = pts.shape

	if pts2 is not None:
		edm  = torch.bmm(-2. * pts2.transpose(1,2), pts)
		fNorm1 = (pts*pts).sum(1,keepdim=True)
		fNorm2 = (pts2*pts2).sum(1,keepdim=True)
		edm += fNorm2.transpose(1,2)  # inplace saves memory
		edm += fNorm1
		# edm    = (fNorm2.transpose(1,2) + fGram) + fNorm1 
	else:
		fGram  = torch.bmm(2 * pts.transpose(1,2), pts)
		fNorm1 = (pts*pts).sum(1,keepdim=True)
		edm    = (fNorm1.transpose(1,2) - fGram) + fNorm1 
	
	return edm.contiguous()


def chamfer(a, b, med=False):
	return 0.5 * (nn_err(a, b, med=med) + nn_err(b, a, med=med))

def nn_err(a, b, med=False):
	D = get_edm(a[None].detach(), b[None].detach())
	minvals, minidx = D.min(dim=1)
	minvals = torch.clamp(minvals,0.).squeeze().sqrt()
	if med:
		assert False
		errs = minvals.median()
	else:
		errs = minvals.mean()
	
	# if True:
	# 	from pykeops.torch import LazyTensor
	# 	a = a.t().contiguous()
	# 	b = b.t().contiguous()
	# 	A = LazyTensor(a[:, None, :])  # (M, 1, 3)
	# 	B = LazyTensor(b[None, :, :])  # (1, N, 3)
	# 	D = ((A - B) ** 2).sum(2)  # (M, N) symbolic matrix of squared distances
	# 	indKNN = D.argKmin(1, dim=1).squeeze()  # Grid <-> Samples, (M**2, K) integer tensor
	# 	errs_ = ((a - b[indKNN,:])**2).sum(1).sqrt()

	# if True:
	# 	nns = b[indKNN,:]
	# 	from tools.vis_utils import get_visdom_connection, \
	# 								visdom_plotly_pointclouds
	# 	viz = get_visdom_connection()
	# 	show = {
	# 		'in': a.t().contiguous().view(3,-1),
	# 		'nns': nns.t().contiguous().view(3,-1),
	# 	}
	# 	visdom_plotly_pointclouds( \
	# 			viz,
	# 			show,
	# 			'pcl_debug',
	# 			title='pcl_debug',
	# 			win='pcl_debug_nns',
	# 			markersize=2,
	# 		)
	# 	import pdb; pdb.set_trace()

	return errs

# def get_best_scale_cov(pcl_pred, pcl_gt):
# 	# compute the pcl centers
# 	pred_cnt, gt_cnt = [ \
# 		p.mean(2, keepdim=True) for p in (pcl_pred, pcl_gt) ]
# 	# center
# 	c_pred, c_gt = [ \
# 		p - c for p, c in zip((pcl_pred, pcl_gt), (pred_cnt, gt_cnt)) ]
	
# 	cov_pred, cov_gt = [torch.bmm(c, c.permute(0,2,1)) * (1. / c.shape[2]) for c in [c_pred, c_gt]]
	
# 	import pdb; pdb.set_trace()

# 	det_pred = torch.stack([torch.det(c) for c in cov_pred])
# 	det_gt   = torch.stack([torch.det(c) for c in cov_gt])

# 	# eigs_pred = torch.stack([torch.eig(c)[0][:,0] for c in cov_pred])
# 	# eigs_gt = torch.stack([torch.eig(c)[0][:,0] for c in cov_gt])

# 	import pdb; pdb.set_trace()

def eval_full_pcl(pcl_pred, 
				  pcl_gt, 
				  K=None,
				  scale_best=None):
				#   faces=None):

	import trimesh

	# batch size
	ba = pcl_pred.shape[0]

	# compute the pcl centers
	pred_cnt, gt_cnt = [ \
		p.mean(2, keepdim=True) for p in (pcl_pred, pcl_gt) ]

	# center
	c_pred, c_gt = [ \
		p - c for p, c in zip((pcl_pred, pcl_gt), (pred_cnt, gt_cnt)) ]

	if False:
		# apply the best scale
		c_pred = c_pred * scale_best[:, None, None]
	else:
		# recompute the best scale
		# scale_best = get_best_scale_cov(pcl_pred, pcl_gt)
		scale_best = (c_gt.std(2) / c_pred.std(2)).mean(1)
		if not np.isfinite(scale_best): 
			scale_best = scale_best.new_ones([1])
		c_pred = c_pred * scale_best[:, None, None]

	e = []
	c_pred_align = []
	for ip in range(ba):
		_, transformed, _ = \
			trimesh.registration.icp( \
				c_pred[ip].numpy().T, c_gt[ip].numpy().T, \
				initial=np.identity(4), threshold=1e-10, \
				max_iterations=30, **{'scale': False})
		c_pred_align.append(torch.FloatTensor(transformed.T))
		e_    = chamfer(c_gt[ip].float().cuda(), c_pred[ip].float().cuda())
		e_al_ = chamfer(c_gt[ip].float().cuda(), c_pred_align[ip].float().cuda())
		e.append([e_, e_al_])
	c_pred_align = torch.stack(c_pred_align)
	e = torch.FloatTensor(e)

	res = collections.OrderedDict( [
			('pcl_error',       e[:, 0]),
			('pcl_error_align', e[:, 1]),
			('scale_best',      scale_best),
			('pred_align',      c_pred_align),
			('pred_orig',       pcl_pred),
			('pred',            c_pred),
			('gt',              c_gt),
		] )

	return res

def eval_sparse_pcl(pred, gt, rescale_factor):
	# get best scale
	xy = pred * gt ; xx = pred * pred
	scale_best = xy.mean((1, 2)) / xx.mean((1, 2)).clamp(1e-4)
	pred_scl = pred * scale_best[:, None, None]
	err = ((pred_scl-gt)**2).sum(1).sqrt().mean(1)
	err_resc = err * rescale_factor
	return err_resc.mean()

def eval_depth_pcl( pred, gt, K=None, masks=None,
					gt_projection_type='perspective',
					pred_projection_type='orthographic',
					debug=False,
					lap_thr=0.3,
					):

	ba = gt.shape[0]

	if masks is not None:
		# mult gt by mask
		gt = gt * (masks > 0.).float()

	gt = depth_flat_filter(gt, size=5, thr=lap_thr)

	dmask      = (gt > 0.).float()
	dmask_mass = torch.clamp(dmask.sum((1,2,3)), 1e-4)

	# convert to point clouds
	pcl_pred = depth2pcl(pred, K, projection_type=pred_projection_type)
	pcl_gt   = depth2pcl(gt, K, projection_type=gt_projection_type)

	if gt_projection_type==pred_projection_type and \
		gt_projection_type=='perspective' and False:

		# estimate the best scale
		xy = pred * gt ; xx = pred * pred
		xy *= dmask ; xx *= dmask
		scale_best = xy.mean((1,2,3)) / torch.clamp(xx.mean((1,2,3)), 1e-12)

		pred = pred * scale_best[:, None, None, None]

		# convert to point clouds
		c_pred = depth2pcl(pred, K, projection_type=pred_projection_type)
		c_gt   = depth2pcl(gt, K, projection_type=gt_projection_type)

		# if debug:
		# 	import pdb; pdb.set_trace()
		# 	c_pred = c_pred * 3

	else:

		# debug visualisations
		# pcl_pred = pcl_pred * masks
		# from tools.vis_utils import get_visdom_connection, visdom_plot_pointclouds
		# pcl_show = pcl_pred[0].view(3,-1)[:,masks[0].view(-1)>0.]
		# viz = get_visdom_connection()
		# visdom_plot_pointclouds(viz, \
		# 		{'pcl_pred': pcl_show.cpu().detach().numpy()},
		# 		'pcl_debug',
		# 		'pcl_debug',
		# 		win='pcl_debug',
		# 	)
		# import pdb; pdb.set_trace()

		# mask the point clouds
		pcl_pred, pcl_gt = [p * dmask for p in (pcl_pred, pcl_gt)]

		# compute the pcl centers
		pred_cnt, gt_cnt = [ \
			p.sum((2,3), keepdim=True) / dmask_mass[:,None,None,None] \
				for p in (pcl_pred, pcl_gt) ]

		# center
		c_pred, c_gt = [ \
			p - c for p, c in zip((pcl_pred, pcl_gt), (pred_cnt, gt_cnt)) ]

		# mask the centered point clouds
		c_pred, c_gt = [p * dmask for p in (c_pred, c_gt)]

		# estimate the best scale
		xy = c_pred * c_gt ; xx = c_pred * c_pred
		xy *= dmask ; xx *= dmask
		scale_best = xy.mean((1,2,3)) / torch.clamp(xx.mean((1,2,3)), 1e-4)
		
		# apply the best scale
		c_pred = c_pred * scale_best[:, None, None, None]

		# translate the point clouds back to original meanxy
		# xy_mask = torch.FloatTensor([1.,1.,0.])[None,:,None,None].type_as(c_pred)
		# d_c_pred, d_c_gt = [ \
		# 	p.clone() + c * xy_mask  \
		# 		for p, c in zip((c_pred, c_gt), (pred_cnt, gt_cnt)) ]

	# compute the per-vertex distance
	df = c_gt - c_pred

	dist = torch.clamp(df**2, 0.).sum(1,keepdim=True).sqrt()
	dist = (dmask * dist).sum((1,2,3)) / dmask_mass

	# if float(dist) <= 1e-3:
	# 	import pdb; pdb.set_trace()

	res = collections.OrderedDict( [
			('dist_pcl', dist),
			('scale_best', scale_best),
			('pred', c_pred),
			('pred_orig', pcl_pred),
			('gt', c_gt),
			('dmask', dmask),
		] )

	return res

def depth_flat_filter(depth, size=5, thr=0.3):

	mask = (depth > 0.).float()

	fsz = size*2+1
	w = depth.new_ones( (2,1,fsz,fsz) ) / float(fsz*fsz)

	depthf = Fu.conv2d( \
		torch.cat((depth, mask), dim=1), \
		w, 
		padding=size, 
		groups=2)
	depthf = depthf[:,0:1,:,:] / torch.clamp(depthf[:,1:2,:,:], 1e-4)

	df = (depth - depthf).abs()

	mask_mass = torch.clamp(mask.sum((1,2,3), keepdim=True), 1e-4)
	dmean = (depth * mask) / mask_mass
	dvar  = (((depth - dmean) * mask) ** 2).sum((1,2,3), keepdim=True)
	dstd  = safe_sqrt(dvar / mask_mass)
	
	bad = (df > dstd * thr).float()

	return depth * (1-bad)

def eval_depth_scale_inv( 
					pred, 
					gt, 
					masks=None,
					lap_thr=0.3,
					):

	if masks is not None:
		# mult gt by mask
		gt = gt * (masks > 0.).float()

	gt = depth_flat_filter(gt, size=5, thr=lap_thr)

	dmask      = (gt > 0.).float()
	dmask_mass = torch.clamp(dmask.sum((1,2,3)), 1e-4)

	# estimate the best scale
	xy = pred * gt ; xx = pred * pred
	xy *= dmask ; xx *= dmask
	scale_best = xy.mean((1,2,3)) / torch.clamp(xx.mean((1,2,3)), 1e-12)

	pred = pred * scale_best[:, None, None, None]

	df = pred - gt
	err = (dmask * df.abs()).sum((1,2,3)) / dmask_mass

	return err
