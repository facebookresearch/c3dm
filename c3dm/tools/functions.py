import torch
import torch.nn.functional as Fu
import numpy as np
import collections
import warnings

def clamp_depth(X, min_depth):
	xy, depth = X[:,0:2], X[:,2:]
	depth = torch.clamp(depth, min_depth)
	return torch.cat((xy,depth), dim=1)

def calc_ray_projection(X, Y, K=None, min_r_len=None, min_depth=None):
	n = X.shape[2]
	ba = X.shape[0]
	append1 = lambda x: \
		torch.cat((x,x.new_ones(x.shape[0],1,x.shape[2])), dim=1)
	if K is None:
		# Y is already calibrated
		r = append1(Y)
	else:
		r = torch.bmm(torch.inverse(K), append1(Y))
	r = Fu.normalize(r, dim=1)
	
	if min_depth is not None:
		X = clamp_depth(X, min_depth)

	r_len = (X * r).sum(1, keepdim=True)
	
	if min_r_len is not None:
		r_len = torch.clamp(r_len, min_r_len)
	
	r_proj = r_len * r

	return r_proj


def minimise_2d_residual_over_T(K, X, Y, v=None):
	ba, _, n = X.size()
	append1 = lambda x: torch.cat((x, x.new_ones(x[:,:1,:].size())), dim=1)
	Y_cam = torch.bmm(torch.inverse(K), append1(Y))

	# construct a system AT = b
	A_u = torch.cat((Y_cam.new_ones(ba, n, 1), Y_cam.new_zeros(ba, n, 1), -Y_cam[:,:1,:].permute(0,2,1)), dim=2)
	A_v = torch.cat((Y_cam.new_zeros(ba, n, 1), Y_cam.new_ones(ba, n, 1), -Y_cam[:,1:2,:].permute(0,2,1)), dim=2)
	b_u = (Y_cam[:,0:1,:] * X[:,2:,:] - X[:,0:1,:]).permute(0,2,1)
	b_v = (Y_cam[:,1:2,:] * X[:,2:,:] - X[:,1:2,:]).permute(0,2,1)

	res = Y_cam.new_empty(ba, 3)

	for i in range(ba):
		if v is not None:
			A = torch.cat((A_u[i, v[i] > 0., :], A_v[i, v[i] > 0., :]), dim=0)
			b = torch.cat((b_u[i, v[i] > 0., :], b_v[i, v[i] > 0., :]), dim=0)
		else:
			A = torch.cat((A_u[i, :, :], A_v[i, :, :]), dim=0)
			b = torch.cat((b_u[i, :, :], b_v[i, :, :]), dim=0)
		#res[i,:] = torch.lstsq(b, A)[0][:3, 0]
		res[i,:] = torch.matmul(torch.pinverse(A), b)[:, 0]

	return res

# TODO: if used, extract to test
def test_minimise_2d_residual_over_T():
	K = torch.eye(3)[None,:,:]
	X = torch.cat((Y, Y.new_ones(1,1,4)), dim=1)
	Y = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).t()[None,:,:]

	res = minimise_2d_residual_over_T(K, X, Y)
	assert torch.allclose(res, torch.tensor([[0., 0., 0.]]))

	X = torch.cat((Y, 2*Y.new_ones(1,1,4)), dim=1)
	assert torch.allclose(res, torch.tensor([[0., 0., -1.]]))

	X = torch.cat((Y, Y.new_ones(1,1,4)), dim=1)
	Y[:,0,:] += 3
	assert torch.allclose(res, torch.tensor([[3., 0., 0.]]))

def find_camera_T(K, X, Y, v=None, eps=1e-4):
	"""
	estimate camera translation given 3D-2D correspondences and cal matrix
	"""

	n = X.shape[2]
	ba = X.shape[0]

	append1 = lambda x: \
		torch.cat((x,x.new_ones(x.shape[0],1,x.shape[2])), dim=1)

	# projection rays
	r = torch.bmm(torch.inverse(K), append1(Y))
	r = Fu.normalize(r, dim=1)

	# outer projection ray product (need to permute the array first)
	rr = r.permute(0,2,1).contiguous().view(n*ba, 3)
	rr = torch.bmm(rr[:,:,None], rr[:,None,:])

	# I - rr
	Irr = torch.eye(3)[None].type_as(X).repeat(ba*n,1,1) - rr

	# [rr - I] x
	rrIx = torch.bmm(-Irr, X.permute(0,2,1).contiguous().view(n*ba, 3, 1))

	Irr  = Irr.view(ba,-1,3,3)
	rrIx = rrIx.view(ba,-1,3)

	if v is not None:
		Irr = Irr * v[:,:,None,None]
		rrIx = rrIx * v[:,:,None]

	Irr_sum = Irr.sum(1)
	rrIx_sum = rrIx.sum(1)

	if v is not None:
		ok = v.sum(1) > 2 # at least three visible
		rrI_sum_i = Irr_sum * 0.
		rrI_sum_i[ok] = torch.inverse(Irr_sum[ok])
	else:
		 rrI_sum_i = torch.inverse(Irr_sum)
	
	T = torch.bmm(rrI_sum_i, rrIx_sum[:,:,None])[:,:,0]

	return T

def image_meshgrid(bounds, resol):
	"""
	bounds in 3x2
	resol  in 3x1
	"""
	# he,wi,de  = resol
	# minw,maxw = bounds[0]
	# minh,maxh = bounds[1]
	# mind,maxd = bounds[2]
	axis = []
	for sz, b in zip(resol, bounds):
		binw = (b[1]-b[0]) / sz
		g = torch.arange(sz).float().cuda() * binw + 0.5 * binw
		axis.append(g)
	return torch.stack(torch.meshgrid(axis))

def masked_kp_mean(kp_loc,kp_vis):
	visibility_mass = torch.clamp(kp_vis.sum(1),1e-4)
	kp_mean = (kp_loc*kp_vis[:,None,:]).sum(2)
	kp_mean = kp_mean / visibility_mass[:,None]  
	return kp_mean

def huber(dfsq, scaling=0.03):
	loss = (safe_sqrt(1+dfsq/(scaling*scaling),eps=1e-4)-1) * scaling
	return loss

def mod1(h):
	ge1 = (h > 1.).float()
	le0 = (h < 0.).float()
	ok  = ((h>=0.) * (h<=1.)).float()
	rem_ge1 = h - h.long().float()
	rem_le0 = 1. - (-h) - (-h).long().float()
	h = ge1 * rem_ge1 + le0 * rem_le0 + ok  * h
	return h


def avg_l2_huber(x, y, mask=None, scaling=0.03, reduce_dims=[1]):
	dist = (x - y) ** 2
	if reduce_dims:
		dist = dist.sum(reduce_dims)
	dist = huber(dist, scaling=float(scaling))
	if mask is not None:
		dist = (dist*mask).sum(1) / \
					torch.clamp(mask.sum(1),1.)
	else:
		if len(dist.shape)==2 and dist.shape[1] > 1: 
			dist = dist.mean(1)
	dist = dist.mean()
	return dist

def avg_l2_dist(x,y,squared=False,mask=None,eps=1e-4):
	diff = x - y
	dist = (diff*diff).sum(1)
	if not squared: dist = safe_sqrt(dist,eps=eps)
	if mask is not None:
		dist = (dist*mask).sum(1) / \
					torch.clamp(mask.sum(1),1.)
	else:
		if len(dist.shape)==2 and dist.shape[1] > 1: 
			dist = dist.mean(1)
	dist = dist.mean()
	return dist

def argmin_translation_scale(x, y, v=None):
	# find translation/scale "T/s" st. s x + T = y
	ba = x.shape[0]	
	x = x.view(ba, 2, -1)
	y = y.view(ba, 2, -1)
	if v is not None:
		v = v.view(ba, -1)
		x_mu = (x * v[:, None]).sum(2) / v.sum(1).clamp(1.)[:, None]
		y_mu = (y * v[:, None]).sum(2) / v.sum(1).clamp(1.)[:, None]
	else:
		x_mu = x.mean(2)
		y_mu = y.mean(2)
	x = x - x_mu[:, :, None]
	y = y - y_mu[:, :, None]
	s = argmin_scale(x, y, v=v)
	T = -x_mu * s[:, None] + y_mu
	return s, T

def argmin_translation(x,y,v=None):
	# find translation "T" st. x + T = y
	x_mu = x.mean(2)
	if v is not None:
		vmass = torch.clamp(v.sum(1,keepdim=True),1e-4)
		x_mu = (v[:,None,:]*x).sum(2) / vmass
		y_mu = (v[:,None,:]*y).sum(2) / vmass
	T = y_mu - x_mu

	return T

def argmin_scale(x,y,v=None):
	# find scale "s" st.: sx=y
	if v is not None: # mask invisible
		x = x * v[:,None,:]        
		y = y * v[:,None,:]
	xtx = (x*x).sum(1).sum(1)
	xty = (x*y).sum(1).sum(1)
	s = xty / torch.clamp(xtx,1e-4)

	return s

def logexploss(x,inv_lbd,coeff=1.,accum=True):
	lbd = 1 / inv_lbd
	conj = lbd.log()
	prob = -x*lbd
	logl = -(prob+coeff*conj) # neg loglikelyhood
	
	if accum:
		return logl.mean()
	else:
		return logl

def safe_sqrt(A,eps=float(1e-4)):
	"""
	performs safe differentiable sqrt
	"""
	return (torch.clamp(A,float(0))+eps).sqrt()


def rgb2hsv(im, eps=0.0000001):
	# img = im * 0.5 + 0.5
	img = im
	# hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)
	hue = im.new_zeros( im.shape[0], im.shape[2], im.shape[3] )

	hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,2]==img.max(1)[0] ]
	hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,1]==img.max(1)[0] ]
	hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

	hue[img.min(1)[0]==img.max(1)[0]] = 0.0
	hue = hue/6

	saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + eps )
	saturation[ img.max(1)[0]==0 ] = 0

	value = img.max(1)[0]

	hsv = torch.stack((hue, saturation, value), dim=1)

	return hsv

def hsv2rgb(hsv):
	C = hsv[:,2] * hsv[:,1]
	X = C * ( 1 - ( (hsv[:,0]*6)%2 - 1 ).abs() )
	m = hsv[:,2] - C

	# zero tensor
	z = hsv[:,0] * 0.

	h = hsv[:,0]

	RGB = \
		((h <= 1/6)            )[:,None,:,:].float() * torch.stack((C,X,z), dim=1) +\
		((h > 1/6) * (h <= 2/6))[:,None,:,:].float() * torch.stack((X,C,z), dim=1) +\
		((h > 2/6) * (h <= 3/6))[:,None,:,:].float() * torch.stack((z,C,X), dim=1) +\
		((h > 3/6) * (h <= 4/6))[:,None,:,:].float() * torch.stack((z,X,C), dim=1) +\
		((h > 4/6) * (h <= 5/6))[:,None,:,:].float() * torch.stack((X,z,C), dim=1) +\
		((h > 5/6) * (h <= 6/6))[:,None,:,:].float() * torch.stack((C,z,X), dim=1)

	# if self.hsv[0] < 1/6:
	# 	R_hat, G_hat, B_hat = C, X, 0
	# elif self.hsv[0] < 2/6:
	# 	R_hat, G_hat, B_hat = X, C, 0
	# elif self.hsv[0] < 3/6:
	# 	R_hat, G_hat, B_hat = 0, C, X
	# elif self.hsv[0] < 4/6:
	# 	R_hat, G_hat, B_hat = 0, X, C
	# elif self.hsv[0] < 5/6:
	# 	R_hat, G_hat, B_hat = X, 0, C
	# elif self.hsv[0] <= 6/6:
	# 	R_hat, G_hat, B_hat = C, 0, X

	RGB = RGB + m[:,None,:,:]

	# R, G, B = (R_hat+m), (G_hat+m), (B_hat+m)
	
	return RGB


def wmean(x, weight, dim=-1):
	return (
		x.mean(dim=dim, keepdim=True) if weight is None
		else (x*weight[:,None,:]).sum(dim=dim, keepdim=True) /
			weight[:,None,:].sum(dim=dim, keepdim=True)
	)

def umeyama(X, Y, weight=None, center=True, allow_reflections=False, eps=1e-9):
	"""
	umeyama finds a rigid motion (rotation R and translation T) between two sets of points X and Y
	s.t. RX+T = Y in the least squares sense

	Inputs:
	X ... Batch x 3 x N ... each column is a 3d point
	Y ... Batch x 3 x N ... each column is a 3d point
	Outputs:
	R ... rotation component of rigid motion
	T ... translation component of rigid motion
	"""

	assert X.shape[1]==Y.shape[1]
	assert X.shape[2]==Y.shape[2]
	assert X.shape[1]==3

	b, _, n = X.size()

	if center:
		Xmu = wmean(X, weight)
		Ymu = wmean(Y, weight)
		X = X - Xmu
		Y = Y - Ymu

	Sxy = (
		torch.bmm(Y, X.transpose(2,1)) / n if weight is None
		else torch.bmm(Y*weight[:,None,:], X.transpose(2,1)*weight[:,:,None])
			/ weight.sum(-1)[:,None,None]
	)

	U, _, V = torch.svd(Sxy)
	R = torch.bmm(U, V.transpose(2,1))

	if not allow_reflections:
		s = torch.eye(3, dtype=X.dtype, device=X.device).repeat(b, 1, 1)
		s[:,-1,-1] = torch.det(R)
		# R = torch.matmul(s, R)
		R = torch.matmul(torch.matmul(U, s), V.transpose(2,1))
		assert torch.all(torch.det(R) >= 0)
	T = (
		Ymu - torch.bmm(R, Xmu[:,:])
		if center else torch.zeros_like(X)
	)[:,:,0]

	return R, T

def get_edm(pts, pts2=None):
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

def sample_random_xy(xy, mask, n=100):
	ba = xy.shape[0]
	xy = xy.reshape(ba, 2, -1)
	mask = mask.reshape(ba, -1)
	xy_sample = []
	for m_, xy_ in zip(mask, xy):
		ok = torch.nonzero(m_)
		if ok.numel() <= 2:
			warnings.warn('nothing in the mask!')
			ok = torch.nonzero(m_ + 1).squeeze()        
		ok = ok.squeeze()
		sel = torch.randint(low=0, high=len(ok), size=(n,), device=xy.device)
		xy_sample.append(xy_[:, ok[sel]])
	xy_sample = torch.stack(xy_sample)
	return xy_sample

def get_mask_chamfer(xy_rdr, gt_mask, image_size, n=100):
	ba = xy_rdr.shape[0]
	render_size = gt_mask.shape[2:]
	grid_gt = image_meshgrid(((0, 2), (0, 2)), render_size)
	grid_gt = grid_gt.type_as(xy_rdr) - 1.
	grid_gt = grid_gt[[1, 0]][None].repeat(ba, 1, 1, 1)
	# sample random points from gt mask
	gt_samples = sample_random_xy(grid_gt, gt_mask, n=n)
	# compute chamfer
	edm = get_edm(gt_samples, xy_rdr)
	edm = huber(edm, scaling=0.1)
	loss = 0.5 * (edm.min(dim=1)[0].mean() + edm.min(dim=2)[0].mean())
	return loss, gt_samples



