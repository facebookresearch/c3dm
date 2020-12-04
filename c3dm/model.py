import copy
from functools import lru_cache
import math
import os
import yaml

import numpy as np

import torch
import torch.nn.functional as Fu
from pytorch3d.renderer import cameras
from pytorch3d.transforms import so3

from visdom import Visdom

import c3dpo
from hypercolumnet import HyperColumNet

from config import get_default_args
from tools import model_io
from tools import so3 as so3int  # TODO: move random 2d rot elsewhere; use 6d from pt3d
from tools import vis_utils
import tools.eval_functions as eval_func
import tools.functions as func
from tools.loss_models import AppearanceLoss, GaussianLayer
from tools import utils
from tools.tensor_accumulator import TensorAccumulator

def conv1x1(in_planes, out_planes, init='no', cnv_args={
		'bias': True,
		'kernel_size': 1,
	}, std=0.01):
	"""1x1 convolution"""
	cnv = torch.nn.Conv2d(in_planes, out_planes, **cnv_args)

	# init weights ...
	if init == 'no':
		pass
	elif init == 'normal0.01':
		cnv.weight.data.normal_(0., std)
		if cnv.bias is not None:
			cnv.bias.data.fill_(0.)
	else:
		assert False
	
	return cnv

# Module that predicts shape and texture parameters, along with rotation
class GlobalHead(torch.nn.Module):
	def __init__( 
		self,
		input_channels,
		alpha_geom_size=0,
		alpha_tex_size=0,
		camera_code_size=0,
		add_shared_layer=True,
		glob_inst_norm=False,
	):
		super(GlobalHead, self).__init__()
		if not(alpha_tex_size > 0 or alpha_geom_size >= 0 or camera_code_size > 0):
			return

		make_fc_layer = lambda dimout: conv1x1(input_channels, dimout, init='normal0.01')

		# conv with dimout 0 does not work; use this instead
		make_degenerate = lambda feat: feat.new_empty(feat.size()[0], 0, 1, 1)
			
		# shared layer by all global stuff
		self.shared_layer = None
		if add_shared_layer:
			self.shared_layer = torch.nn.Sequential(
				make_fc_layer(input_channels),
				torch.nn.InstanceNorm2d(input_channels)
					if glob_inst_norm
					else torch.nn.BatchNorm2d(input_channels),
				torch.nn.ReLU(),
			)

		self.alpha_geom_layer = (
			make_fc_layer(alpha_geom_size)
			if alpha_geom_size > 0
			else make_degenerate if alpha_geom_size == 0 else None
		)

		self.alpha_tex_layer = make_fc_layer(alpha_tex_size) if alpha_tex_size > 0 else None

		self.rot_layer = make_fc_layer(camera_code_size) if camera_code_size else None

	def forward(self, feat):
		if self.shared_layer is not None:
			feat = self.shared_layer(feat)

		return tuple([
			(head(feat)[:,:,0,0] if head is not None else None)
			for head in (self.alpha_geom_layer, self.alpha_tex_layer, self.rot_layer)
		])


class Model(torch.nn.Module):
	def __init__( self,
				  TRUNK = get_default_args(HyperColumNet),
				  APPEARANCE_LOSS = get_default_args(AppearanceLoss),
				  nrsfm_exp_path = '',
				  huber_scaling_basis = 0.01,
				  huber_scaling_repro = 0.01,
				  photo_min_k = 6,
				  photo_reenact = False,
				  repro_loss_min_ray_length = 0.0,
				  app_mask_image = False,
				  detach_app = True,
				  uv_model_use_bn = True,
				  uv_model_l2_norm = False,
				  sampled_sil_n_samples = 1000,
				  sampled_sph_chamfer = 0,
				  spherical_embedding_radius = 1.,
				  c3dpo_flipped=True,
				  reparametrize_nrsfm_mean = True,
				  scale_aug_range = 0.2,
				  t_aug_range = 0.02,
				  rot_aug_range = 3.14/12.,
				  custom_basis_size = -1,
				  n_images_for_app_model = -1,
				  min_depth = 0.,
				  argmin_translation_min_depth = 0.,
				  argmin_translation_ray_projection = True,
				  ray_reprojection = False,
				  dilate_basis_loss = 0.,
				  EMBED_DB = get_default_args(TensorAccumulator),
				  embed_db_eval = False,
				  app_model_mask_gt = False,
				  loss_weights = {
						'loss_basis': 1.,
						'loss_alpha': 0.,
						'loss_rotation': 0.,
						'loss_repro': 0.0,
						'loss_vgg': 0.0,
						'loss_sph_emb_to_cam': 0.0,
						'loss_sph_sample_mask': 0.0,
						'loss_vgg_app': 0.0,
						'loss_l1_app': 0.0,
						'loss_ssim_app': 0.0,
						'loss_repro_2d': 0.0,
						'loss_repro_ray': 0.0,
						},
				  log_vars=[ 'objective',
							 'loss_basis',
							 'loss_alpha',
							 'loss_rotation',
							 'loss_repro',
							 'loss_repro_2d',
							 'loss_repro_ray',
							 'loss_vgg',
							 'loss_sph_emb_to_cam',
							 'loss_sph_sample_mask',
							 'loss_vgg_app',
							 'loss_l1_app',
							 'loss_ssim_app',
							 'sig_avg',
							 # depth error metrics
							 'pclerr_dist',
							  ],
				  **kwargs ):
		super(Model, self).__init__()

		# autoassign constructor params to self
		utils.auto_init_args(self)

		assert not uv_model_use_bn and uv_model_l2_norm, 'Do not use BN UV network!'

		self._load_and_fix_nrsfm()

		self.alpha_bias = None
		self.basis_size = custom_basis_size if custom_basis_size >= 0 else self.nrsfm_model.shape_basis_size
		if self.basis_size == self.nrsfm_model.shape_basis_size:
			# will be able to compute basis matching loss
			basis = torch.cat((
				self.nrsfm_model.shape_layer.bias.data.view(3, -1, 1),
				self.nrsfm_model.shape_layer.weight.data.view(3, -1, self.basis_size),
			), dim=2)

			self.nrsfm_model_basis = basis.permute(2,0,1).detach().cuda(0)
			self.alpha_bias = self.nrsfm_model.alpha_layer.bias[None,:,None,None,None].cuda(0)

		TRUNK['dimout'] = 3
		self.trunk = HyperColumNet(**TRUNK)

		self._make_glob_layers()

		if self.trunk.dimout_glob > 0:
			self._make_texture_model()

		self._make_geom_deformation_model()

		# appearance loss
		self.appearance_loss = AppearanceLoss(**APPEARANCE_LOSS)

		# init the embed database
		EMBED_DB['db_dim'] = TRUNK['dimout']
		self.embed_db = TensorAccumulator(**EMBED_DB)

	def _load_and_fix_nrsfm(self):
		self.nrsfm_model = load_nrsfm_model(self.nrsfm_exp_path)
		self.nrsfm_model.z_augment = False
		self.nrsfm_model.z_equivariance = False
		self.nrsfm_model.canonicalization.use = False
		self.nrsfm_model.perspective_depth_threshold = \
			max(self.nrsfm_model.perspective_depth_threshold, self.min_depth)
		self.nrsfm_model_kp_rescale = float(self.nrsfm_model.keypoint_rescale)
		if self.reparametrize_nrsfm_mean:
			self.nrsfm_model.reparametrize_mean_shape()
		self.nrsfm_mean_radius = self._get_nrsfm_mean_radius()
		for prm in self.nrsfm_model.parameters():
			prm.requires_grad = False
		self.nrsfm_model_basis = None
		self.projection_type = self.nrsfm_model.projection_type

		assert self.nrsfm_model.keypoint_rescale == 1.0 or self.projection_type == 'orthographic'

	def _make_glob_layers(self):
		indim = self.trunk.get_last_layer_numchannels()
		# TODO: move the relevant config params from trunk
		dimout_alpha_tex = self.trunk.dimout_glob
		dimout_alpha_geom = self.basis_size
		self.global_head = GlobalHead(
			indim,
			dimout_alpha_geom,
			dimout_alpha_tex,
			6,
			glob_inst_norm=self.trunk.glob_inst_norm,
		)

	def _make_texture_model(self):
		# make MLP mapping basis vectors + app encoding to colors
		app_dim = 3 + self.trunk.dimout_glob 
		app_layers = c3dpo.make_trunk( 
			dim_in=app_dim, 
			n_fully_connected=512,
			n_layers=3,
			use_bn=self.uv_model_use_bn,
			l2_norm=self.uv_model_l2_norm,
		)
		app_layers.append(torch.nn.Conv2d(512, 3, 1))
		self.app_model = torch.nn.Sequential(*app_layers)

	def _make_geom_deformation_model(self):
		delta_layers = c3dpo.make_trunk( 
			dim_in=3, 
			n_fully_connected=512,
			n_layers=3,
			use_bn=self.uv_model_use_bn,
			l2_norm=self.uv_model_l2_norm,
		)
		dim_out = (self.basis_size+1)*3
		delta_layers.append( torch.nn.Conv2d(512, dim_out, 1) )
		if self.trunk.final_std != 0.01:
			ldelta = delta_layers[-1]
			ldelta.weight.data = \
				ldelta.weight.data.normal_(0., self.trunk.final_std)
			ldelta.bias.data = \
				ldelta.bias.data.fill_(self.trunk.final_bias)
			print('deltanet: final bias = %1.2e, final std=%1.2e' % \
					(ldelta.bias.data.mean(),
						ldelta.weight.data.std())
					)
		# delta vectors predicted from the mean vectors
		self.delta_model = torch.nn.Sequential(*delta_layers)

	def _get_nrsfm_mean_radius(self):
		mu = self.nrsfm_model.get_mean_shape().cuda().detach()
		mumu = mu.mean(dim=1, keepdim=True)
		return ((mu - mumu) ** 2).mean() ** 0.5

	@lru_cache()
	def _get_image_grid(self, image_size, grid_size):
		imgrid = func.image_meshgrid( ((0, image_size[0]), (0, image_size[1])),
								 grid_size )
		imgrid = imgrid[[1,0]] # convert from yx to xy
		return imgrid

	def _get_distance_from_grid(self, predicted_coord, image_size, 
									  masks=None, K=None, ray_reprojection=True):
		ba = predicted_coord.shape[0]
		imgrid = self._get_image_grid(image_size, predicted_coord.size()[2:])
		imgrid = imgrid.type_as(predicted_coord)[None].repeat(ba,1,1,1)
		
		if masks is not None:
			masks = masks.view(ba, -1)

		if ray_reprojection:
			#assert self.projection_type=='perspective'
			imgrid_proj = func.calc_ray_projection(
				predicted_coord.view(ba,3,-1),
				imgrid.view(ba,2,-1),
				K = K,
				min_depth=self.min_depth,
				min_r_len=self.repro_loss_min_ray_length,
			)
			err = func.avg_l2_huber(
				imgrid_proj,
				predicted_coord.view(ba,3,-1),		
				scaling=self.huber_scaling_repro,
				mask=masks
			)
		else:
			shape_reprojected_image, _ = self.nrsfm_model.camera_projection(
				func.clamp_depth(predicted_coord, self.min_depth)
			)
			if self.projection_type=='perspective':
				imgrid = self.nrsfm_model.calibrate_keypoints(imgrid, K)
			err = func.avg_l2_huber(
				shape_reprojected_image.view(ba,2,-1),
				imgrid.view(ba,2,-1),
				scaling=self.huber_scaling_repro,
				mask=masks,
			)

		return err

	def _get_mean_basis_embed(self, embed):
		ba, _, he, wi = embed.shape
		embed_re = embed.view(ba, self.basis_size+1, 3, he, wi)
		embed_mean = embed_re[:, 0, :, :, :]
		# add the bias from the alpha layer!
		if self.alpha_bias is not None:
			embed_mean_add = (embed_re[:,1:,:,:,:] * self.alpha_bias).sum(1)
			embed_mean = embed_mean + embed_mean_add

		return embed_mean

	def _get_deltas_and_concat(self, embed):
		return self.delta_model(embed)

	def _gather_supervised_embeddings(self, embed, kp_loc, image_size):
		# uses grid sampler now (grid of size KP x 1)
		# outputs B x C x KP
		ba = embed.shape[0]
		image_size_tensor = torch.tensor(image_size).type_as(embed).flip(0)
		grid_ = 2. * kp_loc / image_size_tensor[None,:,None] - 1.
		grid_ = grid_.permute(0,2,1).view(ba, -1, 1, 2)
		supervised_embed = Fu.grid_sample(embed, grid_, align_corners=False)[:,:,:,0]
		return supervised_embed

	def _get_basis_loss(self, kp_loc, kp_vis, embed, alpha, image_size):
		assert self.nrsfm_model_basis is not None, "NRSFM basis not compatible."

		ba = kp_loc.shape[0]

		if self.dilate_basis_loss > 0.:
			ga = GaussianLayer(sigma=self.dilate_basis_loss, separated=True).cuda()
			embed = ga(embed)

		kp_embed_view = self._gather_supervised_embeddings(
			embed, kp_loc, image_size
		)
		gt_basis = self.nrsfm_model_basis.reshape(
			-1, self.nrsfm_model.n_keypoints
		)[None].repeat(ba,1,1).detach()

		return func.avg_l2_huber( gt_basis, kp_embed_view,
							 scaling=self.huber_scaling_basis,
							 mask=kp_vis[:,None,:],
							 reduce_dims=[],
							 )

	def _get_rotation_loss(self, est_rotation, nrsfm_rotation):
		rel_rotation = torch.eye(3, 3).expand_as(est_rotation)

		return 1.0 - torch.mean(
			so3.so3_relative_angle(est_rotation, nrsfm_rotation, cos_angle=True)
		)

	def _adjust_nrsfm_model_kp_scale(self, orig_image_size, image_size):
		if self.projection_type=='perspective':
			# dont change ...
			pass
		elif self.projection_type=='orthographic':
			rel_scale = 0.5 * sum( \
				float(orig_image_size.mean(0)[i]) / image_size[i] \
					for i in (0,1) )
			self.nrsfm_model.keypoint_rescale = \
				self.nrsfm_model_kp_rescale * rel_scale
		else:
			raise ValueError(self.projection_type)


	def _similarity_aug(self, images, kp_loc, kp_vis, masks=None, depths=None):
		"""
		augment images, depths, masks and kp_loc using random 
		similarity transformation
		"""
		ba, _, he, wi = images.shape

		# random scale
		r_scl = images.new_zeros(ba,).uniform_(1., 1.+self.scale_aug_range)		

		r_rot = so3int.random_2d_rotation(ba, images.type(), self.rot_aug_range)
		
		# random translation
		imdiag = float(np.sqrt(he * wi))
		r_t = images.new_zeros(ba,2).uniform_( \
			-imdiag*self.t_aug_range, imdiag*self.t_aug_range)

		# orig image grid
		grid_ = self._get_image_grid(images.shape[2:], images.shape[2:])
		grid_flat = grid_.type_as(images).repeat(ba,1,1,1).view(ba,2,-1)

		# 1st transform the keypoints
		kp_loc = torch.bmm(r_rot, kp_loc)
		kp_loc = kp_loc * r_scl[:,None,None]
		kp_loc = kp_loc - r_t[:,:,None]

		# adjust the visibilities
		ok = (kp_loc[:,0,:] >= 0.) * (kp_loc[:,1,:] >= 0.) * \
			 (kp_loc[:,0,:] < wi)  * (kp_loc[:,1,:] < he)
		kp_vis = kp_vis * ok.float()
		kp_loc[kp_vis[:, None, :].expand_as(kp_loc) < 0.5] = 0.0

		# then the image but with inverse trans
		grid_t = torch.bmm(r_rot.permute(0,2,1), grid_flat)
		grid_t = grid_t / r_scl[:,None,None]
		grid_t = grid_t + r_t[:,:,None]
		grid_t = grid_t / torch.FloatTensor([wi,he])[None,:,None].type_as(grid_t) # norm to 0, 1
		grid_t = grid_t * 2. - 1. # norm to -1, 1
		grid_t = grid_t.view(ba,2,he,wi).permute(0,2,3,1).contiguous()

		# sample the images, depth, masks
		images = Fu.grid_sample(images, grid_t, mode='bilinear', align_corners=False)
		if depths is not None:
			depths = Fu.grid_sample(depths, grid_t, mode='nearest', align_corners=False)
		if masks is not None:
			masks = Fu.grid_sample(masks, grid_t, mode='nearest', align_corners=False)

		return images, kp_loc, kp_vis, masks, depths

	def run_on_embed_db(self, preds, texture_desc, K, masks=None, image_size=None):
		embed = self.embed_db.get_db()
		embed = embed[None,:,:,None].repeat(preds['phi']['T'].size()[0], 1, 1, 1)

		# we have to downscale the embeds to make everything well-behaved
		embed_full = self._get_deltas_and_concat(embed)

		phi_out = self._get_shapes_and_projections(embed_full, None, preds['phi'], K)

		out = dict(
			embed_db_mean=embed_full,
			embed_db_shape_canonical=phi_out['shape_canonical_dense'],
			embed_db_shape_camera_coord=phi_out['shape_camera_coord_dense'],
		)
		if texture_desc is not None:
			app = self._run_app_model(embed_full, texture_desc, embed, skip_sph_assert=True)
			out['embed_db_app'] = app

		return out

	def _merge_masked_tensors(self, pcl, masks):
		c = pcl.size()[1]
		pcl = pcl.transpose(0, 1).reshape(1, c, -1)
		if masks is not None:
			pcl = pcl[..., :, masks.reshape(-1) > 0.5]
		return pcl

	def _assert_spherical_embed(self, embed):
		norms = (embed**2).sum(1).sqrt()
		# we assert that the norms are constant (std <= 0.01)
		# (in case we want to have different radius of the sphere)
		assert (
			embed.shape[1]==3
			and float(norms.std()) <= 1e-2
		), 'This can only run on spherical embeds!'

	def _get_sph_embed_towards_camera_loss(self, embed, masks, R, eps=1e-8):
		ba = embed.size()[0]
		embed = embed.reshape(ba, 3, -1)
		masks = masks.reshape(ba, 1, -1)

		avg_emb = Fu.normalize((embed * masks).sum(dim=2) / (masks.sum(dim=2) + eps), dim=-1)

		# Rotated by R, it should be ideally (0, 0, 1)
		# swap - with + for the non-flipped C3DPO
		sign = -1.0 if self.c3dpo_flipped else +1.0
		loss = 1. + sign * torch.matmul(R, avg_emb[..., None])[:, 2].mean()
		return loss

	def _calc_depth_pcl_errs(self, pred, gt, masks=None):
		# reshape the predicted depth to gt size (and rescale the values too)
		pred_up = Fu.interpolate(pred, gt.shape[2:], mode='bilinear')
		errs = eval_func.eval_depth_scale_inv( 
			pred_up.detach(), gt.detach(), masks=masks
		)

		return {'pclerr_dist': errs.mean()}

	def _get_canonical_shape(self, dense_basis, alpha, masks, target_std=2.0):
		ba, di, he, wi = dense_basis.size()
		basis = dense_basis.reshape(ba, -1, 3*he*wi)
		canon = basis[:, :1, :] + torch.bmm(alpha[:, None, :], basis[:, 1:, :])

		return canon.reshape(ba, 3, he, wi)

	def _argmin_translation(self, shape_camera_coord, shape_proj, shape_vis, K=None):
		if self.projection_type=='orthographic':
			projection, _ = self.nrsfm_model.camera_projection(shape_camera_coord)
			T_amin = func.argmin_translation(projection, shape_proj, v=shape_vis)
			T = Fu.pad(T_amin, (0,1), 'constant', float(0))
		elif self.projection_type=='perspective':
			ba = shape_camera_coord.size()[0]
			if K is None:
				K = torch.eye(3).type_as(shape_proj)[None].expand(ba, 3, 3)
			if self.argmin_translation_ray_projection:
				T = func.find_camera_T(
					K, shape_camera_coord, shape_proj, v=shape_vis
				)
			else:
				T = func.minimise_2d_residual_over_T(
					K, shape_camera_coord, shape_proj, v=shape_vis
				)
		else:
			raise ValueError(self.projection_type)

		return T

	def _argmin_camera(self, shape_canonical, masks, grid_normalised, phi):
		ba = shape_canonical.size()[0]
		centre = torch.sum(
			shape_canonical.reshape(ba, 3, -1) * masks.reshape(ba, 1, -1),
			dim=(0,2,),
			keepdim=True,
		) / masks.sum()
		shape_centered = shape_canonical.reshape(ba, 3, -1) - centre

		assert 'R' in phi, "Rotation should be given for argmin_T"
		shape_camera_rotated = torch.bmm(phi['R'], shape_centered)
		T = self._argmin_translation(
			shape_camera_rotated,
			grid_normalised.expand(shape_camera_rotated[:,:2,:].size()),
			masks.reshape(ba, -1),
			K=None, # ! points already calibrated
		)

		min_depth = self.argmin_translation_min_depth
		if min_depth > 0.:
			T = torch.cat((T[:,0:2], torch.clamp(T[:,2:3], min_depth)), dim=1)
		T = T - torch.matmul(phi['R'], centre)[:, :, 0]

		return T


	def _get_shapes_and_projections(
		self, dense_basis, masks, global_desc, K, image_repro_gt=None, alpha=None
	):
		masks = (
			masks if masks is not None
			else dense_basis.new_ones(dense_basis[:, :1, ...].size())
		)
		assert len(masks.size()) == 4

		ba = dense_basis.size()[0]
		kp_mean = global_desc['kp_mean']
		phi = copy.copy(global_desc)
		rescale = self.nrsfm_model.keypoint_rescale

		if alpha is not None:
			phi['shape_coeff'] = alpha

		if self.projection_type=='perspective':
			focal = torch.stack((K[:, 0, 0], K[:, 1, 1]), dim=1)
			p0 = K[:, :2, 2]
			camera = cameras.SfMPerspectiveCameras(
				R=phi['R'].permute(0, 2, 1),
				focal_length=focal, principal_point=p0,
				device=dense_basis.device,
			)
		else:
			camera = cameras.SfMOrthographicCameras(
				R=phi['R'].permute(0, 2, 1),
				device=dense_basis.device,
			)

		shape_canonical = self._get_canonical_shape(
			dense_basis, phi['shape_coeff'], masks
		)
		if 'T' not in phi:
			# the grid has to be calibrated (=pre-multiplied by K^{-1}) first!
			grid_im_coord = Fu.pad(
				image_repro_gt.reshape(1, 2, -1).permute(0,2,1), (0, 1), value=1.0
			).repeat(ba, 1, 1)
			grid_im_coord = camera.unproject_points(
				grid_im_coord, world_coordinates=False
			)[:,:,:2].permute(0,2,1)

			grid_normalised = (grid_im_coord - kp_mean[:,:,None]) * rescale

			phi['T'] = self._argmin_camera(
				shape_canonical, masks, grid_normalised, phi
			)

		camera.T = phi['T']

		shape_canonical_pt3d = shape_canonical.reshape(ba, 3, -1).permute(0, 2, 1)

		shape_camera_coord = camera.get_world_to_view_transform().transform_points(
			shape_canonical_pt3d
		)
		
		shape_image_coord_cal_dense = shape_camera_coord
		depth_dense = shape_camera_coord[:,:,2:]

		shape_proj_image = camera.transform_points(shape_canonical_pt3d)
		shape_reprojected_image = shape_proj_image[:, :, :2]

		# correct for the kp normalisation
		if self.projection_type == 'perspective':
			shape_image_coord_cal_dense = shape_image_coord_cal_dense + Fu.pad(
				kp_mean[:,None] * shape_camera_coord[:,:,2:], (0, 1), value=0.0
			)
			shape_reprojected_image = shape_reprojected_image + (kp_mean * focal)[:, None]
		else:
			assert self.projection_type == 'orthographic'
			shape_image_coord_cal_dense = (
				shape_image_coord_cal_dense  / rescale +
					Fu.pad(kp_mean[:,None], (0, 1), value=0.0)
			)
			shape_reprojected_image = (
				shape_reprojected_image / rescale + kp_mean[:, None]
			)

		return dict(
			phi=phi,
			shape_canonical_dense=shape_canonical,
			shape_camera_coord_dense=shape_camera_coord.permute(0, 2, 1).reshape_as(shape_canonical),
			depth_dense=depth_dense.reshape_as(shape_canonical[:, :1]),
			shape_reprojected_image=shape_reprojected_image.permute(0, 2, 1).reshape_as(shape_canonical[:, :2]),
			shape_image_coord_cal_dense=shape_image_coord_cal_dense.permute(0, 2, 1).reshape_as(shape_canonical),
		)

	def _get_best_scale(self, preds, image_size):
		if self.projection_type=='orthographic':
			shape_camera_coord = preds['shape_image_coord_cal_dense']
			ba = shape_camera_coord.shape[0]
			imgrid = self._get_image_grid(image_size, shape_camera_coord.size()[2:])
			imgrid = imgrid.type_as(shape_camera_coord)[None].repeat(ba,1,1,1)
			projection, depth = self.nrsfm_model.camera_projection(shape_camera_coord)
			s, T = func.argmin_translation_scale(projection, imgrid, v=preds['embed_masks'])
			shape_best = torch.cat((
				s[:, None, None, None] * shape_camera_coord[:, :2] + T[:, :, None, None],
				s[:, None, None, None] * shape_camera_coord[:, 2:]
			), dim=1)
		elif self.projection_type=='perspective':
			# no scale opt here, won't help
			shape_best = preds['shape_image_coord_cal_dense']
		else:
			raise ValueError(self.projection_type)

		return shape_best

	def _get_sampled_sph_loss(self, preds, K, image_size):
		masks = preds['embed_masks']
		ba = masks.shape[0]
		
		embed_sphere = torch.randn(
			size=(ba, 3, self.sampled_sil_n_samples*10, 1),
			dtype=masks.dtype, device=masks.device)
		embed_sphere = Fu.normalize(
			embed_sphere, dim=1) * self.spherical_embedding_radius

		# adjust the mean!
		embed_full = self._get_deltas_and_concat(embed_sphere)
		dense_phi = self._get_shapes_and_projections(embed_full, masks, preds, K)
		image_coords = dense_phi['shape_reprojected_image']
		shape = dense_phi['shape_image_coord_cal_dense']

		image_size_tensor = torch.FloatTensor(
			[s for s in image_size]).type_as(embed_sphere).flip(0)
		grid = 2. * (image_coords / image_size_tensor[None,:,None,None]) - 1.

		grid_prm = grid.permute(0, 2, 3, 1)
		# get all scales until the smallest side is <= 5
		samples = []
		scl = -1
		while min(masks.shape[2:]) > 4:
			scl += 1
			if scl > 0:
				masks = (Fu.interpolate(
					masks, scale_factor=0.5, mode='bilinear') > 0.).float()
			samples.append(Fu.grid_sample(masks, grid_prm, align_corners=False).view(-1))
		samples = torch.cat(samples, dim=0)
		loss = (1 - samples).mean()

		return {
			'loss_sph_sample_mask': loss,
			'sph_sample_projs': grid,
			'sph_sample_3d': shape,
		}

	def _get_photometric_losses(
		self, 
		images,
		image_coords,
		basis_embed, 
		embed_canonical=None,
		n_min=5,
		masks=None,
		texture_desc=None,
	):
		ba = images.shape[0]
		n_min = min(ba-1, n_min)
		assert ba > 1, 'batch_size > 1 for photo losses!'
		assert not (self.photo_reenact and texture_desc is None)

		image_size = list(images.shape[2:])
		image_size_render = list(basis_embed.shape[2:])
		image_size_tensor = torch.FloatTensor(image_size).type_as(basis_embed).flip(0)
		grid = 2. * (image_coords / image_size_tensor[None,:,None,None]) - 1.
		grid = grid.permute(0, 2, 3, 1)

		# image warping loss		
		if self.photo_reenact:
			images_reenact = self._run_app_model(
				basis_embed, texture_desc[0:1].repeat(ba, 1), embed_canonical
			)
		else:
			images_reenact = images
		images_reproject = Fu.grid_sample(images_reenact, grid, align_corners=False)

		# resample ref image to images_resample resolution
		images_ref = Fu.interpolate(images[:1], size=images_reproject.shape[2:])
		images_ref = images_ref.expand_as(images_reproject)

		loss_vgg, _, _ = self.appearance_loss(images_reproject, images_ref)
		loss_vgg = loss_vgg[:, 0]

		# transplant the rendered image by tokp pooling
		assert (~torch.isnan(loss_vgg)).all(), "Some photometric loss values are NaN."
		if masks is not None:
			# weight the losses by seg masks
			loss_vgg = masks[:1, 0] * loss_vgg

		loss_topk, idx_render = torch.topk(loss_vgg[1:], n_min-1, dim=0, largest=False)
		# make sure we include the target view
		loss_vgg = (loss_topk.sum(0) + loss_vgg[0]) / n_min

		idx_render = idx_render[:,None].expand(-1, 3, -1, -1)
		im_render = {
			'loss_vgg': (
				torch.gather(images_reproject, 0, idx_render).sum(0) + images_reproject[0]
			 ) / n_min
		}

		out = {}
		out['loss_vgg'] = loss_vgg.mean()
		out['images_reproject'] = images_reproject.detach()
		out['images_gt'] = images_ref.detach()
		out['image_ref_render'] = im_render
		out['images'] = Fu.interpolate(images, size=images_reproject.shape[2:]).detach()
		out['images_reenact'] = Fu.interpolate(images_reenact, size=images_reproject.shape[2:]).detach()

		return out

	def _mask_gt_image(self, image, mask):
		avgcol = (image * mask).sum((2, 3)) / mask.sum((2, 3)).clamp(1)
		image_m = image * mask + (1-mask) * avgcol[:, :, None, None]
		# blur and mix
		ga = GaussianLayer(sigma=5., separated=True).cuda()			
		image_mf = ga(image_m)
		image_m = mask * image_m + (1-mask) * image_mf
		return image_m

	def _run_app_model(self, embed, texture_desc, embed_canonical, skip_sph_assert=False):
		# run the appearance model taking as input per-pixel uv-like 
		# embeddings `embed` and the global appearance descriptor
		# `texture_desc`

		n_im_use = self.n_images_for_app_model if \
					self.n_images_for_app_model > 0 else embed_canonical.size()[0]
		texture_desc = texture_desc[:n_im_use]

		embed_for_app = embed_canonical[:n_im_use]
		if not skip_sph_assert:
			self._assert_spherical_embed(embed_for_app)
		
		if self.detach_app:
			embed_for_app = embed_for_app.detach()

		embed_app = torch.cat((
			texture_desc[:,:,None,None].expand(-1,-1,*list(embed.shape[2:])), 
			embed_for_app,
		), dim=1)
		app = self.app_model(embed_app)
		
		return app[:, :3] + 0.5

	def _get_app_model_losses(
		self, 
		images,
		preds_app,
		masks=None,
		sigma=None,
	):
		# for now this is the same
		images_pred = preds_app
		ba = images_pred.shape[0]
		image_size = list(images.shape[2:])
		image_size_render = list(images_pred.shape[2:])

		if masks is not None:
			# weight the losses by seg masks
			masks = Fu.interpolate(masks[:ba], size=image_size_render, mode='nearest')
		
		# resample ref image to images_resample resolution
		images_gt = Fu.interpolate(images[:ba], size=image_size_render)

		# mask the images and do NN interp
		if self.app_model_mask_gt:
			images_gt = self._mask_gt_image(images_gt, masks)
		
		loss_vgg, loss_rgb, _ = \
			self.appearance_loss(
				images_pred,
				images_gt,
				sig=sigma,
				mask=masks if self.app_mask_image else None
				)

		if masks is not None:
			# weight the losses by seg masks
			loss_vgg, loss_rgb = \
				[ (masks * l).sum() / torch.clamp(masks.sum(), 1e-1) \
					for l in (loss_vgg, loss_rgb,) ]
		else:
			loss_vgg, loss_rgb = \
				[ l.mean() \
					for l in (loss_vgg, loss_rgb,) ]

		out = {}
		out['loss_vgg'] = loss_vgg
		out['loss_l1'] = loss_rgb
		out['loss_ssim'] = (loss_rgb * 0.0).detach()  # not used
		out['images_pred'] = images_pred
		out['images_pred_clamp'] = torch.clamp(images_pred,0.,1.)
		out['images_gt'] = images_gt
		out['images'] = images_gt

		return out


	def forward(
		self, 
		kp_loc=None,
		kp_vis=None,
		kp_conf=None,
		images=None,
		epoch_now=None,
		orig_image_size=None,
		masks=None,
		depths=None,
		K=None,
		**kwargs
	):

		ba = images.shape[0]  # batch size
		image_size = images.size()[2:]

		# adjust nrsfm model scale
		self._adjust_nrsfm_model_kp_scale(orig_image_size, image_size)

		preds = {}
		preds['nrsfm_mean_shape'] = self.nrsfm_model.get_mean_shape()

		if self.training and (
			self.scale_aug_range > 0. or
			self.t_aug_range > 0. or
			self.rot_aug_range > 0.
		):
			images, kp_loc, kp_vis, masks, depths = \
				self._similarity_aug(images, kp_loc, kp_vis, 
									 masks=masks, depths=depths)
			preds.update(
				{ 'images_aug': images, 'kp_loc_aug': kp_loc,
				  'depths_aug': depths, 'masks_aug': masks }
			)
		
		embed, glob_features = self.trunk(
			images, kp_loc_vis = torch.cat((kp_loc, kp_vis[:,None,:]), dim=1)
		)
		embed = Fu.normalize(embed, dim=1) * self.spherical_embedding_radius
		embed_full = self._get_deltas_and_concat(embed)
		#embed_masks = (Fu.interpolate(masks, embed.shape[2:], mode='bilinear') > 0.49).float()
		embed_masks = Fu.interpolate(masks, embed.shape[2:], mode='nearest')
		image_repro_gt = self._get_image_grid(image_size, embed_full.size()[2:])
			
		preds['embed'] = embed
		preds['embed_full'] = embed_full
		preds['embed_masks'] = embed_masks
		preds['embed_mean'] = self._get_mean_basis_embed(embed_full)
		preds['image_repro_gt'] = image_repro_gt

		alpha_geom, texture_desc, rotation_code = self.global_head(glob_features)

		self.nrsfm_model.eval()
		preds['nrsfm'] = self.nrsfm_model(
			kp_loc=kp_loc,
			kp_vis=kp_vis,
			dense_basis=None,  # estimate dense Phi here
			K=K,
		)

		assert not self.nrsfm_model.camera_scale  # so just ones
		assert self.nrsfm_model.argmin_translation
		#preds['kp_mean'] = preds['nrsfm']['kp_mean']   # TODO: this should go away

		# override top-level preds if regressing directly
		assert rotation_code is not None
		assert alpha_geom is not None

		global_desc = dict(
			shape_coeff=alpha_geom,
			R=so3int.so3_6d_to_rot(rotation_code),
			kp_mean=preds['nrsfm']['kp_mean'],
		)
		preds.update(self._get_shapes_and_projections(
			embed_full, embed_masks, global_desc, K, image_repro_gt
		))

		preds['shape_image_coord_cal'] = self._gather_supervised_embeddings(
			preds['shape_image_coord_cal_dense'], # same as uncal for orthographic
			kp_loc,
			image_size,
		)
		preds['kp_reprojected_image'] = self._gather_supervised_embeddings(
			preds['shape_reprojected_image'],
			kp_loc,
			image_size,
		)

		# compute NR-SFM Prior loss
		if self.loss_weights['loss_basis'] > 0.:           
			preds['loss_basis'] = self._get_basis_loss(
				kp_loc, 
				kp_vis,
				embed_full,
				preds['nrsfm']['phi']['shape_coeff'], 
				image_size,
			)

		if self.loss_weights.loss_alpha > 0.:
			assert alpha_geom is not None
			preds['loss_alpha'] = func.huber( \
				(alpha_geom - preds['nrsfm']['phi']['shape_coeff'])**2,
				scaling=self.huber_scaling_basis,
			).mean()

		if self.loss_weights.loss_rotation > 0.:
			preds['loss_rotation'] = self._get_rotation_loss(
				preds['phi']['R'],
				preds['nrsfm']['phi']['R'],
			)

		# compute reprojection loss
		preds['loss_repro_2d'] = self._get_distance_from_grid(
			preds['shape_image_coord_cal_dense'],
			image_size,
			masks=embed_masks,
			K=K, 
			ray_reprojection=False,
		)

		# preds['loss_repro_ray'] = 0.0
		# if self.projection_type == 'perspective':
		preds['loss_repro_ray'] = self._get_distance_from_grid(
			preds['shape_image_coord_cal_dense'],
			image_size,
			masks=embed_masks,
			K=K,
			ray_reprojection=True,
		)

		preds['loss_repro'] = preds['loss_repro_ray'] if self.ray_reprojection else preds['loss_repro_2d']

		# perceptual loss
		preds['photo_out'] = None
		if self.photo_min_k > 0 and ba > 1:
			# use the first im as a loss as a target
			basis_embed_ref = embed_full[:1].expand_as(embed_full)
			masks_ref = embed_masks[:1].expand_as(embed_masks)
			phi_onto_ref = self._get_shapes_and_projections(basis_embed_ref, masks_ref, preds['phi'], K)

			preds['photo_out'] = self._get_photometric_losses(
				images,
				phi_onto_ref['shape_reprojected_image'],
				embed_full,
				texture_desc=texture_desc,
				n_min=self.photo_min_k,
				masks=embed_masks,
				embed_canonical=embed,
			)
			preds['loss_vgg'] = preds['photo_out']['loss_vgg']

		# embedding-camera alignment loss
		if self.loss_weights['loss_sph_emb_to_cam'] > 0.:
			preds['loss_sph_emb_to_cam'] = self._get_sph_embed_towards_camera_loss(
				preds['embed'], embed_masks, preds['phi']['R'].detach()
			)

		# mask sampling loss
		if self.loss_weights['loss_sph_sample_mask'] > 0.:
			preds.update(self._get_sampled_sph_loss(preds, K, images.shape[2:]))

		# appearance model
		preds['app'] = None
		if texture_desc is not None:
			n_im_use = (
				self.n_images_for_app_model
					if self.n_images_for_app_model > 0
					else ba	
			)
			preds['app'] = self._run_app_model(
				embed_full[:n_im_use], texture_desc[:n_im_use], embed
			)

			preds['app_out'] = self._get_app_model_losses( 
				images, preds['app'][:, :3], masks=masks,
			)
			for k in ('loss_vgg', 'loss_l1', 'loss_ssim'):
				preds[k+'_app'] = preds['app_out'][k]

		# finally get the optimization objective using self.loss_weights
		preds['objective'] = self.get_objective(preds, epoch_now=epoch_now)

		# =================
		# the rest is only for visualisation/metrics

		# run on cached embed_db
		if self.embed_db is not None and self.embed_db_eval:
			preds.update(self.run_on_embed_db(preds, texture_desc, K, 
						 masks=embed_masks, image_size=image_size))

		# accumulate into embed db
		self.embed_db(embed, masks=embed_masks)

		depth_pcl_metrics = self._calc_depth_pcl_errs(
			preds['depth_dense'], depths, masks=masks
		)
		preds.update(depth_pcl_metrics)

		# find the scale of shape_image_coord that minimizes the repro loss
		preds['shape_image_coord_best_scale'] = self._get_best_scale(preds, image_size)

		preds['nrsfm_shape_image_coord'] = preds['nrsfm'][{ 
			'orthographic': 'shape_image_coord',
			'perspective': 'shape_image_coord_cal',
		}[self.projection_type]]

		# a hack for vis purposes
		preds['misc'] = {}
		for k in ('images', 'images_app', 'images_geom', 'embed'):
			if k in preds:
				preds['misc'][k] = preds[k].detach()
			elif k in vars():
				preds['misc'][k] = vars()[k]

		return preds

	def get_objective(self, preds, epoch_now=None):
		losses_weighted = {
			k: preds[k] * float(w)
			for k, w in self.loss_weights.items()
			if k in preds and w != 0.0  # avoid adding NaN * 0
		}

		if not hasattr(self,'_loss_weights_printed') or \
				not self._loss_weights_printed:
			print('-------\nloss_weights:')
			for k,w in self.loss_weights.items():
				print('%20s: %1.2e' % (k,w) )
			print('-------')
		
			print('-------\nweighted losses:')
			for k,w in losses_weighted.items():
				print('%20s: %1.2e' % (k,w) )
			print('-------')
		
			self._loss_weights_printed = True

		loss = torch.stack(list(losses_weighted.values())).sum()
		return loss


	def visualize( self, visdom_env_imgs, trainmode, \
						preds, stats, clear_env=False ):
		if stats is not None:
			it = stats.it[trainmode]
			epoch = stats.epoch
			viz = vis_utils.get_visdom_connection(
				server=stats.visdom_server,
				port=stats.visdom_port,
			)
		else:
			it = 0
			epoch = 0
			viz = vis_utils.get_visdom_connection()
		
		if not viz.check_connection():
			print("no visdom server! -> skipping batch vis")
			return

		idx_image = 0

		title="e%d_it%d_im%d"%(epoch,it,idx_image)

		imvar = 'images_aug' if 'images_aug' in preds else 'images'
		dvar  = 'depths_aug' if 'depths_aug' in preds else 'depths'
		mvar  = 'masks_aug' if 'masks_aug' in preds else 'masks'

		# show depth
		ds  = preds['depth_dense'].cpu().detach().repeat(1,3,1,1)
		ims = preds[imvar].cpu().detach()
		ims = Fu.interpolate(ims,size=ds.shape[2:])
		if mvar in preds: # mask depths, ims by masks
			masks = Fu.interpolate(preds[mvar].cpu().detach(),
								   size=ds.shape[2:], mode='nearest' )
			ims *= masks ; ds *= masks
		ds = vis_utils.denorm_image_trivial(ds)
		if 'pred_mask' in preds:
			pred_mask = torch.sigmoid(preds['pred_mask'][:, None].detach()).cpu().expand_as(ims)
			ims_ds = torch.cat( (ims, ds, pred_mask), dim=2 )
		else:
			ims_ds = torch.cat( (ims, ds), dim=2 )
		viz.images(ims_ds, env=visdom_env_imgs, opts={'title':title}, win='depth')

		# show aug images if present
		imss = []
		for k in (imvar, 'images_app', 'images_geom'):
			if k in preds:
				ims = preds[k].cpu().detach()
				ims = Fu.interpolate(ims, scale_factor=0.25)
				ims = vis_utils.denorm_image_trivial(ims)
				R, R_gt = preds['phi']['R'], preds['nrsfm']['phi']['R']
				angle_to_0 = np.rad2deg(
					so3.so3_relative_angle(R[0].expand_as(R), R).data.cpu().numpy()
				)
				angle_to_0_gt = np.rad2deg(
					so3.so3_relative_angle(R_gt[0].expand_as(R_gt), R_gt).data.cpu().numpy()
				)
				if ~np.isnan(angle_to_0).any():
					ims = np.stack([
						vis_utils.write_into_image(
							(im*255.).astype(np.uint8), "%d° / %d°" % (d, d_gt), color=(255,0,255)
						) for im, d, d_gt in zip(ims.data.numpy(), angle_to_0, angle_to_0_gt)
					])
				else:
					ims = (ims.data.numpy()*255.).astype(np.uint8)
				imss.append(ims)
		if len(imss) > 0:
			viz.images(
				#torch.cat(imss, dim=2), 
				np.concatenate(imss, axis=2).astype(np.float32)/255.,
				env=visdom_env_imgs, 
				opts={'title': title}, 
				win='imaug',
			)
	
		# show reprojections
		p1 = preds['kp_loc_aug' if 'kp_loc_aug' in preds else 'kp_loc'][idx_image]
		p2 = preds['kp_reprojected_image'][idx_image,0:2]
		p3 = preds['nrsfm']['kp_reprojected_image'][idx_image]
		p = np.stack([p_.detach().cpu().numpy() for p_ in (p1, p2, p3)])
		v = preds['kp_vis'][idx_image].detach().cpu().numpy()
		vis_utils.show_projections( viz, visdom_env_imgs, p, v=v, 
					title=title, cmap__='rainbow', 
					markersize=50, sticks=None, 
					stickwidth=1, plot_point_order=False,
					image=preds[imvar][idx_image].detach().cpu().numpy(),
					win='projections' )

		# dense reprojections
		p1 = preds['image_repro_gt'].detach().cpu()
		p2 = preds['shape_reprojected_image'][idx_image].detach().cpu()
		# override mask with downsampled (augmentation applied if any)
		mvar = 'embed_masks'
		if mvar in preds:
			masks = preds[mvar].detach().cpu()
			#masks = Fu.interpolate(masks, size=p2.shape[1:], mode='nearest')
			p1 = p1 * masks[idx_image]
			p2 = p2 * masks[idx_image]

		# TEMP
		img = (preds[imvar][idx_image].cpu() * Fu.interpolate(
			preds[mvar].cpu()[idx_image:idx_image+1], size=preds[imvar][0, 0].size(), mode='nearest'
		)[0]).data.cpu().numpy()
		p = np.stack([p_.view(2,-1).numpy() for p_ in (p1, p2)])
		vis_utils.show_projections( viz, visdom_env_imgs, p, v=None, 
					title=title, cmap__='rainbow', 
					markersize=1, sticks=None, 
					stickwidth=1, plot_point_order=False,
					image=img,
					win='projections_dense' )
		vis_utils.show_flow(viz, visdom_env_imgs, p,
			image=preds[imvar][idx_image].detach().cpu().numpy(),
			title='flow ' + title,
			linewidth=1,
			win='projections_flow',
		)

		if 'sph_sample_projs' in preds:
			p = preds['sph_sample_projs'][idx_image].detach().cpu().view(2, -1)
			if 'sph_sample_gt' in preds:
				p_ = preds['sph_sample_gt'][idx_image].detach().cpu().view(2, -1)
				p_ = p_.repeat(1, math.ceil(p.shape[1]/p_.shape[1]))
				p = [p, p_[:, :p.shape[1]]]
			else:
				p = [p.view(2, -1)]
			# p = (torch.stack(p) + 1.) / 2.
			p = (torch.stack(p) + 1.) / 2.
			imsize = preds[imvar][idx_image].shape[1:] 
			p[:, 0, :] *= imsize[1]
			p[:, 1, :] *= imsize[0]
			vis_utils.show_projections(viz, visdom_env_imgs, 
				p, v=None,
				title=title + '_spl_sil', 
				cmap__='rainbow', 
				markersize=1, sticks=None, 
				stickwidth=1, plot_point_order=False,
				image=preds[imvar][idx_image].detach().cpu().numpy(),
				win='projections_spl_sil'
			)

		merged_embed = self._merge_masked_tensors(
			preds['embed_full'], preds['embed_masks']
		)[..., None]
		gl_desc_0 = {k: v[:1] for k, v in preds['phi'].items()}
		merged_with_pivot_phis = self._get_shapes_and_projections(
			merged_embed, None, gl_desc_0, preds['K'][:1]
		)
		preds['shape_canonical_same_alphas'] = merged_with_pivot_phis[
			'shape_canonical_dense'
		][0 ,..., 0]

		# dense 3d
		pcl_show = {}
		vis_list = ['dense3d', 'mean_shape', 'embed_db', 'batch_fused', 'sph_embed']
		if self.loss_weights['loss_sph_sample_mask'] > 0:
			vis_list.append('sph_sample_3d')

		for vis in vis_list:
			if vis=='canonical':
				pcl = preds['shape_canonical_dense']
			elif vis=='dense3d':
				pcl = preds['shape_image_coord_cal_dense']
			elif vis=='batch_fused':
				pcl = preds['shape_canonical_same_alphas'].detach().cpu()
				pcl = torch.cat((pcl, pcl), dim=0)
				pcl[3:5,:] = 0.0
				pcl[5,:] = 1.0
			elif vis=='mean_shape':
				pcl = preds['embed_mean']
			elif vis=='mean_c3dpo_shape':
				pcl = preds['nrsfm_mean_shape']
			elif vis=='shape_canonical':
				pcl = preds['shape_canonical_dense']
			elif vis == 'sph_embed':
				pcl = preds['embed'].detach().clone()
			elif vis == 'sph_sample_3d':
				pcl = preds['sph_sample_3d'][idx_image].detach().cpu().view(3, -1)
				pcl = torch.cat((pcl, pcl.clone()), dim=0)
				pcl[4:,:] = 0.0
				pcl[3,:] = 1.0
				# filtering outliers
				pcl[:3] -= pcl[:3].mean(dim=1, keepdim=True) # will be centered anyway
				std = pcl[:3].std(dim=1).max()
				pcl[:3] = pcl[:3].clamp(-2.5*std, 2.5*std)
			elif vis == 'embed_db':
				pcl = self.embed_db.get_db(uniform_sphere=False).cpu().detach().view(3, -1)
				pcl = torch.cat((pcl, pcl.clone()), dim=0)
				pcl[3:5,:] = 0.0
				pcl[4,:] = 1.0
			else:
				raise ValueError(vis)

			if vis not in ('mean_c3dpo_shape', 'batch_fused', 'sph_sample_3d', 'embed_db'):
				pcl_rgb = preds[imvar].detach().cpu()
				#pcl = Fu.interpolate(pcl.detach().cpu(), pcl_rgb.shape[2:], mode='bilinear')
				pcl_rgb = Fu.interpolate(pcl_rgb, size=pcl.shape[2:], mode='bilinear')
				if (mvar in preds):
					masks = preds[mvar].detach().cpu()
					masks = Fu.interpolate(masks, \
						size=pcl.shape[2:], mode='nearest')
				else:
					masks = None
				
				pcl = pcl.detach().cpu()[idx_image].view(3,-1)
				pcl_rgb = pcl_rgb[idx_image].view(3,-1)
				pcl = torch.cat((pcl, pcl_rgb), dim=0)
				if masks is not None:
					masks = masks[idx_image].view(-1)
					pcl = pcl[:,masks>0.] 

				# if vis == 'sph_embed':
				# 	import pdb; pdb.set_trace()

			if pcl.numel()==0: 
				continue
			pcl_show[vis] = pcl.numpy()

		vis_utils.visdom_plotly_pointclouds(viz, pcl_show, visdom_env_imgs,
							title=title+'_'+vis,
							markersize=1,
							sticks=None, win=vis,
							height=700, width=700 ,
							normalise=True,
		)

		var3d = { 
			'orthographic': 'shape_image_coord',
			'perspective': 'shape_image_coord_cal',
		}[self.projection_type]
		sparse_pcl = {
			'nrsfm': preds['nrsfm'][var3d][idx_image].detach().cpu().numpy().copy(),
			'dense': preds['shape_image_coord_cal'][idx_image].detach().cpu().numpy().copy(),
		}
		if 'kp_loc_3d' in preds:
			sparse_pcl['gt'] = preds['kp_loc_3d'][idx_image].detach().cpu().numpy().copy()

		if 'class_mask' in preds:
			class_mask = preds['class_mask'][idx_image].detach().cpu().numpy()
			sparse_pcl = {k: v*class_mask[None] for k,v in sparse_pcl.items()}
		
		vis_utils.visdom_plotly_pointclouds(viz, sparse_pcl, visdom_env_imgs, \
								title=title+'_sparse3d', \
								markersize=5, \
								sticks=None, win='nrsfm_3d',
								height=500,
								width=500 )

		if 'photo_out' in preds and preds['photo_out'] is not None:
			# show the source images and their renders
			ims_src     = preds['photo_out']['images'].detach().cpu()
			ims_repro   = preds['photo_out']['images_reproject'].detach().cpu()
			ims_reenact = preds['photo_out']['images_reenact'].detach().cpu()
			ims_gt      = preds['photo_out']['images_gt'].detach().cpu()

			# cat all the images
			ims = torch.cat((ims_src,ims_reenact,ims_repro,ims_gt), dim=2)
			ims = torch.clamp(ims,0.,1.)
			viz.images(ims, env=visdom_env_imgs, opts={'title':title}, win='imrepro')
			
			im_renders = preds['photo_out']['image_ref_render']
			for l in im_renders:
				im_gt = preds['photo_out']['images_gt'][0].detach().cpu()
				im_render = im_renders[l].detach().cpu()
				im = torch.cat((im_gt, im_render), dim=2)
				im = torch.clamp(im, 0., 1.)
				viz.image(im, env=visdom_env_imgs, \
					opts={'title':title+'_min_render_%s' % l}, win='imrender_%s' % l)

		if 'app_out' in preds and preds['app_out'] is not None:
			# show the source images and their predictions
			ims_src  = preds['app_out']['images'].detach().cpu()
			ims_pred = preds['app_out']['images_pred_clamp'].detach().cpu()
			ims = torch.cat((ims_src,ims_pred), dim=2)
			viz.images(ims, env=visdom_env_imgs, opts={'title':title}, win='impred')


def load_nrsfm_model(exp_name, get_cfg=False):
	from dataset.dataset_configs import C3DPO_MODELS, C3DPO_URLS
	if exp_name in C3DPO_MODELS:
		exp_path = C3DPO_MODELS[exp_name]
	else:
		exp_path = exp_name

	if not os.path.exists(exp_path):
		url = C3DPO_URLS[exp_name]
		print('Downloading C3DPO model %s from %s' % (exp_name, url))
		utils.untar_to_dir(url, exp_path)
	
	cfg_file = os.path.join(exp_path, 'expconfig.yaml')
	assert os.path.isfile(cfg_file), 'no config for NR SFM %s!' % cfg_file

	with open(cfg_file, 'r') as f:
		cfg = yaml.load(f)

	# exp = ExperimentConfig(cfg_file=cfg_file)
	nrsfm_model = c3dpo.C3DPO(**cfg.MODEL)
	model_path = model_io.find_last_checkpoint(exp_path)
	assert model_path is not None, "cannot found previous NR SFM model %s" % model_path
	print("Loading the model from", model_path)
	model_state_dict, _, _ = model_io.load_model(model_path)
	nrsfm_model.load_state_dict(model_state_dict, strict=True)

	if get_cfg:
		return nrsfm_model, cfg
	else:
		return nrsfm_model

