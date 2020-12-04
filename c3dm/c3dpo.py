## TODO - try shape predictor from model.py
## TODO - try repro loss from model.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import torch
from torch import nn as nn
import torch.nn.functional as Fu
import numpy as np

from tools.utils import NumpySeedFix, auto_init_args
from tools.vis_utils import get_visdom_connection, \
                            denorm_image_trivial, \
                            show_projections, \
                            visdom_plot_pointclouds
from tools.functions import masked_kp_mean, avg_l2_dist, \
                            safe_sqrt, \
                            argmin_translation, argmin_scale, \
                            avg_l2_huber, \
                            find_camera_T
from tools.so3 import so3_exponential_map, rand_rot

from dataset.dataset_configs import STICKS

from PIL import Image 

class C3DPO(torch.nn.Module):

    def __init__( self, n_keypoints               = 17, 
                        shape_basis_size          = 10, 
                        mult_shape_by_class_mask  = False,
                        squared_reprojection_loss = False, 
                        n_fully_connected         = 1024,
                        n_layers                  = 6, 
                        keypoint_rescale          = float(1),
                        keypoint_norm_type        = 'to_mean',
                        projection_type           = 'orthographic',
                        z_augment                 = True,
                        z_augment_rot_angle       = float(np.pi),
                        z_equivariance            = False,
                        z_equivariance_rot_angle  = float(np.pi)/4, # < 0 means same as z_augment_rot_angle
                        compose_z_equivariant_rot = True, # TODO: remove this soon!
                        camera_translation        = False,
                        camera_xy_translation     = True,
                        argmin_translation        = False,
                        argmin_translation_test   = False,
                        argmin_translation_min_depth = 3.,
                        argmin_to_augmented       = False,
                        camera_scale              = False,
                        argmin_scale              = False,
                        argmin_scale_test         = False,
                        loss_normalization        = 'kp_total_count',
                        independent_phi_for_aug   = False,
                        shape_pred_wd             = 1.,
                        connectivity_setup        = 'NONE',
                        custom_param_groups       = False,
                        use_huber                 = False,
                        huber_scaling             = 0.1,
                        alpha_bias                = True,
                        canonicalization = {
                            'use':               False,
                            'n_layers':          6,
                            'n_rand_samples':    4,
                            'rot_angle':         float(np.pi),
                            'n_fully_connected': 1024,
                        },
                        linear_instead_of_conv       = False,
                        perspective_depth_threshold  = 0.1,
                        depth_offset                 = 0.,
                        replace_keypoints_with_input = False,
                        root_joint                   = 0,
                        loss_weights = { 'l_reprojection':     1., 
                                         'l_canonicalization': 1. },
                        log_vars = [ \
                            'objective', 
                            'dist_reprojection',
                            'l_reprojection', 
                            'l_canonicalization' ],
                        **kwargs ):
        super(C3DPO, self).__init__()

        # autoassign constructor params to self
        auto_init_args(self) 

        # factorization net
        self.phi = nn.Sequential( \
            *make_trunk( dim_in=self.n_keypoints * 3 , # 2 dim loc, 1 dim visibility
                              n_fully_connected=self.n_fully_connected,
                              n_layers=self.n_layers ) )
        
        if linear_instead_of_conv:
            layer_init_fn = linear_layer
        else:
            layer_init_fn = conv1x1

        # shape coefficient predictor
        self.alpha_layer = layer_init_fn( self.n_fully_connected,
                                    self.shape_basis_size,
                                    init='normal0.01',
                                    cnv_args = {'bias': self.alpha_bias, 
                                                'kernel_size': 1 } )
    
        # 3D shape predictor
        self.shape_layer = layer_init_fn( self.shape_basis_size,
                                        3*n_keypoints, 
                                        init='normal0.01' )

        # rotation predictor (predicts log-rotation)
        self.rot_layer = layer_init_fn(self.n_fully_connected,3,init='normal0.01')
        if self.camera_translation:
            # camera translation
            self.translation_layer = layer_init_fn(self.n_fully_connected,3,init='normal0.01')
        if self.camera_scale:
            # camera scale (non-negative predictions)
            self.scale_layer   = nn.Sequential(  \
                                layer_init_fn(self.n_fully_connected,1,init='normal0.01'),
                                nn.Softplus() )

        if self.canonicalization['use']:
            # canonicalization net:
            self.psi = nn.Sequential( \
                    *make_trunk( dim_in=self.n_keypoints*3 ,
                                      n_fully_connected=self.canonicalization['n_fully_connected'],
                                      n_layers=self.canonicalization['n_layers'] ) )
            self.alpha_layer_psi = conv1x1( \
                        self.n_fully_connected, 
                        self.shape_basis_size,
                        init='normal0.01')

    # def _get_param_groups(self,lr,wd=0.):

    #     # make sure to set correct weight decay for the shape predictor
    #     shape_param_names = [ 'shape_pred_layer.weight', \
    #                             'shape_pred_layer.bias' ]

    #     prm_shape  = []
    #     prm_remain = []

    #     for name,prm in self.named_parameters():
    #         if not prm.requires_grad: continue
    #         if name in shape_param_names:
    #             prm_list = prm_shape
    #         else:
    #             prm_list = prm_remain
    #         prm_list.append(prm)

    #     p_groups = [ { 'params':prm_remain,'lr':float(lr), \
    #                     'weight_decay': wd },
    #                     { 'params':prm_shape, 'lr':float(lr), \
    #                     'weight_decay': float(wd*self.shape_pred_wd) } ]

    #     return p_groups


    def _get_param_groups(self,lr,wd=0.):

        assert False

        # make sure to set correct weight decay for the shape predictor
        shape_param_names = [ 'shape_pred_layer.weight', \
                              'shape_pred_layer.bias' ]

        prm_shape  = []
        prm_remain = []

        for name,prm in self.named_parameters():
            if not prm.requires_grad: continue
            if name in shape_param_names:
                prm_list = prm_shape
            else:
                prm_list = prm_remain
            prm_list.append(prm)

        p_groups = [ { 'params':prm_remain,'lr':float(lr), \
                       'weight_decay': wd },
                     { 'params':prm_shape, 'lr':float(lr), \
                       'weight_decay': float(wd*self.shape_pred_wd) } ]

        return p_groups


    def forward( self, kp_loc=None, kp_vis=None, \
                 class_mask=None, K=None, dense_basis=None, \
                 phi_out = None, dense_basis_mask=None, 
                 shape_coeff_in = None, **kwargs ):

        # dictionary with outputs of the fw pass
        preds = {}

        # input sizes ...
        ba,kp_dim,n_kp = kp_loc.shape
        dtype = kp_loc.type()        

        assert kp_dim==2, 'bad input keypoint dim'
        assert n_kp==self.n_keypoints, 'bad # of keypoints!'

        if self.projection_type=='perspective':
            kp_loc_cal = self.calibrate_keypoints(kp_loc, K)
        else:
            kp_loc_cal = kp_loc

        # save for later visualisations ...
        kp_loc_norm, kp_mean, kp_scale = \
            self.normalize_keypoints( \
                    kp_loc_cal, kp_vis, rescale=self.keypoint_rescale )
        preds['kp_loc_norm'] = kp_loc_norm
        preds['kp_mean'], preds['kp_scale'] = kp_mean, kp_scale

        # run the shape predictor
        if phi_out is not None: # bypass the predictor and use input
            preds['phi'] = phi_out
        else:
            preds['phi'] = self.run_phi(kp_loc_norm, kp_vis, \
                                class_mask=class_mask, \
                                shape_coeff_in=shape_coeff_in)

        if self.canonicalization['use']:
            preds['l_canonicalization' ], preds['psi'] = \
                self.canonicalization_loss( preds['phi'], \
                class_mask=class_mask )

        # 3D->2D project shape to camera
        kp_reprojected, depth = self.camera_projection( \
            preds['phi']['shape_camera_coord'])
        preds['kp_reprojected'] = kp_reprojected

        if dense_basis is not None:
            preds['phi_dense'] = self.run_phi_dense(dense_basis, preds['phi'])
            kp_reprojected_dense, depth_dense = self.camera_projection( \
                                preds['phi_dense']['shape_camera_coord_dense'])
            preds['kp_reprojected_dense'] = kp_reprojected_dense
            preds['depth_dense'] = depth_dense

        # compute the repro loss for backpropagation
        if self.loss_normalization=='kp_count_per_image':
            preds['l_reprojection'] = avg_l2_dist( \
                        kp_reprojected,
                        kp_loc_norm,
                        mask=kp_vis,
                        squared=self.squared_reprojection_loss )
            # print(float(preds['l_reprojection']))
        elif self.loss_normalization=='kp_total_count':
            kp_reprojected_flatten = \
                kp_reprojected.permute(1,2,0).contiguous().view(1,2,self.n_keypoints*ba)
            kp_loc_norm_flatten = \
                kp_loc_norm.permute(1,2,0).contiguous().view(1,2,self.n_keypoints*ba)
            kp_vis_flatten = \
                kp_vis.permute(1,0).contiguous().view(1,self.n_keypoints*ba)

            if self.use_huber:
                preds['l_reprojection'] = avg_l2_huber( \
                    kp_reprojected_flatten,
                    kp_loc_norm_flatten,
                    mask=kp_vis_flatten,
                    scaling=self.huber_scaling )
            else:
                assert False
                preds['l_reprojection'] = avg_l2_dist( \
                            kp_reprojected_flatten,
                            kp_loc_norm_flatten,
                            mask=kp_vis_flatten,
                            squared=self.squared_reprojection_loss )

        else:
            raise ValueError('undefined loss normalization %s' % self.loss_normalization)

        if self.squared_reprojection_loss:
            assert False
            # compute the average reprojection distance
            #   = easier to interpret than the squared repro loss
            preds['dist_reprojection'] = avg_l2_dist( \
                                            kp_reprojected,
                                            kp_loc_norm,
                                            mask=kp_vis,
                                            squared=False )

        # unnormalize the shape projections
        kp_reprojected_image = \
            self.unnormalize_keypoints(kp_reprojected, kp_mean, \
                rescale=self.keypoint_rescale, kp_scale=kp_scale)

        if dense_basis is not None:
            kp_reprojected_image_dense = \
                self.unnormalize_keypoints( \
                    preds['kp_reprojected_dense'], kp_mean, \
                    rescale=self.keypoint_rescale, kp_scale=kp_scale)
            preds['kp_reprojected_image_dense'] = kp_reprojected_image_dense
            
        # projections in the image coordinate frame
        if self.replace_keypoints_with_input and not self.training:
            # use the input points
            kp_reprojected_image = (1-kp_vis[:,None,:]) * kp_reprojected_image + \
                                    kp_vis[:,None,:]    * kp_loc_cal
            
        preds['kp_reprojected_image'] = kp_reprojected_image

        # projected 3D shape in the image space 
        #   = unprojection of kp_reprojected_image
        shape_image_coord, depth_image_coord = \
            self.camera_unprojection( \
                                kp_reprojected_image, depth, \
                                rescale=self.keypoint_rescale, \
                                kp_scale=kp_scale )
        
        if dense_basis is not None:
            shape_image_coord_dense, depth_image_coord_dense = \
                self.camera_unprojection( \
                    kp_reprojected_image_dense, depth_dense, \
                    rescale=self.keypoint_rescale, \
                    kp_scale=kp_scale )
            
        if self.projection_type=='perspective':
            preds['kp_reprojected_image_cal'] = kp_reprojected_image
            preds['shape_image_coord_cal'] = shape_image_coord
            preds['shape_image_coord'] = \
                self.uncalibrate_keypoints(shape_image_coord, K)
            preds['kp_reprojected_image'], _ = \
                self.camera_projection(preds['shape_image_coord'])
            if dense_basis is not None:
                preds['shape_image_coord_cal_dense'] = shape_image_coord_dense
                preds['shape_image_coord_dense'] = \
                    self.uncalibrate_keypoints(shape_image_coord_dense, K)
                preds['kp_reprojected_image_dense'], _ = \
                    self.camera_projection(preds['shape_image_coord_dense'])

                # if True:
                #     preds['shape_image_coord_dense'].register_hook(\
                #         lambda grad: print(grad.abs().view(-1).topk(10)[0][-1]))
                #     preds['kp_reprojected_image_dense'].register_hook(\
                #         lambda grad: print(grad.abs().view(-1).topk(10)[0][-1]))

                preds['depth_image_coord_dense'] = depth_image_coord_dense

        elif self.projection_type=='orthographic':
            preds['shape_image_coord'] = shape_image_coord
            preds['depth_image_coord'] = depth_image_coord
            if dense_basis is not None:
                preds['shape_image_coord_dense'] = shape_image_coord_dense
                preds['depth_image_coord_dense'] = depth_image_coord_dense
        
        else:
            raise ValueError()

        
        # get the final loss
        preds['objective'] = self.get_objective(preds)
        assert np.isfinite(preds['objective'].sum().data.cpu().numpy()), "nans!"
        
        return preds
        
    def camera_projection(self, shape):
        out   = {}
        depth = shape[:,2:3]
        if self.projection_type=='perspective':
            if self.perspective_depth_threshold > 0:
                depth = torch.clamp(depth, self.perspective_depth_threshold)
            projections = shape[:,0:2] / depth
        elif self.projection_type=='orthographic':
            projections = shape[:,0:2]
        else:
            raise ValueError('no such projection type %s' % \
                                            self.projection_type )

        return projections, depth

    def camera_unprojection(self,kp_loc,depth,kp_scale=None,rescale=float(1)):
        corr_scale = 1./rescale if kp_scale is None else kp_scale / rescale
        if kp_scale is not None:
            depth = depth * corr_scale[:,None,None]
        else:
            depth = depth * corr_scale
        if self.projection_type=='perspective':
            shape = torch.cat((kp_loc * depth, depth), dim=1)
        elif self.projection_type=='orthographic':
            shape = torch.cat((kp_loc, depth), dim=1)
        else:
            raise ValueError('no such projection type %s' % self.projection_type)

        return shape, depth

    def calibrate_keypoints(self, kp_loc, K):
        # undo the projection matrix
        assert K is not None
        orig_shape = kp_loc.shape
        kp_loc = kp_loc.view(orig_shape[0],2,-1) - K[:,0:2,2:3]
        focal  = torch.stack((K[:,0,0], K[:,1,1]), dim=1)
        kp_loc = kp_loc / focal[:,:,None]
        kp_loc = kp_loc.view(orig_shape)
        return kp_loc

    def uncalibrate_keypoints(self, kp_loc, K):
        assert K is not None
        ba = kp_loc.shape[0]
        kp_loc = torch.bmm(K, kp_loc.view(ba,3,-1) ).view(kp_loc.shape)
        return kp_loc

    def normalize_keypoints( self, 
                             kp_loc, 
                             kp_vis, 
                             rescale=1.,
                             kp_mean=None ):
        if self.keypoint_norm_type=='to_root':
            if kp_mean is None:
                # center around the root joint
                kp_mean = kp_loc[:,:,self.root_joint]
            kp_loc_norm = kp_loc - kp_mean[:,:,None]
            kp_scale = None
        elif self.keypoint_norm_type=='to_mean':
            if kp_mean is None:
                # calc the mean of visible points
                kp_mean = masked_kp_mean( kp_loc, kp_vis )
            # remove the mean from the keypoint locations
            kp_loc_norm = kp_loc - kp_mean[:,:,None]
            kp_scale = None
        else:
            raise BaseException( 'no such kp norm  %s' % \
                                    self.keypoint_norm_type )        

        # rescale
        kp_loc_norm = kp_loc_norm * rescale

        return kp_loc_norm, kp_mean, kp_scale

    def unnormalize_keypoints( self,
                               kp_loc_norm,
                               kp_mean,
                               rescale=1.,
                               kp_scale=None,
                               K=None ):
        kp_loc = kp_loc_norm * (1. / rescale)
        if kp_scale is not None: 
            kp_loc = kp_loc * kp_scale[:,None,None]
        
        kp_loc = (kp_loc.view(kp_loc.shape[0],2,-1)
                  + kp_mean[:, :, None]).view(kp_loc.shape)
        
        return kp_loc

    def run_phi( 
        self,
        kp_loc,
        kp_vis,
        class_mask=None,
        shape_coeff_in=None,
    ):

        preds = {}

        # batch size
        ba    = kp_loc.shape[0]
        dtype = kp_loc.type()
        eps   = 1e-4

        kp_loc_orig = kp_loc.clone()

        if self.z_augment and self.training:
            R_rand = rand_rot( ba,
                    dtype=dtype,
                    max_rot_angle=float(self.z_augment_rot_angle),
                    axes=(0,0,1) )
            kp_loc_in = torch.bmm(R_rand[:,0:2,0:2],kp_loc)
        else:
            R_rand    = torch.eye(3).type(dtype)[None].repeat( (ba,1,1) )
            kp_loc_in = kp_loc_orig
            
        if self.z_equivariance and self.training:
            if self.z_equivariance_rot_angle < 0.:
                zeq_angle = self.z_augment_rot_angle
            else:
                zeq_angle = self.z_equivariance_rot_angle
            # random xy rot
            R_rand_eq = rand_rot( ba,
                    dtype=dtype,
                    max_rot_angle=float(zeq_angle),
                    axes=(0,0,1) )

            kp_loc_in = torch.cat( \
                ( kp_loc_in, \
                torch.bmm(R_rand_eq[:,0:2,0:2], 
                    kp_loc_in if self.compose_z_equivariant_rot else kp_loc_orig) \
                ), dim=0  )    
            kp_vis_in = kp_vis.repeat( (2,1) )
        else:
            kp_vis_in = kp_vis

        # mask kp_loc by kp_visibility
        kp_loc_masked = kp_loc_in * kp_vis_in[:,None,:]

        # vectorize
        kp_loc_flatten = kp_loc_masked.view(-1, 2*self.n_keypoints)

        # concatenate visibilities and kp locations
        l1_input = torch.cat( (kp_loc_flatten,kp_vis_in) , dim=1 )

        # pass to network
        if self.independent_phi_for_aug and l1_input.shape[0]==2*ba:
            feats = torch.cat([ self.phi(l1_[:,:,None,None]) for \
                                    l1_ in l1_input.split(ba, dim=0) ], dim=0)
        else:
            feats = self.phi( l1_input[:,:,None,None] )

        # here the network runs once on concatenated input ... maybe split it?

        # coefficients into the linear basis
        shape_coeff = self.alpha_layer(feats)[:,:,0,0]     

        if self.z_equivariance and self.training:
            # use the shape coeff from the second set of preds
            shape_coeff = shape_coeff[ba:]
            # take the feats from the first set
            feats       = feats[:ba]

        if shape_coeff_in is not None:
            preds['shape_coeff_orig'] = shape_coeff
            shape_coeff = shape_coeff_in
        
        # shape prediction is just a linear layer implemented as a conv        
        shape_canonical = self.shape_layer( \
                            shape_coeff[:,:,None,None])[:,:,0,0]
        shape_canonical = shape_canonical.view(ba,3,self.n_keypoints)
        
        if self.keypoint_norm_type=='to_root':
            # make sure we fix the root at 0
            root_j = shape_canonical[:,:,self.root_joint]
            shape_canonical = shape_canonical - root_j[:,:,None]

        # predict camera params
        # ... log rotation (exponential representation)
        R_log = self.rot_layer(feats)[:,:,0,0]
        
        # convert from the 3D to 3x3 rot matrix
        R = so3_exponential_map(R_log)
        
        # T vector of the camera
        if self.camera_translation:
            T = self.translation_layer(feats)[:,:,0,0]
            if self.camera_xy_translation: # kill the last z-dim
                T = T * torch.tensor([1.,1.,0.]).type(dtype)[None,:]
        else:
            T = R_log.new_zeros(ba, 3)

        # offset the translation vector of the camera
        if self.depth_offset > 0.:
            T[:,2] = T[:,2] + self.depth_offset

        # scale of the camera
        if self.camera_scale:
            scale = self.scale_layer(feats)[:,0,0,0]
        else:
            scale = R_log.new_ones(ba)

        # rotated+scaled shape into the camera ( Y = sRX + T  )
        shape_camera_coord = self.apply_similarity_t(shape_canonical,R,T,scale)

        # undo equivariant transformation
        if (self.z_equivariance or self.z_augment) and self.training:
            R_rand_inv = R_rand.transpose(2,1)
            R = torch.bmm(R_rand_inv,R)
            T = torch.bmm(R_rand_inv,T[:,:,None])[:,:,0]
            shape_camera_coord = torch.bmm(R_rand_inv,shape_camera_coord)

        # estimate translation
        if self.argmin_translation or \
            (self.argmin_translation_test and not self.training) :
            if self.projection_type=='orthographic':
                projection, _ = self.camera_projection(shape_camera_coord)
                if self.argmin_to_augmented:
                    assert False
                    T_amin = argmin_translation( projection, kp_loc_in[:ba], v=kp_vis )
                else:
                    T_amin = argmin_translation( projection, kp_loc_orig, v=kp_vis )
                T_amin = Fu.pad(T_amin,(0,1),'constant',float(0))
                shape_camera_coord = shape_camera_coord + T_amin[:,:,None]
                T = T + T_amin
            elif self.projection_type=='perspective':
                K_ = torch.eye(3).type_as(kp_loc)[None].repeat(ba,1,1)
                T = find_camera_T(\
                    K_, shape_camera_coord, kp_loc_orig, v=kp_vis)
                if self.argmin_translation_min_depth > 0.:
                    T = torch.cat( \
                        ( T[:,0:2], \
                          torch.clamp(T[:,2:3], self.argmin_translation_min_depth)),
                        dim = 1 )
                shape_camera_coord = shape_camera_coord + T[:,:,None]
            else:
                raise ValueError(self.projection_type)

        # estimate scale
        if self.argmin_scale or \
            (self.argmin_scale_test and not self.training) :
            assert self.projection_type=='orthographic'
            # assert False
            projection, _ = self.camera_projection(shape_camera_coord)
            scale_correct = argmin_scale(projection, kp_loc_orig, v=kp_vis)
            scale = scale_correct * scale
            shape_camera_coord = scale_correct[:,None,None] * shape_camera_coord
            T = scale_correct[:,None] * T
            
        if class_mask is not None and self.mult_shape_by_class_mask:
            shape_camera_coord = shape_camera_coord * class_mask[:,None,:]
            shape_canonical    = shape_canonical * class_mask[:,None,:]

        preds['R_log']              = R_log
        preds['R']                  = R
        preds['scale']              = scale
        preds['T']                  = T
        preds['shape_camera_coord'] = shape_camera_coord
        preds['shape_coeff']        = shape_coeff
        preds['shape_canonical']    = shape_canonical

        return preds

    def run_phi_dense(self, dense_basis, phi_out):
        R, T, scale, shape_coeff = [phi_out[k] for k in ['R', 'T', 'scale', 'shape_coeff']]
        preds = {}
        ba, dim, he, wi = dense_basis.shape
        shape_basis_size = dim // 3
        dense_basis_ = dense_basis.view(ba, shape_basis_size, 3*he*wi)
        shape_coeff_1 = Fu.pad(shape_coeff, (1,0), value=1.) # mean shape goes first

        if False:
            dense_basis_decomp = dense_basis_.permute(0, 2, 1).contiguous()
            dense_basis_decomp = dense_basis_decomp.view(ba, 3, -1)
            # only rotate the basis
            dense_basis_decomp_t = \
                self.apply_similarity_t(dense_basis_decomp,R,T*0.,scale*0.+1.)
            dense_basis_decomp_t = \
                dense_basis_decomp_t.view(ba,3,he,wi, shape_basis_size)
            dense_basis_decomp_rot = dense_basis_decomp_t.permute(0,4,1,2,3)
            preds['dense_basis_rot'] = dense_basis_decomp_rot

        shape_canonical_dense = torch.bmm(shape_coeff_1[:,None,:], 
                                          dense_basis_).view(ba, 3, -1)
        shape_camera_coord_dense = self.apply_similarity_t(shape_canonical_dense,R,T,scale)
        preds['shape_canonical_dense'] = shape_canonical_dense.view(ba, 3, he, wi)
        preds['shape_camera_coord_dense'] = shape_camera_coord_dense.view(ba, 3, he, wi)
        return preds

    def apply_similarity_t( self, S, R, T, s ):
        return torch.bmm( R, s[:,None,None] * S ) + T[:,:,None]

    def canonicalization_loss( self, phi_out, class_mask=None ):

        shape_canonical = phi_out['shape_canonical']
        
        dtype = shape_canonical.type()
        ba    = shape_canonical.shape[0]

        n_sample = self.canonicalization['n_rand_samples']

        # rotate the canonical point cloud
        # generate random rotation around all axes
        R_rand = rand_rot( ba * n_sample,
                    dtype=dtype,
                    max_rot_angle=self.canonicalization['rot_angle'],
                    axes=(1,1,1) )

        unrotated = shape_canonical.repeat(n_sample, 1, 1)
        rotated   = torch.bmm( R_rand, unrotated )

        psi_out = self.run_psi( rotated ) # psi3( Rrand X )

        a , b = psi_out['shape_canonical'] , unrotated
        if self.use_huber:
            l_canonicalization = avg_l2_huber(a, b, \
                scaling=self.huber_scaling,
                mask=class_mask.repeat(n_sample,1) if class_mask is not None else None)
        else:
            l_canonicalization = avg_l2_dist(a, b, \
                squared=self.squared_reprojection_loss,
                mask=class_mask.repeat(n_sample,1) if class_mask is not None else None)

        # reshape the outputs in the output list
        psi_out = { k : v.view( \
            self.canonicalization['n_rand_samples'] , \
            ba, *v.shape[1:] ) for k,v in psi_out.items() }
        
        return l_canonicalization, psi_out

    def run_psi( self, shape_canonical ):

        preds = {}

        # batch size
        ba = shape_canonical.shape[0]
        assert shape_canonical.shape[1]==3, '3d inputs only please'
    
        # reshape and pass to the network ...
        l1_input = shape_canonical.view(ba,3*self.n_keypoints)
        
        # pass to network
        feats = self.psi( l1_input[:,:,None,None] )

        # coefficients into the linear basis
        shape_coeff = self.alpha_layer_psi(feats)[:,:,0,0]
        preds['shape_coeff'] = shape_coeff

        # use the shape_pred_layer from 2d predictor    
        shape_pred = self.shape_layer( \
                    shape_coeff[:,:,None,None])[:,:,0,0]
        shape_pred = shape_pred.view(ba,3,self.n_keypoints)
        preds['shape_canonical'] = shape_pred

        return preds

    def get_objective(self,preds):
        losses_weighted = [ preds[k] * float(w) for k,w in \
                                self.loss_weights.items() \
                                if k in preds ]
        if not hasattr(self,'_loss_weights_printed') or \
                not self._loss_weights_printed:
            print('-------\nloss_weights:')
            for k,w in self.loss_weights.items():
                print('%20s: %1.2e' % (k,w) )
            print('-------')
            self._loss_weights_printed = True
        loss = torch.stack(losses_weighted).sum()
        return loss

    def get_alpha_mean_complement(self):
        delta = self.shape_layer.weight.view(3, -1, self.shape_basis_size)
        alpha_bias = self.alpha_layer.bias.data
        mu_add = (delta * alpha_bias[None,None,:]).sum(2)
        return mu_add

    def reparametrize_mean_shape(self):
        if self.alpha_layer.bias is None:
            print('no alpha bias => skipping reparametrization')
            return
        else:
            print('reparametrizing nrsfm model mean')
        mu = self.shape_layer.bias.data.view(3, self.n_keypoints)
        mu_add = self.get_alpha_mean_complement()
        mu_new = mu + mu_add
        self.shape_layer.bias.data = mu_new.view(-1)
        self.alpha_layer.bias.data.fill_(0.)
        self.reparametrized = True

    def get_mean_shape(self):
        mu = self.shape_layer.bias.data.view(3, self.n_keypoints)
        mu_orig = mu.clone()
        if self.alpha_layer.bias is not None:
            mu_add = self.get_alpha_mean_complement()
            mu = mu + mu_add
            if hasattr(self, 'reparametrized') and self.reparametrized:
                assert (mu - mu_orig).abs().max() <= 1e-6
        return mu

    def visualize( self, visdom_env, trainmode, \
                        preds, stats, clear_env=False ):
        viz = get_visdom_connection(server=stats.visdom_server,\
                                    port=stats.visdom_port )
        if not viz.check_connection():
            print("no visdom server! -> skipping batch vis")
            return;

        if clear_env: # clear visualisations
            print("  ... clearing visdom environment")
            viz.close(env=visdom_env,win=None)

        print('vis into env:\n   %s' % visdom_env)

        it        = stats.it[trainmode]
        epoch     = stats.epoch
        idx_image = 0

        title="e%d_it%d_im%d"%(stats.epoch,stats.it[trainmode],idx_image)

        # get the connectivity pattern
        sticks = STICKS[self.connectivity_setup] if \
            self.connectivity_setup in STICKS else None

        var_kp = { 'orthographic': 'kp_reprojected_image',
                   'perspective':  'kp_reprojected_image_uncal'}[self.projection_type]

        # show reprojections
        p = np.stack( \
            [ preds[k][idx_image].detach().cpu().numpy() \
            for k in (var_kp, 'kp_loc') ] )
        v = preds['kp_vis'][idx_image].detach().cpu().numpy()

        show_projections( viz, visdom_env, p, v=v, 
                    title=title, cmap__='gist_ncar', 
                    markersize=50, sticks=sticks, 
                    stickwidth=1, plot_point_order=True,
                    image_path=preds['image_path'][idx_image],
                    win='projections' )

        # show 3d reconstruction
        if True:
            var3d = { 'orthographic': 'shape_image_coord',
                      'perspective': 'shape_image_coord_cal'}[self.projection_type]
            pcl = {'pred': preds[var3d][idx_image].detach().cpu().numpy().copy()}
            if 'kp_loc_3d' in preds:
                pcl['gt'] = preds['kp_loc_3d'][idx_image].detach().cpu().numpy().copy()
                if self.projection_type=='perspective':
                    # for perspective projections, we dont know the scale
                    # so we estimate it here ...
                    scale = argmin_scale( torch.from_numpy(pcl['pred'][None]),
                                          torch.from_numpy(pcl['gt'][None]) )
                    pcl['pred'] = pcl['pred'] * float(scale)
                elif self.projection_type=='orthographic':
                    pcl['pred'] = pcl['pred'] - pcl['pred'].mean(1)

                visdom_plot_pointclouds(viz, pcl, visdom_env, title, \
                                        plot_legend=False, markersize=20, \
                                        sticks=sticks, win='3d' )


#TODO: Make these layers nicer + move somewhere else ...

def make_trunk( 
        n_fully_connected=None,
        dim_in=None,
        n_layers=None,
        use_bn=True,
        l2_norm=False,
    ):

        layer1 = ConvBNLayer( dim_in,
                              n_fully_connected,
                              use_bn=use_bn,
                              l2_norm=l2_norm )
        layers = [layer1]

        for l in range(n_layers):
            layers.append(
                ResLayer(n_fully_connected, int(n_fully_connected/4),
                         use_bn=use_bn, l2_norm=l2_norm)
            )
    
        # print('made a trunk net:')
        # print(layers)    

        return layers


def conv1x1(in_planes, out_planes, init='no', cnv_args={'bias':True,'kernel_size':1},std=0.01):
    """1x1 convolution"""
    cnv = nn.Conv2d(in_planes, out_planes, **cnv_args)

    # init weights ...
    if init=='no':
        pass
    elif init=='normal0.01':
        # print("warning: N(0.0.01) conv weight init (different from previous exps)")
        # print('init std = %1.2e' % std)
        cnv.weight.data.normal_(0.,std)
        if cnv.bias is not None:
            cnv.bias.data.fill_(0.)
    else:
        assert False
    
    return cnv

class ConvBNLayer(nn.Module):

    def __init__(self, inplanes, planes, use_bn=True, stride=1, l2_norm=False):
        super(ConvBNLayer, self).__init__()
        
        # do a reasonable init
        cnv_args = {'kernel_size':1, 'stride':stride, 'bias':True}
        self.conv1 = conv1x1(inplanes, planes, init='normal0.01', cnv_args=cnv_args)
        self.use_bn = use_bn
        self.l2_norm = l2_norm
        if use_bn: self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        
    def forward(self, x):
        out = self.conv1(x)
        if self.l2_norm: out = Fu.normalize(out, dim=1)
        if self.use_bn: out = self.bn1(out)
        out = self.relu(out)
        return out



class ResLayer(nn.Module):

    def __init__(self, inplanes, planes, expansion=4, use_bn=True, l2_norm=False):
        super(ResLayer, self).__init__()
        self.expansion=expansion
        
        self.conv1 = conv1x1(inplanes, planes,init='normal0.01')
        if use_bn: self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv1x1(planes, planes, init='normal0.01' )
        if use_bn: self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, init='normal0.01')
        if use_bn: self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.skip = inplanes==(planes*self.expansion)
        
        self.use_bn = use_bn
        self.l2_norm = l2_norm

        # print( "reslayer skip = %d" % self.skip )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.l2_norm: out = Fu.normalize(out, dim=1)
        if self.use_bn: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.l2_norm: out = Fu.normalize(out, dim=1)
        if self.use_bn: out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.l2_norm: out = Fu.normalize(out, dim=1)
        if self.use_bn: out = self.bn3(out)
        
        if self.skip: out += residual
        out = self.relu(out)

        return out
