# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import copy
import io
import os

from matplotlib import cm
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import torch

from tools.utils import NumpySeedFix

from visdom import Visdom

import plotly.graph_objects as go
from plotly.subplots import make_subplots

viz = None

def get_visdom_env(cfg):
	if len(cfg.visdom_env)==0:
		visdom_env = os.path.basename(cfg.exp_dir)
	else:
		visdom_env = cfg.visdom_env
	return visdom_env

def get_visdom_connection(server='http://localhost',port=8097): 
	global viz
	if viz is None:    
		viz = Visdom(server=server,port=port)
	return viz

def denorm_image_trivial(im):
	im = im - im.min()
	im = im / (im.max()+1e-7)
	return im

def ensure_im_width(img,basewidth):
	# basewidth = 300
	# img = Image.open('somepic.jpg')
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((basewidth,hsize), Image.ANTIALIAS)
	return img

def denorm_image_trivial(im):
	im = im - im.min()
	im = im / (im.max()+1e-7)
	return im


def fig2data(fig, size=None):
	"""Convert a Matplotlib figure to a numpy array

	Based on the ICARE wiki.

	Args:
		fig (matplotlib.Figure): a figure to be converted

	Returns:
		(ndarray): an array of RGB values
	"""
	# TODO(samuel): convert figure to provide a tight fit in image
	buf = io.BytesIO()
	plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
	buf.seek(0)
	im = Image.open(buf).convert('RGB')
	if size:
		im = im.resize(size)
	# fig.canvas.draw()
	# import ipdb ; ipdb.set_trace()
	# # w,h = fig.canvas.get_width_height()
	# # buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
	# buf.shape = (h, w, 3)
	# return buf
	return np.array(im)

def get_depth_image(depth, mask=None, cmap='gray'):
	cmap_ = cm.get_cmap(cmap)
	clr = cmap_(depth)
	clr = clr[0].transpose(2,0,1)[:3]
	# if mask is not None:
	#     clr = clr * mask + (1-mask)
	return clr


def show_flow(
	viz,
	env,
	p,
	image=None,
	title='flow',
	linewidth=2,
	win=None,
):
	fig = plt.figure(figsize=[11,11])
	if image is not None:
		plt.imshow(image.transpose( (1,2,0) ))
		plt.axis('off')

	plt.plot(p[:,0,:], p[:,1,:], '-', color='m', linewidth=linewidth, zorder=1)

	if image is None:
		plt.gca().invert_yaxis()
		plt.axis('equal')
		plt.gca().axes.get_xaxis().set_visible(False)
		plt.gca().axes.get_yaxis().set_visible(False)
		plt.gca().set_axis_off()

	# return fig
	improj = np.array(fig2data(fig))
	if env is not None:
		win = viz.image(
			np.array(improj).transpose(2,0,1),
			env=env,
			opts={'title': title},
			win=win,
		)
	else:
		win = None

	plt.close(fig)

	return improj, win

def show_projections( viz, 
					  env, 
					  p, 
					  v=None, 
					  image_path=None, 
					  image=None, 
					  title='projs', 
					  cmap__='gist_ncar', 
					  markersize=None,
					  sticks=None, 
					  stickwidth=2,
					  stick_color=None,
					  plot_point_order=False,
					  bbox = None,
					  win=None ):
	
	if image is None:
		try:
			im = Image.open(image_path).convert('RGB')
			im = np.array(im).transpose(2,0,1)
		except:
			im = None
			print('!cant load image %s' % image_path)
	else:
		im = image

	nkp = int(p.shape[2])

	pid = np.linspace(0.,1.,nkp);             

	if v is not None:
		okp = np.where(v > 0)[0]
	else:
		okp = np.where(np.ones(nkp))[0]

	possible_markers = ['.','*','+']
	markers = [possible_markers[i%len(possible_markers)] for i in range(len(p))]

	if markersize is None:
		msz = 50
		if nkp > 40:
			msz = 5
		markersizes = [msz]*nkp
	else:
		markersizes = [markersize]*nkp

	fig = plt.figure(figsize=[11,11])
	
	if im is not None:
		plt.imshow( im.transpose( (1,2,0) ) ); plt.axis('off')

	if sticks is not None:
		if stick_color is not None:
			linecol = stick_color
		else:
			linecol = [0.,0.,0.]
		
		for p_ in p:
			for stick in sticks:
				if v is not None:
					if v[stick[0]]>0 and v[stick[1]]>0:
						linestyle='-'
					else:
						continue
				plt.plot( p_[0,stick], p_[1,stick], linestyle,
						  color=linecol, linewidth=stickwidth, zorder=1 )

	for p_, marker, msz in zip(p, markers, markersizes):        
		plt.scatter( p_[0,okp], p_[1,okp], msz, pid[okp],
					 cmap=cmap__, linewidths=2, marker=marker, zorder=2, \
					 vmin=0., vmax=1. )
		if plot_point_order:
			for ii in okp:
				plt.text( p_[0,ii], p_[1,ii], '%d' % ii, fontsize=int(msz*0.25) )
				
	if bbox is not None:
		import matplotlib.patches as patches
		# Create a Rectangle patch
		rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],\
					linewidth=1,edgecolor='r',facecolor='none')
		plt.gca().add_patch(rect)

	if im is None:
		plt.gca().invert_yaxis()
		plt.axis('equal')
		plt.gca().axes.get_xaxis().set_visible(False)
		plt.gca().axes.get_yaxis().set_visible(False)
		# plt.gca().set_frame_on(False)
		plt.gca().set_axis_off()
	else: # remove all margins    
		# plt.gca().axes.get_xaxis().set_visible(False)
		# plt.gca().axes.get_yaxis().set_visible(False)
		# plt.gca().set_frame_on(False)
		# plt.gca().set_axis_off()
		pass
		
	# return fig
	improj = np.array(fig2data(fig))  
	if env is not None:
		win = viz.image( np.array(improj).transpose(2,0,1), \
				   env=env, opts={ 'title': title }, win=win )
	else:
		win = None

	plt.close(fig)

	return improj, win



def extend_to_3d_skeleton_simple(ptcloud,sticks,line_resol=10,rgb=None):

	H36M_TO_MPII_PERM = [ 3,  2,  1,  4,  5,  6,  0,  8,  9, 10, 16, 15, 14, 11, 12, 13]

	rgb_now     = rgb.T if rgb is not None else None
	ptcloud_now = ptcloud.T

	ptcloud = ptcloud.T
	rgb     = rgb.T if rgb is not None else rgb

	if ptcloud_now.shape[1]==16: # MPII
		sticks_new = []
		for stick in sticks:
			if stick[0] in H36M_TO_MPII_PERM and  stick[1] in H36M_TO_MPII_PERM:
				s1 = H36M_TO_MPII_PERM.index(int(stick[0]))
				s2 = H36M_TO_MPII_PERM.index(int(stick[1]))
				sticks_new.append( [s1,s2] )
		sticks = sticks_new

	for sticki,stick in enumerate(sticks):
		alpha = np.linspace(0,1,line_resol)[:,None]
		linepoints = ptcloud[stick[0],:][None,:] * alpha + \
					 ptcloud[stick[1],:][None,:] * ( 1. - alpha )
		ptcloud_now = np.concatenate((ptcloud_now,linepoints),axis=0)
		if rgb is not None:
			linergb = rgb[stick[0],:][None,:] * alpha + \
					  rgb[stick[1],:][None,:] * ( 1.-alpha )
			rgb_now = np.concatenate((rgb_now,linergb.astype(np.int32)),axis=0)
	
	if rgb is not None:
		rgb_now = rgb_now.T

	return ptcloud_now.T, rgb_now

def autocolor_point_cloud(pcl, dim=1):
	d = pcl[dim]
	d = d - d.mean()
	d = d / d.std()
	d = np.minimum(np.maximum(d,-2.),2.)
	d = (d + 2.) / 4.
	rgb  = (cm.get_cmap('jet')(d)[:,:3]*255.).astype(np.int32)
	return rgb.T



def visdom_plot_pointclouds(  viz, pcl, visdom_env, title,\
							  plot_legend=False, markersize=2,\
							  nmax=5000, sticks=None, win=None, \
							  autocolor=False ):

	if sticks is not None:
		pcl = { k:extend_to_3d_skeleton_simple(v,sticks)[0] \
										for k,v in pcl.items() }

	legend = list(pcl.keys())

	cmap = 'tab10'
	npcl = len(pcl)
	rgb  = (cm.get_cmap(cmap)(np.linspace(0,1,10)) \
							[:,:3]*255.).astype(np.int32).T
	rgb = np.tile(rgb,(1,int(np.ceil(npcl/10))))[:,0:npcl]

	rgb_cat = { k:np.tile(rgb[:,i:i+1],(1,p.shape[1])) for \
							i,(k,p) in enumerate(pcl.items()) }

	rgb_cat = np.concatenate(list(rgb_cat.values()),axis=1)
	pcl_cat = np.concatenate(list(pcl.values()),axis=1)

	if pcl_cat.shape[0] > 3:
		rgb_cat = (pcl_cat[3:6, :] * 255).astype(np.int32)
		pcl_cat = pcl_cat[0:3, :]
	elif autocolor:
		rgb_cat = autocolor_point_cloud(pcl_cat)

	if pcl_cat.shape[1] > nmax:
		with NumpySeedFix():
			prm = np.random.permutation( \
						pcl_cat.shape[1])[0:nmax]
		pcl_cat = pcl_cat[:,prm]
		rgb_cat = rgb_cat[:,prm]

	win = viz.scatter( pcl_cat.T, env=visdom_env, \
			opts= { 'title': title, 'markersize': markersize,  \
					'markercolor': rgb_cat.T }, win=win )

	# legend
	if plot_legend:
		dummy_vals = np.tile(np.arange(npcl)[:,None],(1,2)).astype(np.float32)
		title = "%s_%s" % (title,legend)  
		opts = dict( title=title, legend=legend, width=400, height=400 )
		viz.line( dummy_vals.T,env=visdom_env,opts=opts,win=win+'_legend') 

	return win


def visdom_plotly_pointclouds(  viz, pcl, visdom_env, 
								title=None,
								markersize=2,
								nmax=5000, 
								sticks=None, 
								win=None,
								autocolor=False,
								in_subplots=False,
								height=500,
								width=500,
								normalise=False ):

	if sticks is not None:
		pcl = { k:extend_to_3d_skeleton_simple(v,sticks)[0] \
										for k,v in pcl.items() }
	
	npcl = len(pcl)
	rgb = np.linspace(0,1,10)
	rgb = np.array([rgb[i%10] for i in range(npcl)])

	if in_subplots:
		cols = npcl
	else:
		cols = 1

	titles = [None]*cols; titles[0] = title
	fig = make_subplots(
				rows = 1, cols = cols, 
				specs=[[{"type": "scene"}]*cols],
				subplot_titles=titles,
				column_widths=[1.]*cols,
			)

	
	for pcli, ((pcl_name, pcl_data),color) in enumerate(zip(pcl.items(), rgb)):
		
		if pcl_data.shape[1] > nmax:
			with NumpySeedFix():
				prm = np.random.permutation(pcl_data.shape[1])[0:nmax]
			pcl_data = pcl_data[:,prm]

		if pcl_data.shape[0]==6:
			# we have color
			pcl_color = np.minimum(np.maximum(pcl_data[3:],0.),1.)
			pcl_data  = pcl_data[:3]
			pcl_color = [(255.*c).astype(int).tolist() for c in pcl_color.T]
			marker=dict(
					size=markersize,
					color=pcl_color,
					opacity=1.)
		else:
			marker=dict(
					size=markersize,
					color=color,
					colorscale='Spectral',
					opacity=1.)

		if normalise:
			pcl_data -= pcl_data.mean(axis=1, keepdims=True)
			pcl_data /= (pcl_data.max(axis=1) - pcl_data.min(axis=1)).max()
			pcl[pcl_name] = pcl_data

		fig.add_trace(
			go.Scatter3d(
					x=pcl_data[0, :],
					y=pcl_data[1, :],
					z=pcl_data[2, :],
					mode='markers',
					name=pcl_name,
					visible=True,
					marker=marker,
				), 
				row = 1, 
				col = pcli+1 if in_subplots else 1
				)

	pcl_cat = np.concatenate(list(pcl.values()),axis=1)[:3]
	pcl_c = pcl_cat.mean(1)
	maxextent = (pcl_cat.max(axis=1) - pcl_cat.min(axis=1)).max()
	bounds = np.stack((pcl_c-maxextent, pcl_c+maxextent))

	height = height
	width = width * cols
	fig.update_layout(height = height, width = width,
					 scene = dict(
						xaxis=dict(range=[bounds[0,0],bounds[1,0]]),
						yaxis=dict(range=[bounds[0,1],bounds[1,1]]),
						zaxis=dict(range=[bounds[0,2],bounds[1,2]]),
						aspectmode='cube',
						)
					)

	# print(win)
	viz.plotlyplot(fig, env=visdom_env, win=win)

	return win

def write_into_image(image_np, txt, color=(0,0,255)):
    img  = Image.fromarray(image_np.transpose((1,2,0)))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), txt, color)
    image_np = np.transpose(np.array(img),(2,0,1))
    return image_np


def make_match_image(
    im_kps,
    im_paths,
    nmax=5000,
    line_width=5
):
    # images
    ims = [np.array(Image.open(im).convert('RGB')) for im in im_paths]
    _, img_width, _ = ims[0].shape
    
    # pad smaller image height if not the same
    if ims[0].shape[0] != ims[1].shape[0]:
        pad_amnt = np.abs(ims[0].shape[0] - ims[1].shape[0])
        if ims[0].shape[0] < ims[1].shape[0]:
            im_to_pad = 0
        else:
            im_to_pad = 1
        ims[im_to_pad] = np.pad(
            ims[im_to_pad], ((0, pad_amnt), (0, 0), (0, 0)), mode='constant')
        assert ims[0].shape[0] == ims[1].shape[0]
    
    ims = np.concatenate(ims, axis=1)
    ims = Image.fromarray(ims.astype(np.uint8))

    if im_kps is not None:

        # image keypoints
        if im_kps.shape[0] > nmax:
            prm = np.random.permutation(im_kps.shape[0])[0:nmax]
            im_kps = im_kps[prm]
        else:
            im_kps = im_kps.copy()
        im_kps[:,0,1] += img_width
        
        # round for imdraw
        im_kps = np.round(im_kps).astype(int)

        cmap = cm.get_cmap('rainbow')

        d = ImageDraw.Draw(ims)
        for mi, match in enumerate(im_kps):
            clr = cmap(float(mi) / im_kps.shape[0])
            clr = (np.array(clr) * 255.).astype(int).tolist()
            d.line((
                tuple(match[:,0].tolist()), 
                tuple(match[:,1].tolist())
            ), fill=tuple(clr), width=line_width)

    return ims

def visdom_show_many_image_matches( 
        viz,
        ims_kps,
        ims_paths,
        visdom_env='main', 
        visdom_win=None,
        title=None,
        line_width=10,
        nmax=5000, 
        max_im_sz=200, 
    ):

    ims = []
    for im_kps, im_paths in zip(ims_kps, ims_paths):
        im_ = make_match_image(
            im_kps,
            im_paths,
            nmax=nmax,
            line_width=line_width,
        )
        sz_ = (
            np.array(im_.size) * (max_im_sz / max(im_.size))
        ).astype(int).tolist()
        im_ = im_.resize(sz_, Image.BILINEAR)
        im_ = np.array(im_).astype(float)/255.
        im_ = np.transpose(im_, (2,0,1))
        ims.append(im_)
    
    # pad all images so that we can stack
    max_h = max(im.shape[1] for im in ims)
    max_w = max(im.shape[2] for im in ims)
    for imi, im in enumerate(ims):
        pad_h = max_h - im.shape[1]
        pad_w = max_w - im.shape[2]
        ims[imi] = np.pad(
            im, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    
    ims = np.stack(ims)
    viz.images(ims, env=visdom_env, win=visdom_win)

def _get_camera_wireframe(scale=1.):
	a = 0.5*np.array([-2,  1.5, 4])
	b = 0.5*np.array([ 2,  1.5, 4])
	c = 0.5*np.array([-2, -1.5, 4]) 
	d = 0.5*np.array([ 2, -1.5, 4])
	C = np.zeros(3)
	F = np.array([0, 0, 3])

	lines = np.array([a,b,d,c,a,C,b,d,C,c,C,F]) * scale

	return lines

def visdom_plotly_cameras(
		viz,
		cameras,
		visdom_env='main', 
		visdom_win=None,
		title=None,
		markersize=2,
		nmax=5000, 
		in_subplots=False,
		camera_scale=0.05, # in multiples of std_dev of the scene pointcloud
		height=1000,
		width=1000,
	):
	
	titles = [title]
	fig = make_subplots(
				rows = 1, cols = 1, 
				specs=[[{"type": "scene"}]],
				subplot_titles=titles,
				column_widths=[1.],
			)
	
	all_pts = []

	# add cameras
	R = cameras[:,:,:3]
	t = cameras[:,:,3:]
	C = -np.matmul(R.transpose(0, 2, 1), t)
	all_pts = C[:,:,0]

	scene_std = all_pts.std(0).mean()

	cam_lines_canonical = _get_camera_wireframe(scale=camera_scale*scene_std)

	cmap = cm.get_cmap('rainbow')
	camera_colors = cmap(np.linspace(0., 1., R.shape[0]))[:, :3]
	# mult by 220 here to make the colors a bit darker
	camera_colors = ['rgb(%s)' % ','.join(
		[str(int(c*220.)) for c in clr]
	) for clr in camera_colors]

	for clr_, R_, t_ in zip(camera_colors, R, t):
		cam_lines_world = R_.T @ (cam_lines_canonical.T - t_)
		x, y, z = cam_lines_world
		fig.add_trace(
			go.Scatter3d(
				x=x, y=y, z=z,
				marker=dict(
					size=2,
					# colorscale='Spectral',
					color=clr_,
				),
				line=dict(
					# colorscale='Spectral',
					color=clr_,
					width=2,
				)
			),
			row=1,
			col=1,
		)

	pcl_c = all_pts.mean(0)
	maxextent = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
	bounds = np.stack((pcl_c.T-maxextent, pcl_c.T+maxextent))

	fig.update_layout(height = height, width = width,
					 showlegend=False,
					 scene = dict(
							xaxis=dict(range=[bounds[0,0], bounds[1,0]]),
							yaxis=dict(range=[bounds[0,1], bounds[1,1]]),
							zaxis=dict(range=[bounds[0,2], bounds[1,2]]),
							aspectmode='cube',
						)
					)

	viz.plotlyplot(fig, env=visdom_env, win=visdom_win)