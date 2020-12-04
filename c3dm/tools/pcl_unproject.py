import torch
import torch.nn.functional as Fu

def image_meshgrid(bounds,resol):
	"""
	bounds in 3x2
	resol  in 3x1
	"""
	# he,wi,de  = resol
	# minw,maxw = bounds[0]
	# minh,maxh = bounds[1]
	# mind,maxd = bounds[2]
	axis = [ ((torch.arange(sz).float())/(sz-1))*(b[1]-b[0])+b[0] \
									for sz,b in zip(resol,bounds) ]
	return torch.stack(torch.meshgrid(axis))

def append1(X, mask=1.):
	"""
	append 1 as the last dim
	"""
	X = torch.cat( (X, X[:,-2:-1]*0. + mask), dim=1 )
	return X

def depth2pcl( D, K, image_size=None, projection_type='perspective' ):
	"""
	convert depth D in B x 1 x He x Wi 
	to a point cloud xyz_world in B x 3 x He x Wi
	using projection matrix KRT in B x 3 x 7 (K,R,T stacked along dim=2)
	the convention is: K[R|T] xyz_world = xyz_camera
	"""
	grid_size = D.shape[2:4]
	ba        = D.shape[0]
	if image_size is None: 
		image_size = grid_size
	he , wi = image_size
	projection_bounds = torch.FloatTensor( \
								[ [0.5,he-0.5],
								  [0.5,wi-0.5], ] )

	yx_cam    = image_meshgrid(projection_bounds,grid_size).type_as(D)
	xy_cam    = yx_cam[[1,0],:,:]
	xy_cam    = xy_cam[None].repeat(ba,1,1,1)
	xyz_cam   = torch.cat( (xy_cam, D), dim=1 )		

	if projection_type=='perspective':
		xyz_world = unproject_from_camera( \
						xyz_cam.view(ba,3,-1), K )
		xyz_world = xyz_world.view(ba,3,grid_size[0],grid_size[1])
	elif projection_type=='orthographic':
		xyz_world = xyz_cam
	else:
		raise ValueError(projection_type)

	return xyz_world

def unproject_from_camera( xyz_cam, K ):
	""" 
	unprojects the points from the camera coordinates xyz_cam to
	the world coordinates xyz_world
	xyz_cam in (B,3,N), 3rd dimension is depth, first two x,y pixel coords
	projection matrix KRT in B x 3 x 7 (K,R,T stacked along dim=2)
	"""
	# decompose KRT
	xy_cam    = xyz_cam[:,0:2,:]
	depth     = xyz_cam[:,2:3,:]
	# calibrate the points
	xyz_world = torch.bmm(torch.inverse(K),append1(xy_cam))
	# append depth and mult by inverse of the transformation
	xyz_world = xyz_world * depth
	
	return xyz_world