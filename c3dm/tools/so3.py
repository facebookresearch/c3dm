#!/usr/bin/env python3
import torch
import math
import torch.nn.functional as Fu


def so3_6d_to_rot(d6):
    """
    d6 ... batch x 6

    Follows Sec. B in the appendix of:
    https://arxiv.org/pdf/1812.07035.pdf
    """

    a1, a2 = d6[:, :3], d6[:, 3:]
    b1 = Fu.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(1, keepdim=True) * b1
    b2 = Fu.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2)
    R  = torch.stack((b1, b2, b3), dim=1)

    # if True:
    #     assert torch.allclose(torch.det(R), R.new_ones(R.shape[0]))

    return R


def so3_relative_angle(R1, R2):
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with
    :math: `(\\phi = \text{acos}\frac{\text{Trace}(R_1 R_2^T)-1}{2})`.

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape :math:`(\text{minibatch}, 3, 3)`.
        R2: Batch of rotation matrices of shape :math:`(\text{minibatch}, 3, 3)`.

    Returns:
        Corresponding rotation angles of shape :math:`(\text{minibatch},)`.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12)


def so3_rotation_angle(R, eps: float = 1e-4):
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    :math: `\\phi = \text{acos}\frac{\text{Trace}(R)-1}{2}`. The trace of the
    input matrices is checked to be in the valid range [-1-`eps`,3+`eps`].
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape :math:`(\text{minibatch}, 3, 3)`.
        eps: Tolerance for the valid trace check.

    Returns:
        Corresponding rotation angles of shape :math:`(\text{minibatch},)`.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N , dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError('Input has to be a batch of 3x3 Tensors.')

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1. - eps) + (rot_trace > 3. + eps)).any():
        raise ValueError('A matrix has trace outside valid range [-1-eps,3+eps].')

    # clamp to valid range
    rot_trace = torch.clamp(rot_trace, -1., 3.)

    # phi ... rotation angle
    phi = (0.5 * (rot_trace - 1.)).acos()

    return phi


def rand_rot(N,dtype=torch.float32,max_rot_angle=float(math.pi),\
                axes=(1,1,1),get_ss=False):

    rand_axis  = torch.zeros( (N,3) ).type(dtype).normal_()

    # apply the axes mask
    axes = torch.Tensor(axes).type(dtype)
    rand_axis = axes[None,:] * rand_axis

    rand_axis  = Fu.normalize( rand_axis, dim=1, p=2 )
    rand_angle = torch.ones( N ).type(dtype).uniform_(0,max_rot_angle)
    R_ss_rand  = rand_axis * rand_angle[:,None]
    R_rand     = so3_exponential_map(R_ss_rand)

    # if max_rot_angle < float(np.pi)-1:
    #     e_ = torch.eye(3).type(R_rand.type())
    #     angles = so3_geod_dist(e_[None,:,:].repeat(N,1,1),R_rand).acos()
    #     print( "rand rot angles: mu=%1.3f std=%1.3f" % (angles.mean(),angles.std()) )

    if get_ss:
        return R_rand, R_ss_rand
    else:
        return R_rand

def random_2d_rotation(size, dtype, max_angle):
		theta = (torch.rand(size).type(dtype) - 0.5) * 2 * max_angle
		sins = torch.sin(theta)
		coss = torch.cos(theta)

		return torch.stack((
			torch.stack((coss, -sins), dim=-1),
			torch.stack((sins,  coss), dim=-1),
		), dim=-2)

def so3_exponential_map(log_rot: torch.Tensor, eps: float = 0.0001):
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula.
    The conversion has a singularity around 0 which is handled by clamping
    controlled with the `eps` argument.

    Args:
        log_rot: batch of vectors of shape :math:`(\text{minibatch} , 3)`
        eps: a float constant handling the conversion singularity around 0

    Returns:
        batch of rotation matrices of shape :math:`(\text{minibatch} , 3 , 3)`

    Raises:
        ValueError if `log_rot` is of incorrect shape
    """

    _ , dim = log_rot.shape
    if dim != 3:
        raise ValueError('Input tensor shape has to be Nx3.')

    nrms   = (log_rot * log_rot).sum(1)
    phis   = torch.clamp(nrms, 0.).sqrt()
    phisi  = 1. / (phis+eps)
    fac1   = phisi * phis.sin()
    fac2   = phisi * phisi * (1. - phis.cos())
    ss     = hat(log_rot)

    R = fac1[:, None, None] * ss + \
        fac2[:, None, None] * torch.bmm(ss, ss) + \
        torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]

    # from old.functions import rotss2rot
    # R_ = rotss2rot(log_rot)
    # print((R-R_).abs().max())
    # import pdb; pdb.set_trace()
    
    return R


def so3_log_map(R, eps: float = 0.0001):
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)` which is handled
    by clamping controlled with the `eps` argument.
    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: A float constant handling the conversion singularity.
    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.
    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    phi = so3_rotation_angle(R)

    phi_valid = torch.clamp(phi.abs(), eps) * phi.sign()
    phi_valid = phi_valid + (phi_valid==0).type_as(phi_valid) * eps

    log_rot_hat = (phi_valid / 
        (2.0 * phi_valid.sin()))[:, None, None] * (R - R.permute(0, 2, 1))
    log_rot = hat_inv(log_rot_hat)

    return log_rot


def hat_inv(h: torch.Tensor):
    """
    Compute the inverse Hat operator [1] of a batch of 3x3 matrices.

    Args:
        h: batch of skew-symmetric matrices of shape :math:`(\text{minibatch}, 3, 3)`

    Returns:
        batch of 3d vectors of shape :math:`(\text{minibatch}, 3)`

    Raises:
        ValueError if `h` is of incorrect shape
        ValueError if `h` not skew-symmetric

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N , dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError('Input has to be a batch of 3x3 Tensors.')

    ss_diff = (h + h.permute(0, 2, 1)).abs().max()
    if float(ss_diff) > 1e-5:
        raise ValueError('One of input matrices not skew-symmetric.')

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v


def hat(v: torch.Tensor):
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: batch of vectors of shape :math:`(\text{minibatch} , 3)`

    Returns:
        batch of skew-symmetric matrices of shape :math:`(\text{minibatch}, 3 , 3)`

    Raises:
        ValueError if `v` is of incorrect shape

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N , dim = v.shape
    if dim != 3:
        raise ValueError('Input vectors have to be 3-dimensional.')

    h = v.new_zeros(N, 3, 3)

    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


if __name__=='__main__':
    a12 = torch.randn(10, 6)
    R = so3_6d_to_rot(a12)
