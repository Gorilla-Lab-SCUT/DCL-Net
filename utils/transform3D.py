import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import torch
import os
def normalize_vector(v, return_mag = False, np_or_torch="torch"):
    '''
    Description: normalize the vectors
    
    Args:
        v: a batch of vectors. np.array. [bs, dim]

        return: the normalized vector
    
    '''
    batch=v.shape[0]
    if np_or_torch == "np":
        v = torch.tensor(v).cuda()
    v_mag = torch.sqrt(v.pow(2).sum(1, keepdim=True))# batch 1
    v_mag = v_mag + 1e-8
    v     = v/v_mag
    if(return_mag==True):
        if np_or_torch == np:
            v     = v.detach().cpu().numpy()
            v_mag = v_mag.detach().cpu().numpy()
        return v, v_mag[:,0]
    else:
        if np_or_torch == np:
            v     = v.detach().cpu().numpy()
        return v
def cross_product(u, v, np_or_torch="torch"):
    '''
    Description: get the cross product of vectors u and v
    
    Args:
        u/v: np.array, [N, 3].
    
    '''
    if np_or_torch == "np":
        u = torch.tensor(u)
        v = torch.tensor(v)
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]

    out = torch.cat((i.reshape(batch,1), j.reshape(batch,1), k.reshape(batch,1)),1)  # batch*3
    if np_or_torch == "np":
        out = out.numpy()
    return out
def ortho6d2matrix(x_raw, y_raw, np_or_torch="torch"):
    '''
    Description: get the rotation matrix computed by the two orthored vectors
    
    Args:
    
    '''
    y = normalize_vector(y_raw, np_or_torch)
    z = cross_product(x_raw, y, np_or_torch)
    z = normalize_vector(z, np_or_torch)#batch*3
    x = cross_product(y,z, np_or_torch)#batch*3
    if np_or_torch=="torch":
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        z = z.unsqueeze(-1)
        matrix = torch.cat((x,y,z), dim=2)  #batch*3*3
    elif np_or_torch=="np":
        x = x.reshape(-1,3,1)
        y = y.reshape(-1,3,1)
        z = z.reshape(-1,3,1)
        matrix = np.concatenate((x,y,z), axis=2)  #batch*3*3
    return matrix

def quaternion_to_axis_angle(q):
    r"""convertion from quaternion to axis-angle
    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    Returns
    -------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`
    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    angle = 2 * torch.acos(q[..., 0].clamp(-1, 1))
    axis = torch.nn.functional.normalize(q[..., 1:], dim=-1)
    return axis, angle
def axis_angle_to_matrix(axis, angle):
    r"""conversion from axis-angle to matrix
    Parameters
    ----------
    axis : `torch.Tensor`
        tensor of shape :math:`(..., 3)`
    angle : `torch.Tensor`
        tensor of shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`
    """
    axis, angle = torch.broadcast_tensors(axis, angle[..., None])
    alpha, beta = xyz_to_angles(axis)
    R = angles_to_matrix(alpha, beta, torch.zeros_like(beta))
    Ry = matrix_y(angle[..., 0])
    return R @ Ry @ R.transpose(-2, -1)
def quaternion_to_matrix(q):
    r"""convertion from quaternion to matrix, only support tensor
    Parameters
    ----------
    q : `torch.Tensor`
        tensor of shape :math:`(..., 4)`
    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`
    """
    return axis_angle_to_matrix(*quaternion_to_axis_angle(q))
def translate_rotate(cloud, rot, trans, mode_rot="matrix", np_or_torch="torch"):
    '''
    Description: translate and then rotate the point cloud. You can specify the representation method of roatetion.
                 rot@(cloud+trans).

                 Only the mode of ortho6d can pass the grad!
    
    Args:
        cloud: the point cloud, np.array, [N, 3]
        rot  : the rotation. may be rotation matrix or quaternion or decoupled representation. np.array
        trans: the translation, np.array [1, 3]
        mode_rot: the representation method of rotation: (matrix, quat, ortho6d)

        return: the transformed points. np.array [N, 3]
    
    '''

    if mode_rot == "matrix":
        # rot: 
        rot_matrix = rot
    elif mode_rot == "quat":
        if np_or_torch == "np":
            r = R.from_quat(rot)
            rot_matrix = r.as_matrix()
        else:
            r = quaternion_to_matrix(rot)
    elif mode_rot == "ortho6d":
        rot_matrix = ortho6d2matrix(rot, np_or_torch)

    cloud_transformed = cloud + trans
    cloud_transformed = (rot_matrix @ cloud_transformed.transpose(0,1)).transpose(0,1)

    return cloud_transformed

dir_path = os.path.dirname(os.path.abspath(__file__))
_Jd = torch.load(os.path.join(dir_path, 'new_constants.pt'))
def _z_rot_mat(angle, l):
    r"""
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).
    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M
    
def matrix_x(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around X axis
    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([o, z, z], dim=-1),
        torch.stack([z, c, -s], dim=-1),
        torch.stack([z, s, c], dim=-1),
    ], dim=-2)


def matrix_y(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around Y axis
    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([c, z, s], dim=-1),
        torch.stack([z, o, z], dim=-1),
        torch.stack([-s, z, c], dim=-1),
    ], dim=-2)


def matrix_z(angle: torch.Tensor) -> torch.Tensor:
    r"""matrix of rotation around Z axis
    Parameters
    ----------
    angle : `torch.Tensor`
        tensor of any shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    c = angle.cos()
    s = angle.sin()
    o = torch.ones_like(angle)
    z = torch.zeros_like(angle)
    return torch.stack([
        torch.stack([c, -s, z], dim=-1),
        torch.stack([s, c, z], dim=-1),
        torch.stack([z, z, o], dim=-1)
    ], dim=-2)

def xyz_to_angles(xyz):
    r"""convert a point :math:`\vec r = (x, y, z)` on the sphere into angles :math:`(\alpha, \beta)`
    .. math::
        \vec r = R(\alpha, \beta, 0) \vec e_z
    Parameters
    ----------
    xyz : `torch.Tensor`
        tensor of shape :math:`(..., 3)`
    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    xyz = torch.nn.functional.normalize(xyz, p=2, dim=-1)  # forward 0's instead of nan for zero-radius
    xyz = xyz.clamp(-1, 1)

    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta

def angles_to_matrix(alpha, beta, gamma):
    r"""conversion from angles to matrix
    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    Returns
    -------
    `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    """
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    return matrix_y(alpha) @ matrix_x(beta) @ matrix_y(gamma)

def matrix_to_angles(R):
    r"""conversion from matrix to angles
    Parameters
    ----------
    R : `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`
    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    assert torch.allclose(torch.det(R), R.new_tensor(1))
    x = R @ R.new_tensor([0.0, 1.0, 0.0])
    a, b = xyz_to_angles(x)
    R = angles_to_matrix(a, b, torch.zeros_like(a)).transpose(-1, -2) @ R
    c = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c

def wigner_D(l, alpha, beta, gamma):
    r"""Wigner D matrix representation of :math:`SO(3)`.
    It satisfies the following properties:
    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`
    * :math:`D(\text{rotation around Y axis})` has some property that allows us to use FFT in `ToS2Grid`
    Code of this function has beed copied from `lie_learn <https://github.com/AMLab-Amsterdam/lie_learn>`_ made by Taco Cohen.
    Parameters
    ----------
    l : int
        :math:`l`
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.
    Returns
    -------
    `torch.Tensor`
        tensor :math:`D^l(\alpha, \beta, \gamma)` of shape :math:`(2l+1, 2l+1)`
    """
    if not l < len(_Jd):
        raise NotImplementedError(f'wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more')

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    batchsize = alpha.size(0)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    J = J.unsqueeze(0).expand(batchsize, 2*l+1, 2*l+1)

    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc

def D_from_angles(alpha, beta, gamma, l, k=None):
    r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`
    (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.
    Parameters
    ----------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\alpha` around Y axis, applied third.
    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\beta` around X axis, applied second.
    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
        Rotation :math:`\gamma` around Y axis, applied first.
    k : `torch.Tensor`, optional
        tensor of shape :math:`(...)`
        How many times the parity is applied.
    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 2l+1, 2l+1)`
    See Also
    --------
    o3.wigner_D
    Irreps.D_from_angles
    """
    if k is None:
        k = torch.zeros_like(alpha)

    p = 1 # ! check p = {-1, 1}, p=1 even parity, p=-1 odd parity

    alpha, beta, gamma, k = torch.broadcast_tensors(alpha, beta, gamma, k)
    return wigner_D(l, alpha, beta, gamma) * p**k[..., None, None]

def D_from_matrix(R, l):
    r"""Matrix of the representation, see `Irrep.D_from_angles`
    Parameters
    ----------
    l : int
        :math:`l`
    R : `torch.Tensor`
        tensor of shape :math:`(..., 3, 3)`
    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., 2l+1, 2l+1)`
    Examples
    -------
    >>> m = Irrep(1, -1).D_from_matrix(-torch.eye(3))
    >>> m.long()
    tensor([[-1,  0,  0],
            [ 0, -1,  0],
            [ 0,  0, -1]])
    """
    d = torch.det(R).sign()
    R = d[..., None, None] * R
    k = (1 - d) / 2
    return D_from_angles(*matrix_to_angles(R), l, k)