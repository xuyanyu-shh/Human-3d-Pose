""" Util functions for SMPL
@@batch_skew
@@batch_rodrigues
@@batch_lrotmin
@@batch_global_rigid_transformation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np


def batch_skew(vec, batch_size=None, device="cpu"):
    """
    vec is N x 3, batch_size is int.

    e.g. r = [rx, ry, rz]
        skew(r) = [[ 0,    -rz,      ry],
                   [ rz,     0,     -rx],
                   [-ry,    rx,       0]]

    returns N x 3 x 3. Skew_sym version of each matrix.
    """

    if batch_size is None:
        batch_size = vec.shape[0]

    col_inds = np.array([1, 2, 3, 5, 6, 7], dtype=np.int32)
    
    indices = torch.from_numpy(np.reshape(
        np.reshape(np.arange(0, batch_size) * 9, [-1, 1]) + col_inds,
        newshape=(-1,))).to(device)

    updates = torch.stack(
            [
                -vec[:, 2],  vec[:, 1], vec[:, 2],
                -vec[:, 0], -vec[:, 1], vec[:, 0]
            ],
            dim=1
    ).view(-1).to(device)

    res = torch.zeros(batch_size * 9, dtype=vec.dtype).to(device)
    res[indices] = updates
    res = res.view(batch_size, 3, 3)

    return res


def batch_rodrigues(theta, device="cpu"):
    """
    Theta is N x 3

    rodrigues (from cv2.rodrigues):
    source: https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    input: r (3 x 1)
    output: R (3 x 3)

        angle = norm(r)
        r = r / angle

        skew(r) = [[ 0,    -rz,      ry],
                   [ rz,     0,     -rx],
                   [-ry,    rx,       0]]

        R = cos(theta * eye(3) + (1 - cos(theta)) * r * r.T + sin(theta) *  skew(r)
    """
    batch_size = theta.shape[0]

    # angle (batch_size, 1), r (batch_size, 3)
    angle = torch.norm(theta + 1e-8, p=2, dim=1, keepdim=True)
    r = torch.div(theta, angle)

    # angle (batch_size, 1, 1), r (batch_size, 3, 1)
    angle = angle.unsqueeze(-1)
    r = r.unsqueeze(-1)

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # outer (batch_size, 3, 3)
    outer = torch.matmul(r, r.permute(0, 2, 1))
    eyes = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    R = cos * eyes + (1 - cos) * outer + sin * batch_skew(r, batch_size=batch_size, device=device)

    return R


def batch_lrotmin(theta, device="cpu"):
    """ NOTE: not used bc I want to reuse R and this is simple.
    Output of this is used to compute joint-to-pose blend shape mapping.
    Equation 9 in SMPL paper.


    Args:
      pose: `Tensor`, N x 72 vector holding the axis-angle rep of K joints.
            This includes the global rotation so K=24

    Returns
      diff_vec : `Tensor`: N x 207 rotation matrix of 23=(K-1) joints with identity subtracted.,
    """

    # ignore global, N x 72
    theta = theta[:, 3:]
    # (N*23) x 3 x 3
    # reshape = contiguous + view
    Rs = batch_rodrigues(theta.reshape(-1, 3))
    eye = torch.eye(3).to(torch.eye(3))
    lrotmin = (Rs - eye).view(-1, 207)

    return lrotmin


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False, device="cpu"):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index

    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 x 4 x 4 relative joint transformations for LBS.
    """

    N = Rs.shape[0]
    if rotate_base:
        print('Flipping the SMPL coordinate frame!!!!')
        # rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
        # rot_x = np.reshape(np.tile(rot_x, [N, 1]), (N, 3, 3))
        # root_rotation = np.matmul(Rs[:, 0, :, :], rot_x)

        rot_x = torch.from_numpy(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                                          dtype=np.float32)).type(Rs.dtype).to(device)
        rot_x = rot_x.repeat(N, 1).view(N, 3, 3)
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)

    def make_A(R, t):
        """
        Composite homogeneous matrix.
        Args:
            R: N x 3 x 3 rotation matrix.
            t: N x 3 x 1 translation vector.

        Returns:
            homogeneous matrix N x 4 x 4.
        """

        # # Rs is N x 3 x 3, ts is N x 3 x 1
        # R_homo = np.pad(R, [[0, 0], [0, 1], [0, 0]], mode='constant')
        # t_homo = np.concatenate([t, np.ones((N, 1, 1))], 1)
        # return np.concatenate([R_homo, t_homo], 2)

        # Pad to (N, 4, 3)
        R_homo = F.pad(R, (0, 0, 0, 1, 0, 0), mode='constant', value=0)
        # Concatenate to (N, 4, 1)
        t_homo = torch.cat([t, torch.ones(N, 1, 1, dtype=Rs.dtype).to(device)], dim=1)
        return torch.cat([R_homo, t_homo], dim=2)

    # root_rotation: (N, 3, 3), Js[:, 0]: (N, 3, 1)
    # ---------- A0: (N, 4, 4)
    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    # N x 24 x 4 x 4
    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---

    # Js_w0: (N, 24, 4, 1)
    Js_w0 = torch.cat([Js, torch.zeros(N, 24, 1, 1, dtype=Rs.dtype).to(device)], dim=2)

    # init_bone: (N, 24, 4, 1) = (N, 24, 4, 4) x (N, 24, 4, 1)
    init_bone = torch.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = F.pad(init_bone, (3, 0, 0, 0, 0, 0, 0, 0), mode='constant', value=0)
    A = results - init_bone

    return new_J, A

