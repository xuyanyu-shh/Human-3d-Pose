""" 
Util functions implementing the camera

@@batch_orth_proj_idrot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def batch_orth_proj_idrot(X, camera, device="cpu"):
    """
    X is N x num_points x 3
    camera is N x 3
    same as applying orth_proj_idrot to each N 
    """

    # TODO check X dim size.

    # # reshape (N, 3) -> (N, 1, 3)
    # camera = np.reshape(camera, (-1, 1, 3))
    #
    # # X_trans is (N, num_points, 2)
    # X_trans = X[:, :, :2] + camera[:, :, 1:]
    #
    # shape = X_trans.shape
    #
    # # reshape X_trans, (N, num_points * 2)
    # # --- * operation, (N, 1) x (N, num_points * 2) -> (N, num_points * 2)
    # # ------- reshape, (N, num_points, 2)
    #
    # return np.reshape(
    #     camera[:, :, 0] * np.reshape(X_trans, (shape[0], -1)), shape)

    # reshape (N, 3) -> (N, 1, 3)
    camera = camera.view(-1, 1, 3)

    # X_trans is (N, num_points, 2)
    X_trans = X[:, :, :2] + camera[:, :, 1:]

    shape = X_trans.shape

    # reshape X_trans, (N, num_points * 2)
    # --- * operation, (N, 1) x (N, num_points * 2) -> (N, num_points * 2)
    # ------- reshape, (N, num_points, 2)

    return (camera[:, :, 0] * torch.reshape(X_trans, (shape[0], -1))).view(shape)
