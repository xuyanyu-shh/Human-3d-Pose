""" 
Tensorflow SMPL implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
from global_utils.api_compatibility import load_pickle_file


VERT_NOSE = 331
VERT_EAR_L = 3485
VERT_EAR_R = 6880
VERT_EYE_L = 2802
VERT_EYE_R = 6262


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMPL(nn.Module):
    def __init__(self, pkl_path, device="cpu", joint_type='cocoplus'):
        """
        pkl_path is the path to a SMPL model
        """
        super(SMPL, self).__init__()

        # -- Load SMPL params --
        dd = load_pickle_file(pkl_path)

        device = device
        # define faces
        self.register_buffer('faces', torch.from_numpy(undo_chumpy(dd['f']).astype(np.int32)).type(dtype=torch.int32))

        # Mean template vertices
        self.register_buffer('v_template', torch.FloatTensor(undo_chumpy(dd['v_template'])))
        # Size of mesh [Number of vertices, 3], (6890, 3)
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis (shapedirs): (6980, 3, 10)
        # reshaped to (6980*3, 10), transposed to (10, 6980*3)
        self.register_buffer('shapedirs', torch.FloatTensor(np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T))

        # Regressor for joint locations given shape -> (24, 6890)
        # Transpose to shape (6890, 24)
        self.register_buffer('J_regressor', torch.FloatTensor(
            np.asarray(dd['J_regressor'].T.todense())))

        # Pose blend shape basis: (6890, 3, 207)
        num_pose_basis = dd['posedirs'].shape[-1]

        # Pose blend pose basis is reshaped to (6890*3, 207)
        # posedirs is transposed into (207, 6890*3)
        self.register_buffer('posedirs', torch.FloatTensor(np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T))

        # indices of parents for each joints
        self.parents = np.array(dd['kintree_table'][0].astype(np.int32))

        # LBS weights (6890, 24)
        self.register_buffer('weights', torch.FloatTensor(undo_chumpy(dd['weights'])))

        # This returns 19 keypoints: 6890 x 19
        joint_regressor = torch.FloatTensor(
            np.asarray(dd['cocoplus_regressor'].T.todense()))

        if joint_type not in ['coco', 'cocoplus', 'lsp']:
            print('BAD!! Unknown joint type: %s, it must be either "cocoplus" or "lsp"' % joint_type)
            import ipdb
            ipdb.set_trace()

        if joint_type == 'lsp':  # 14 LSP joints!
            joint_regressor = joint_regressor[:, :14]
            self.register_buffer('joint_regressor', joint_regressor)
            self.joint_func = self.joints_cocoplus_lsp
        elif joint_type == 'cocoplus':
            self.register_buffer('joint_regressor', joint_regressor)
            self.joint_func = self.joints_cocoplus_lsp
        else:
            self.joint_func = self.joints_coco

    def forward(self, beta, theta, get_skin=False, rotate=False):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)
          get_skin: boolean, return skin or not
          rotate: boolean, rotate or not.

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """
        device = beta.device

        num_batch = beta.shape[0]

        # 1. Add shape blend shapes
        #       matmul  : (N, 10) x (10, 6890*3) = (N, 6890*3)
        #       reshape : (N, 6890*3) -> (N, 6890, 3)
        #       v_shaped: (N, 6890, 3)
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template

        # 2. Infer shape-dependent joint locations.
        # ----- J_regressor: (6890, 24)
        # ----- Jx (Jy, Jz): (N, 6890) x (6890, 24) = (N, 24)
        # --------------- J: (N, 24, 3)
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        # ------- theta    : (N, 72)
        # ------- reshape  : (N*24, 3)
        # ------- rodrigues: (N*24, 9)
        # -- Rs = reshape  : (N, 24, 3, 3)
        Rs = batch_rodrigues(theta.view(-1, 3), device=device).view(-1, 24, 3, 3)
        # Ignore global rotation.
        #       Rs[:, 1:, :, :]: (N, 23, 3, 3)
        #           - np.eye(3): (N, 23, 3, 3)
        #          pose_feature: (N, 207)
        pose_feature = (Rs[:, 1:, :, :] - torch.eye(3).to(device)).view(-1, 207)

        # (N, 207) x (207, 6890*3) -> (N, 6890, 3)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        # 4. Get the global joint location
        # ------- Rs is (N, 24, 3, 3),         J is (N, 24, 3)
        # ------- J_transformed is (N, 24, 3), A is (N, 24, 4, 4)
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents,
                                                                  rotate_base=rotate, device=device)

        # 5. Do skinning:
        # ------- weights is (6890, 24)
        # ---------- tile is (N*6890, 24)
        # --- W = reshape is (N, 6890, 24)
        W = self.weights.repeat(num_batch, 1).view(num_batch, -1, 24)

        # ------ reshape A is (N, 24, 16)
        # --------- matmul is (N, 6890, 24) x (N, 24, 16) -> (N, 6890, 16)
        # -------- reshape is (N, 6890, 4, 4)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        # axis is 2, (N, 6890, 3) concatenate (N, 6890, 1) -> (N, 6890, 4)
        v_posed_homo = torch.cat(
            [v_posed, torch.ones(num_batch, v_posed.shape[1], 1, dtype=torch.float32).to(device)], dim=2)

        # -unsqueeze_ is (N, 6890, 4, 1)
        # --------- T is (N, 6890, 4, 4)
        # ---- matmul is (N, 6890, 4, 4) x (N, 6890, 4, 1) -> (N, 6890, 4, 1)
        v_posed_homo = v_posed_homo.unsqueeze(-1)
        v_homo = torch.matmul(T, v_posed_homo)

        # (N, 6890, 3)
        verts = v_homo[:, :, :3, 0]

        joints = self.joint_func(verts)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def joints_coco(self, vertes):
        # J is (batch_size, 24, 3)
        J = self.J_transformed
        nose = vertes[:, VERT_NOSE, :]
        ear_l = vertes[:, VERT_EAR_L, :]
        ear_r = vertes[:, VERT_EAR_R, :]
        eye_l = vertes[:, VERT_EYE_L, :]
        eye_r = vertes[:, VERT_EYE_R, :]

        shoulders_m = torch.sum(J[:, [14, 13], :], dim=1) / 2.
        neck = J[:, 12, :] - 0.55 * (J[:, 12, :] - shoulders_m)

        joints = torch.stack((
            nose,
            neck,
            2.1 * (J[:, 14, :] - shoulders_m) + neck,
            J[:, 19, :],
            J[:, 21, :],
            2.1 * (J[:, 13, :] - shoulders_m) + neck,
            J[:, 18, :],
            J[:, 20, :],
            J[:, 2, :] + 0.38 * (J[:, 2, :] - J[:, 1, :]),
            J[:, 5, :],
            J[:, 8, :],
            J[:, 1, :] + 0.38 * (J[:, 1, :] - J[:, 2, :]),
            J[:, 4, :],
            J[:, 7, :],
            eye_r,
            eye_l,
            ear_r,
            ear_l,
        ), dim=1)

        return joints

    def joints_cocoplus_lsp(self, verts):
        # Get cocoplus or lsp joints:
        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        return joints
