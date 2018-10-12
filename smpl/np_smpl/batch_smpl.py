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

import numpy as np
from platform import python_version
if python_version()[0] == '2':
    import cPickle as pickle
else:
    import _pickle as pickle

from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMPL(object):
    def __init__(self, pkl_path, joint_type='cocoplus', dtype=np.float32):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        if python_version()[0] == '2':
            with open(pkl_path, 'r') as f:
                dd = pickle.load(f)
        else:
            with open(pkl_path, 'rb') as f:
                dd = pickle.load(f, encoding='latin1')

        # Mean template vertices
        self.v_template = undo_chumpy(dd['v_template'])

        # Size of mesh [Number of vertices, 3], (6890, 3)
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis (shapedirs): (6980, 3, 10)
        # reshaped to (6980*3, 10), transposed to (10, 6980*3)
        self.shapedirs = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T

        # Regressor for joint locations given shape -> (24, 6890)
        # Transpose to shape (6890, 24)
        self.J_regressor = np.asarray(dd['J_regressor'].T.todense())

        # Pose blend shape basis: (6890, 3, 207)
        num_pose_basis = dd['posedirs'].shape[-1]

        # Pose blend pose basis is reshaped to (6890*3, 207)
        # posedirs is transposed into (207, 6890*3)
        self.posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights (6890, 24)
        self.weights = undo_chumpy(dd['weights'])

        # This returns 19 keypoints: 6890 x 19
        self.joint_regressor = np.asarray(dd['cocoplus_regressor'].T.todense())

        if joint_type == 'lsp':  # 14 LSP joints!
            self.joint_regressor = self.joint_regressor[:, :14]

        if joint_type not in ['cocoplus', 'lsp']:
            print('BAD!! Unknown joint type: %s, it must be either "cocoplus" or "lsp"' % joint_type)
            import ipdb
            ipdb.set_trace()

    def __call__(self, beta, theta, get_skin=False, name=None):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6980 x 3
        """

        num_batch = beta.shape[0]

        # 1. Add shape blend shapes
        #       matmul  : (N, 10) x (10, 6890*3) = (N, 6890*3)
        #       reshape : (N, 6890*3) -> (N, 6890, 3)
        #       v_shaped: (N, 6890, 3)
        v_shaped = np.reshape(
            a=np.matmul(beta, self.shapedirs),
            newshape=(-1, self.size[0], self.size[1])) + self.v_template

        # 2. Infer shape-dependent joint locations.
        # ----- J_regressor: (6890, 24)
        # ----- Jx (Jy, Jz): (N, 6890) x (6890, 24) = (N, 24)
        # --------------- J: (N, 24, 3)
        Jx = np.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = np.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = np.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = np.stack([Jx, Jy, Jz], axis=2)

        # 3. Add pose blend shapes
        # ------- theta    : (N, 72)
        # ------- reshape  : (N*24, 3)
        # ------- rodrigues: (N*24, 9)
        # -- Rs = reshape  : (N, 24, 3, 3)
        Rs = np.reshape(batch_rodrigues(np.reshape(theta, (-1, 3))), (-1, 24, 3, 3))
        # Ignore global rotation.
        #       Rs[:, 1:, :, :]: (N, 23, 3, 3)
        #           - np.eye(3): (N, 23, 3, 3)
        #          pose_feature: (N, 207)
        pose_feature = np.reshape(Rs[:, 1:, :, :] - np.eye(3), (-1, 207))

        # (N, 207) x (207, 6890*3) -> (N, 6890, 3)
        v_posed = np.reshape(
            np.matmul(pose_feature, self.posedirs),
            (-1, self.size[0], self.size[1])) + v_shaped

        # 4. Get the global joint location
        # ------- Rs is (N, 24, 3, 3),         J is (N, 24, 3)
        # ------- J_transformed is (N, 24, 3), A is (N, 24, 4, 4)
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)

        # 5. Do skinning:
        # ------- weights is (6890, 24)
        # ---------- tile is (N*6890, 24)
        # --- W = reshape is (N, 6890, 24)
        W = np.reshape(np.tile(self.weights, (num_batch, 1)), (num_batch, -1, 24))

        # ------ reshape A is (N, 24, 16)
        # --------- matmul is (N, 6890, 24) x (N, 24, 16) -> (N, 6890, 16)
        # -------- reshape is (N, 6890, 4, 4)
        T = np.reshape(
            np.matmul(W, np.reshape(A, (num_batch, 24, 16))),
            (num_batch, -1, 4, 4))

        # axis is 2, (N, 6890, 3) concatenate (N, 6890, 1) -> (N, 6890, 4)
        v_posed_homo = np.concatenate(
            [v_posed, np.ones((num_batch, v_posed.shape[1], 1))], 2)

        # expand_dims is (N, 6890, 4, 1)
        # --------- T is (N, 6890, 4, 4)
        # ---- matmul is (N, 6890, 4, 4) x (N, 6890, 4, 1) -> (N, 6890, 4, 1)
        v_homo = np.matmul(T, np.expand_dims(v_posed_homo, -1))

        # (N, 6890, 3)
        verts = v_homo[:, :, :3, 0]

        # Get cocoplus or lsp joints:
        joint_x = np.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = np.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = np.matmul(verts[:, :, 2], self.joint_regressor)
        joints = np.stack([joint_x, joint_y, joint_z], axis=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints
