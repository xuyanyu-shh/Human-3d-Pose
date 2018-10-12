# -*- coding: utf-8 -*-
# @Time    : 2018/10/12 4:41 PM
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

# base
import cv2
import os
import sys
import h5py
import numpy as np

# pytorch
import torch
import torch.nn as nn
import visdom

# custom
from tools.visdom3d import Visdom3d
from global_utils.api_compatibility import load_pickle_file
from smpl.pt_smpl.batch_smpl import SMPL
import neural_renderer
import tools.beautiful_output

# (69, )
MEAN_POSE = [
    -0.028459642415367765, 0.08563495230516475, 0.010893310751686274, -0.02556717563575722, -0.07126141726018405, 0.011916545344739134,
    -0.014334669512208945, 0.025287648409039545, -0.012713923522915593, -0.06925146088868198, -0.07694441421705589, -0.016044732344857416,
    -0.059927550252285765, 0.0306949924822848, -0.0015415111553419981, 0.031111794682313502, -0.022445615406321173, 0.041647071710290355,
    0.002025386233351077, 0.1696618331013911, 0.0870693944715315, -0.02807943010619272, -0.22356410707920113, -0.05709368301174937,
    -0.0486720540867594, 0.0007454933442330367, -0.01771736729532008, 0.004103324898622594, 0.12722273895943867, -0.09341871928309027,
    0.07265922429632879, -0.04121701416552643, -0.031021408655868547, -0.02284029177752916, -0.017027063195089755, -0.03271392701559076,
    -0.09121819774354978, 0.05030192370933572, -0.31858401651095763, -0.10690893008590782, 0.02390829229617488, 0.32572590518927824, 0.063154631190144,
    0.02571055313674613, 0.03338427241664041, 0.11138716910178505, -0.17030753147695354, -0.8117714863988047, 0.16492615249163936, 0.1003683838645389,
    0.8131148455424325, -0.09989537814609242, -0.06107996135342513, 0.1295532344950452, -0.1979823108014614, 0.08587578712581728, -0.1490851749294026,
    0.022997559542242797, 0.07595730868851289, -0.13840055502625648, 0.05213788722381248, -0.07001335517289746, 0.10119602318858596,
    0.1354746450030287, 0.0194249219865765, 0.12720356446438308, 0.08819458877340475, -0.0042133696308108565, -0.08734498047087008
]


def intrinsic_mtx(f, c):
    """
    Obtain intrisic camera matrix.
    Args:
        f: np.array, 1 x 2, the focus lenth of camera, (fx, fy)
        c: np.array, 1 x 2, the center of camera, (px, py)
    Returns:
        - cam_mat: np.array, 3 x 3, the intrisic camera matrix.
    """
    return np.array([[f[1], 0, c[1]],
                     [0, f[0], c[0]],
                     [0, 0, 1]], dtype=np.float32)


def extrinsic_mtx(rt, t):
    """
    Obtain extrinsic matrix of camera.
    Args:
        rt: np.array, 1 x 3, the angle of rotations.
        t: np.array, 1 x 3, the translation of camera center.
    Returns:
        - ext_mat: np.array, 3 x 4, the extrinsic matrix of camera.
    """
    # R is (3, 3)
    R = cv2.Rodrigues(rt)[0]
    t = np.reshape(t, newshape=(3, 1))
    Rc = np.dot(R, t)
    ext_mat = np.hstack((R, -Rc))
    ext_mat = np.vstack((ext_mat, [0, 0, 0, 1]))
    ext_mat = ext_mat.astype(np.float32)
    return ext_mat


def get_project_mtx(f, c, rt, t):
    intrin_mat = intrinsic_mtx(f, c)
    extrin_mat = extrinsic_mtx(rt, t)

    iden_mat = np.hstack((np.identity(3), np.zeros((3, 1))))
    iden_mat = iden_mat.astype(np.float32)
    proj_mat = intrin_mat.dot(iden_mat).dot(extrin_mat)
    return proj_mat


def diff_heat_map(points, img_size, sigma=5):
    """
    :param points: N x num_points x 2
    :param sigma:
    :return: N x IAMGE_SIZE X IMAGE_SIZE
    """
    batch_size, num_keypoints = points.shape[0:2]

    X_MAP = torch.arange(img_size, dtype=torch.float32).repeat(img_size, 1).cuda()
    Y_MAP = X_MAP.permute(1, 0).cuda()

    X_MAPs = X_MAP.repeat(batch_size, 1, 1)
    Y_MAPs = Y_MAP.repeat(batch_size, 1, 1)

    heat_maps = []
    for i in range(num_keypoints):
        x = points[:, i, 0]
        y = points[:, i, 1]
        dist_map = (X_MAPs - x) ** 2 + (Y_MAPs - y) ** 2
        heat_map = torch.exp(-dist_map / (sigma ** 2))
        heat_maps.append(heat_map)
    heat_maps = torch.stack(heat_maps, dim=1)
    return heat_maps


def save_obj(vertices, faces):
    assert vertices.ndimension() == 2
    assert faces.ndimension() == 2

    faces = faces.detach().cpu().numpy()

    str_list = ['#\n', 'g\n']
    for vertex in vertices:
        str_list.append('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
    str_list.append('\n')

    for face in faces:
        str_list.append('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))
    str_list.append('s off\n')

    obj_str = "".join(str_list)

    return obj_str


def numpy_to_tensor(x, tensor_type='float', use_cuda=True):
    if tensor_type == 'float':
        x = torch.FloatTensor(x).cuda()
    return x


class SMPLModel(nn.Module):
    def __init__(self, img_size, camera_data, vis3d):
        super(SMPLModel, self).__init__()

        # camera
        self.camera_t = torch.load('runner_srcipts/camera.pt')['camera_t'][None, :]
        self.camera_f = torch.FloatTensor(camera_data['camera_f']).cuda()[None, :]
        self.camera_c = torch.FloatTensor(camera_data['camera_c']).cuda()[None, :]
        self.camera_rt = torch.FloatTensor(camera_data['camera_rt']).cuda()[None, :]
        self.dist_coeffs = torch.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(self.camera_f.shape[0], 1).cuda()

        self.img_size = img_size
        self.orig_size = camera_data['height']
        self.vis3d = vis3d

        # SMPL
        self.smpl = SMPL('hmr/models/neutral_smpl_with_cocoplus_reg.pkl', 'cuda', 'coco').cuda()
        initial_thetas = torch.FloatTensor(np.concatenate([[np.pi, 0, 0], MEAN_POSE])).cuda()
        initial_betas = torch.randn(10).cuda()
        self.faces = self.smpl.faces[None, :, :]
        self.theta = nn.Parameter(initial_thetas[None, :])
        self.betas = nn.Parameter(initial_betas[None, :])

        # renderer
        self.renderer = neural_renderer.Renderer(
            image_size=self.img_size,
            camera_f=self.camera_f,
            camera_c=self.camera_c,
            camera_rt=self.camera_rt,
            camera_t=self.camera_t,
            orig_size=self.orig_size,
            near=1.0,
            far=100.0,
            camera_mode='projection_by_params'
        )

    def forward(self):
        """
        betas : 1 x 10
        theta : 1 x 72
        :return:
        """
        vertices, joints3d, _ = self.smpl(self.betas, self.theta, get_skin=True)
        joints2d, image_points = neural_renderer.projection_by_params(
            vertices=joints3d,
            camera_f=self.camera_f,
            camera_c=self.camera_c,
            rt=self.camera_rt,
            t=self.camera_t,
            dist_coeffs=self.dist_coeffs,
            orig_size=self.orig_size,
            get_image_points=True
        )

        obj_str = save_obj(vertices[0], self.smpl.faces)
        data = {"obj": obj_str, 'iter': 0}
        self.vis3d.post(data)

        # pred_keypoints_heatmap = diff_heat_map(image_points, self.X_MAP, self.Y_MAP)
        pred_keypoints = image_points / self.orig_size * self.img_size
        pred_mask = self.renderer(vertices, self.faces, mode='silhouettes')
        pred_mask = torch.flip(pred_mask, dims=(1,))

        return pred_keypoints, pred_mask


class FineTune:
    def __init__(self):
        self.img_size = 256

        # data path
        self.dataset_root_path = '/datasets/human_pose/raw/people_snapshot_public'

        self.video_name = 'male-1-sport'
        self.masks_path = os.path.join(self.dataset_root_path, self.video_name, 'masks.hdf5')
        self.camera_path = os.path.join(self.dataset_root_path, self.video_name, 'camera.pkl')
        self.keypoints_path = os.path.join(self.dataset_root_path, self.video_name, 'keypoints.hdf5')
        self.camera_path = os.path.join(self.dataset_root_path, self.video_name, 'camera.pkl')
        self.frame_idx = 50

        # hyper-parameter
        self.lr = 1e-1
        self.iterations = 50000

        # result
        self.loss_list = []
        self.keypoints_loss_list = []
        self.mask_loss_list = []

        # visdom
        self.vis = visdom.Visdom(server='http://10.10.10.100', env='opt-smpl-pose', port=8097)
        self.vis3d = Visdom3d("http://10.10.10.100", env='test', port=8096)
        self.interval = 10

    def load_data(self):
        camera_data = load_pickle_file(self.camera_path)
        self.orig_size = camera_data['height']

        # camera data
        self.camera_data = load_pickle_file(self.camera_path)

        # mask / silhouette
        gt_mask_list = h5py.File(self.masks_path, 'r')
        gt_mask = gt_mask_list['masks'][self.frame_idx].astype(np.float32)
        gt_mask = cv2.resize(gt_mask, (self.img_size, self.img_size))
        self.gt_mask = numpy_to_tensor(gt_mask)

        # key points and key points heatmap
        gt_keypoints_list = h5py.File(self.keypoints_path)['keypoints']
        gt_keypoints = gt_keypoints_list[self.frame_idx].reshape(-1, 3) / self.orig_size * self.img_size
        gt_keypoints = numpy_to_tensor(gt_keypoints)[None, :, 0:2]
        self.gt_keypoints = gt_keypoints
        self.gt_keypoints_heatmap = diff_heat_map(gt_keypoints, self.img_size)

    def init_net(self):
        self.model = SMPLModel(self.img_size, self.camera_data, self.vis3d).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def show_result(self, iter, pred_keypoints_heatmap, pred_mask, loss, keypoints_loss, mask_loss):
        if iter % self.interval == 0:
            show_gt_keypoints_heatmap = self.gt_keypoints_heatmap.detach().sum(dim=1).cpu().numpy()
            show_gt_keypoints_heatmap[show_gt_keypoints_heatmap > 1] = 1
            show_gt_keypoints_heatmap = (show_gt_keypoints_heatmap * 255).astype(np.uint8)

            show_pred_keypoints_heatmap = pred_keypoints_heatmap.detach().sum(dim=1).cpu().numpy()
            show_pred_keypoints_heatmap[show_pred_keypoints_heatmap > 1] = 1
            show_pred_keypoints_heatmap = (show_pred_keypoints_heatmap * 255).astype(np.uint8)

            self.vis.image(self.gt_mask.detach(), win='gt_mask', opts={'title': 'gt_mask'})
            self.vis.image(pred_mask.detach(), win='pred_mask', opts={'title': 'pred_mask'})
            self.vis.image(show_pred_keypoints_heatmap, win='pred_keypoints_heatmap', opts={'title': 'pred_keypoints_heatmap'})
            self.vis.image(show_gt_keypoints_heatmap, win='gt_keypoints_heatmap', opts={'title': 'gt_keypoints_heatmap'})

            self.loss_list.append(loss)
            self.keypoints_loss_list.append(keypoints_loss)
            self.mask_loss_list.append(mask_loss)

            self.vis.line(Y=np.array(self.loss_list), X=np.arange(len(self.loss_list)), win='loss', opts={'title': 'loss'})
            self.vis.line(Y=np.array(self.keypoints_loss_list), X=np.arange(len(self.keypoints_loss_list)), win='keypoints_loss',
                          opts={'title': 'keypoints_loss'})
            self.vis.line(Y=np.array(self.mask_loss_list), X=np.arange(len(self.mask_loss_list)), win='mask_loss', opts={'title': 'mask_loss'})

            print('iter = {}, loss = {:.4f}'.format(iter, loss))
            print('keypoints_heatmap_loss = {}, mask_loss = {}'.format(keypoints_loss, mask_loss))

    def run(self):
        """
        keypoints: [1, 18, 2]
        :return:
        """
        self.init_net()

        for i in range(self.iterations):
            self.optimizer.zero_grad()

            # pred
            pred_keypoints, pred_mask = self.model()
            pred_keypoints_heatmap = diff_heat_map(pred_keypoints, self.img_size)

            # compute loss
            # keypoints_loss = 1e2 * ((pred_keypoints_heatmap - self.gt_keypoints_heatmap) ** 2).mean()
            keypoints_loss = 10 * (((pred_keypoints - self.gt_keypoints) / self.img_size) ** 2).mean()

            # mask_loss = ((pred_mask - self.gt_mask) ** 2).mean()
            mask_loss = torch.zeros(1).cuda()
            loss = keypoints_loss + mask_loss

            # backward
            loss.backward()
            self.optimizer.step()

            # show
            self.show_result(i, pred_keypoints_heatmap, pred_mask, loss.item(), keypoints_loss.item(), mask_loss.item())


def main():
    fine_tune = FineTune()
    fine_tune.load_data()
    fine_tune.run()


if __name__ == '__main__':
    main()
