
### NOW IN SMALST PROJECT. THIS AREA IS REDUNDANT. TODO DELETE.

"""Script that applies deformations to joints"""

from smalst.smal_model.smal_torch import SMAL
import torch
from smalst.smal_model.batch_lbs import batch_rodrigues

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False, opts=None):
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
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """
    if rotate_base:
        print('Flipping the SMPL coordinate frame!!!!')
        rot_x = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rot_x = torch.reshape(torch.repeat(rot_x, [N, 1]), [N, 3, 3]) # In tf it was tile
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)
    N = Rs.shape[0]

    def make_A(R, t):
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = torch.nn.functional.pad(R, (0,0,0,1,0,0))
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).cuda(device=opts.gpu_id)], 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(
            results[parent[i]], A_here)
        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = torch.cat([Js, torch.zeros([N, 35, 1, 1]).cuda(device=opts.gpu_id)], 2)
    init_bone = torch.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = torch.nn.functional.pad(init_bone, (3,0,0,0,0,0,0,0))
    A = results - init_bone

    return new_J, A

class SMALX(SMAL):

    def __init__(self):
        super().__init__(pkl_path=r"C:\Users\Ollie\Dropbox\Ollie\University\IIB\Project\Pipeline\smalst\smal_CVPR2017.pkl", opts =[])

    def __call__(self, beta, theta, trans=None, del_v=None, get_skin=True):

        if self.opts.use_smal_betas:
            nBetas = beta.shape[1]
        else:
            nBetas = 0

        # 1. Add shape blend shapes

        if nBetas > 0:
            if del_v is None:
                v_shaped = self.v_template + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas, :]),
                                                           [-1, self.size[0], self.size[1]])
            else:
                v_shaped = self.v_template + del_v + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas, :]),
                                                                   [-1, self.size[0], self.size[1]])
        else:
            if del_v is None:
                v_shaped = self.v_template.unsqueeze(0)
            else:
                v_shaped = self.v_template + del_v

                # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        Rs = torch.reshape(batch_rodrigues(torch.reshape(theta, [-1, 3]), opts=self.opts), [-1, 35, 3, 3])
        # Ignore global rotation.
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).cuda(device=self.opts.gpu_id), [-1, 306])

        v_posed = torch.reshape(
            torch.matmul(pose_feature, self.posedirs),
            [-1, self.size[0], self.size[1]]) + v_shaped

        # 4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, opts=self.opts)

        # 5. Do skinning:
        num_batch = theta.shape[0]

        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])

        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])),
            [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat(
            [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).cuda(device=self.opts.gpu_id)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch, 3)).cuda(device=self.opts.gpu_id)

        verts = verts + trans[:, None, :]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints


model = SMALX()

n_joints = 10
verts, *_ = SMALX(betas = np.zeros(10), thetas = np.zeros((n_joints, 3)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(verts.shape)
