"""Define an optimiser that, given a sequence of points common in a multiview camera system, optimises the camera parameters of the system."""

import torch
import numpy as np
import sys
sys.path.append(r"E:\IIB Project Data\smalst")
from smal_deformer_mod import SMALX, opts
nn = torch.nn

device = "cpu"

kp_to_joint = [14, 13, 12, 24, 23, 22, 10, 9, 8, 20, 19, 18, 25, 31, 34, 33]  # index = keypoint, value = joint number up to joint=15


class MultiCam(nn.Module):

    def __init__(self, data, img_shapes, discard_cams=[], device = "cpu"):
        """
        :param n_cams: int number of cameras in system
        :param data: array of size (n_frames, n_cams, n_keypoints, 2)
        :param device:
        :np array: img_shapes - (ncams, 2) array that gives width, height in pixels for each camera
        """

        nn.Module.__init__(self)

        n_frames, n_cams, n_kp, _ = data.shape

        self.n_cams = n_cams
        self.n_frames = n_frames
        self.n_kp = n_kp
        self.iters = 0 # number of iterations

        self.discard_cams = []

        self.valid_cams = [i for i in range(n_cams) if i not in discard_cams]

        self.data = torch.FloatTensor(data, device=device)
        # Produce mask of all keypoints with no data - keypoints which have x=0 or y=0.
        # TODO: improve so it looks at (x,y) pairs for zero data
        self.mask = self.data.clone().detach() != 0.0

        # Predicted global coordinates
        # self.pred_global = nn.Parameter(torch.full((n_frames, n_kp, 3), 0.0, device = device, requires_grad=True))

        # Predicted global coordinates based on SMAL Model.
        self.smal_model = SMALX(opts, n_batch=n_frames)

        # EXTRINSICS
        self.rots = nn.Parameter(torch.full((n_cams, 3), 0.0, device=device, requires_grad=True)) # Rot in global X, Y, Z for each cam
        self.trans = nn.Parameter(torch.full((n_cams, 3), 0.1, device=device, requires_grad=True))
        self.scale_factor = nn.Parameter(torch.full((n_cams,), 1.0, device=device, requires_grad=True))

        # INTRINSICS
        # focal params in form n_cams * [fx fy cx cy s]
        self.focal = nn.Parameter(torch.full((n_cams, 5), 1.0, device = device, requires_grad=True))

        # distortion params in form n_cams * [k1 k2 p1 p2 k3]
        self.distortion = nn.Parameter(torch.full((n_cams, 5), 0.0, device = device, requires_grad=True))

        # get distance of each coordinate from centre to work out radius
        self.img_size = torch.from_numpy(img_shapes)
        self.img_lengths = torch.mean(self.img_size, dim=1) # characteristic length of each image

        centres = self.img_size  / 2 # centres at (W/2, H/2)
        XC = self.data[..., 0].clone()
        YC = self.data[..., 1].clone()
        for cam in range(n_cams):
            XC[:, cam] -= centres[cam, 0]
            YC[:, cam] -= centres[cam, 1]

        self.r = (XC ** 2 + YC ** 2) ** 0.5
        self.r.requires_grad = True

        self.params = [self.rots, self.trans, self.scale_factor, self.focal,
                       self.smal_model.global_rot, self.smal_model.joint_rot, self.smal_model.trans, self.smal_model.multi_betas]

    def get_kp_locations(self):
        """Returns all keypoint locations of joint model. Works in batch (aka across all frames)"""

        verts, _ = self.smal_model.get_verts()
        # print(self.smal_model.J_regressor.shape, )
        joint_pos = torch.matmul(self.smal_model.J_regressor.T, verts)  # calculate joint positions from vertex locations

        # positions for each keypoint
        kp_pos = torch.zeros((self.n_frames, 20, 3))
        for kp in range(16):
            kp_pos[:, kp] = joint_pos[:, kp_to_joint[kp]]  # normal joints included in standard model
        # joints added to SMAL (nose, chin, r ear, l ear) in order
        for kp, vert in zip([16, 17, 18, 19], [1863, 26, 2124, 150]):
            kp_pos[:, kp] = verts[:, vert]

        for f in range(self.n_frames):
            kp_pos[f] += self.smal_model.trans[f]

        return kp_pos

    def forward(self):
        """Given multicam params, calculate current predicted dataset.

        Loss based on:
        - Distance (in pix) between pred and actual keypoints, as a fraction of image length (avg of width and height0
        - Inter frame loss

        Losses to add:
        - Penalty for difference in relative spread of pred_kp from actual. This aims to prevent the cameras from moving
        a long distance away to get all the kp at a point in the image that minimises the LSE
        - Large penalty for any predicted kp outside the image bounds (if the kp is actually labelled)
        """

        # for now, no distortion

        all_losses = {} # dict of loss_type: [array of loss vals]

        pred_locals = torch.zeros((self.n_frames, self.n_cams, self.n_kp, 2))

        # GET PRED GLOBAL
        pred_global = self.get_kp_locations()[:, :self.n_kp] # trim to only desired number of keypoints
        pred_global = torch.cat([pred_global, torch.ones(self.n_frames, self.n_kp, 1)], dim = 2) # add row of ones for dimensional consistency (See equations)

        self.pred_global = pred_global[..., :3]
        pred_global_com = torch.mean(pred_global[..., :3], dim=1)


        # For each camera, convert pred_global into camera coordinates, and calculate loss between pred and actual
        for cam in self.valid_cams:

            R = torch.eye(3) # rotation matrix

            for axis, rot in enumerate(self.rots[cam]):
                axis_left, axis_right = (axis-1)%3, (axis+1)%3
                this_R = torch.full((3,3), 0.0)
                this_R[axis, axis] = 1

                this_R[axis_right, axis_right] = torch.cos(rot)
                this_R[axis_left, axis_left] = torch.cos(rot)
                this_R[axis_right, axis_left] = -torch.sin(rot)
                this_R[axis_left, axis_right] = torch.sin(rot)

                R = torch.mm(R, this_R)

            # Convert to correct 4x4 matrix [ R | t, 0 | 1]
            ext = torch.cat([R, self.trans[cam].view(1,3)], dim=0)
            # ext = torch.cat([ext, torch.FloatTensor([[0, 0, 0, 1]])])

            fx, fy, cx, cy, s = self.focal[cam]
            E = ext
            K = torch.FloatTensor([[fx, 0, 0], [s, fy, 0], [cx, cy, 1]])
            P = torch.mm(E, K) * self.scale_factor[cam]
            # print(E)
            # raise ValueError

            pixel_coords = torch.matmul(pred_global.view(-1, 4), P)

            # TODO: FIGURE OUT WAY TO MANAGE RADIAL & TANGENTIAL DISTORTION
            # k1, k2, p1, p2, k3 = self.distortion[cam]
            # print(self.r.requires_grad)
            # x, y, r = pixel_coords[..., 0], pixel_coords[..., 1], self.r[:, cam].flatten()
            # pixel_coords[..., :2] *= (1 + k1 * r ** 2)
            # x, y = x*(1+ k1 * r**2 + k2 * r**4 + k3 * r**6), y*(1+ k1 * r**2 + k2 * r**4 + k3 * r**6) # radial distortion
            # x, y = x+2*p1*x*y + p2*(r**2 + 2*x**2), y + p1*(r**2 + 2*y**2) + 2*p2*x*y
            # pixel_coords[..., 0] = x

            predict = pixel_coords.reshape(self.n_frames, self.n_kp, 3)[..., :2]

            pred_locals[:, cam] = predict

            target = self.data[:, cam]

            # mask any values for which (x,y) == [0,0] - or point is not labelled
            mask = self.mask[:, cam]
            img_length = self.img_lengths[cam]
            # kp_loss = (torch.sum(((predict - target) * mask) ** 2.0) / (torch.sum(mask)*1)) ** 0.5

            kp_loss = ((torch.sum( ( (predict - target) * mask) ** 2, dim = 2)).sum() / torch.sum(mask)) **0.5
            kp_loss *= 1/img_length # normalise by image length

            # ^2 loss for number of pixels outside of the range of the image (if the keypoint is labelled in the image).
            u_bounds = predict - self.img_size[cam] # positive values are out of the upper bounds of the image
            l_bounds = - target # positive values are out of the lower bounds of the image
            bounds_loss = ((torch.max(torch.zeros_like(u_bounds), u_bounds)*mask)**2).sum() + \
                          ((torch.max(torch.zeros_like(l_bounds), l_bounds)*mask)**2).sum()

            # rms loss for difference between variance of keypoints in pred and actual
            target_var = self.data[:, cam].std(dim=1)

            #var_loss = 0.001*nn.functional.mse_loss(predict.std(dim=1), target_var) ** 0.5
            #print(var_loss)

            ## RMS LOSS FOR DIFFERENCES IN POSE AND TRANS.
            # this if statement prevents from running on initial stage, to prevent infinite gradients
            weightings = [.1, .1, .1]  # weightings of theta / global rot / global trans
            smal_vars = [self.smal_model.joint_rot, self.smal_model.global_rot, self.smal_model.trans]
            if 100 < self.iters < 500:
                loss_time = 0
                for smal_var, weight in zip(smal_vars, weightings):
                    var_diff = smal_var[1:] - smal_var[:-1]
                    loss_time += weight * (var_diff**2).mean()**0.5
            else:
                loss_time = 0

            for name, loss in [("keypoint", kp_loss), ("time", loss_time)]:
                # for name, loss in [("keypoint", kp_loss), ("bounds", bounds_loss), ("variance", var_loss)]:
                all_losses[name] = all_losses.get(name, []) + [loss] # add loss to dict

        self.pred_local = pred_locals # store local camera predictions
        self.iters += 1

        # Return sum of all losses
        return sum([sum(ls) for ls in all_losses.values()]), all_losses


