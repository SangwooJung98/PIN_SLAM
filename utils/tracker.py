#!/usr/bin/env python3
# @file      tracker.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import math

import numpy as np
import open3d as o3d
import torch
from rich import print
from tqdm import tqdm

from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.tools import color_to_intensity, get_gradient, get_time, transform_torch


class Tracker:
    def __init__(
        self,
        config: Config,
        neural_points: NeuralPoints,
        geo_decoder: Decoder,
        sem_decoder: Decoder,
        color_decoder: Decoder,
        radar_decoder: Decoder,
    ):

        self.config = config
        self.silence = config.silence
        self.neural_points = neural_points
        self.geo_decoder = geo_decoder
        self.sem_decoder = sem_decoder
        self.color_decoder = color_decoder
        self.radar_decoder = radar_decoder
        self.device = config.device
        self.dtype = config.dtype
        # NOTE: use torch.float64 for all the transformations and poses

        self.reg_local_map = True # for localization mode, set to False

        self.sdf_scale = config.logistic_gaussian_ratio * config.sigma_sigmoid_m

    # already under the scaled coordinate system
    def tracking(
        self,
        source_points,
        init_pose=None,
        source_colors=None,
        source_normals=None,
        source_semantics=None,
        source_radar=None,
        source_sdf=None,
        cur_ts=None,
        loop_reg: bool = False,
        vis_result: bool = False,
    ):

        if init_pose is None:
            T = torch.eye(4, dtype=torch.float64, device=self.device)
        else:
            T = init_pose  # to local frame

        cov_mat = None

        min_grad_norm = self.config.reg_min_grad_norm  # should be smaller than 1
        max_grad_norm = self.config.reg_max_grad_norm  # should be larger than 1
        if self.config.reg_GM_dist_m > 0:
            cur_GM_dist_m = self.config.reg_GM_dist_m
        else:
            cur_GM_dist_m = None
        if self.config.reg_GM_grad > 0:
            cur_GM_grad = self.config.reg_GM_grad
        else:
            cur_GM_grad = None
        lm_lambda = self.config.reg_lm_lambda
        iter_n = self.config.reg_iter_n
        term_thre_deg = self.config.reg_term_thre_deg
        term_thre_m = self.config.reg_term_thre_m

        max_valid_final_sdf_residual_cm = (
            self.config.surface_sample_range_m * self.config.final_residual_ratio_thre * 100.0
        )
        min_valid_ratio = 0.2
        if loop_reg:
            min_valid_ratio = 0.15

        max_increment_sdf_residual_ratio = 1.1
        eigenvalue_ratio_thre = 0.005
        min_valid_points = 30
        converged = False
        valid_flag = True
        last_sdf_residual_cm = 1e5

        source_point_count = source_points.shape[0]

        if not self.silence:
            print("# Source point for registeration :", source_point_count)

        if source_sdf is None:  # only use the surface samples (all zero)
            source_sdf = torch.zeros(source_point_count, device=self.device)

        for i in tqdm(range(iter_n), disable=self.silence):

            T01 = get_time()

            cur_points = transform_torch(source_points, T)  # apply transformation

            T02 = get_time()

            reg_result = self.registration_step(
                cur_points,
                source_normals,
                source_sdf,
                source_colors,
                source_radar,
                min_grad_norm,
                max_grad_norm,
                cur_GM_dist_m,
                cur_GM_grad,
                lm_lambda,
                (vis_result and converged),
            )

            (
                delta_T,
                cov_mat,
                eigenvalues,
                weight_point_cloud,
                valid_points_torch,
                sdf_residual_cm,
                photo_residual,
                radar_residual,
            ) = reg_result

            T03 = get_time()

            T = delta_T @ T

            # the sdf residual should not increase too much during the optimization
            if (
                sdf_residual_cm - last_sdf_residual_cm
            ) / last_sdf_residual_cm > max_increment_sdf_residual_ratio:
                if not self.silence:
                    print(
                        "[bold yellow](Warning) registration failed: wrong optimization[/bold yellow]"
                    )
                valid_flag = False
            else:
                last_sdf_residual_cm = sdf_residual_cm

            valid_point_count = valid_points_torch.shape[0]
            if (valid_point_count < min_valid_points) or (
                1.0 * valid_point_count / source_point_count < min_valid_ratio
            ):
                if not self.silence:
                    print(
                        "[bold yellow](Warning) registration failed: not enough valid points[/bold yellow]"
                    )
                    # print("test: " + str(valid_point_count) + " " + str(source_point_count))
                valid_flag = False

            if not valid_flag or converged:
                break

            rot_angle_deg = (
                rotation_matrix_to_axis_angle(delta_T[:3, :3]) * 180.0 / np.pi
            )
            tran_m = delta_T[:3, 3].norm()

            if (
                abs(rot_angle_deg) < term_thre_deg
                and tran_m < term_thre_m
                or i == iter_n - 2
            ):
                converged = True  # for the visualization (save the computation)

            T04 = get_time()

            # print("transformation time:", (T02 - T01) * 1e3)
            # print("reg time:", (T03 - T02) * 1e3)
            # print("judge time:", (T04 - T03) * 1e3)

        if not self.silence:
            print("# Valid source point             :", valid_point_count)
            print("Odometry residual (cm):", sdf_residual_cm)
            if photo_residual is not None:
                print("Photometric residual:", photo_residual)
            if radar_residual is not None:
                print("Radar residual:", radar_residual)

        if sdf_residual_cm > max_valid_final_sdf_residual_cm:
            if not self.silence:
                print(
                    "[bold yellow](Warning) registration failed: too large final residual[/bold yellow]"
                )
            valid_flag = False

        if eigenvalues is not None:
            min_eigenvalue = torch.min(eigenvalues).item()
            # print("Smallest eigenvalue:", min_eigenvalue)
            if (
                self.config.eigenvalue_check
                and min_eigenvalue < valid_point_count * eigenvalue_ratio_thre
            ):
                if not self.silence:
                    print(
                        "[bold yellow](Warning) registration failed: eigenvalue check failed[/bold yellow]"
                    )
                valid_flag = False

        if cov_mat is not None:
            cov_mat = cov_mat.detach().cpu().numpy()

        if not valid_flag and i < 10:  # NOTE: if not valid and without enough iters, just take the initial guess
            T = init_pose
            cov_mat = None
        
        
        
        
        
        # add NDT based odometry calculation at here (only for temporary test)
        
        # start with generating SDF map by using local neural map
        
        (
            sdf_pred,
            sdf_grad,
            _,
            _,
            _,
            _,
            _,
            mask,
            certainty,
            sdf_std,
        ) = self.query_source_points(
            self.neural_points.local_neural_points,
            self.config.infer_bs,
            True,
            True,
            False,
            False,
            False,
            False,
            query_locally=True,
            mask_min_nn_count=self.config.track_mask_query_nn_k,
        )
                       
        # check the center of local neural points
        # print("neural point center: ", self.neural_points.local_neural_points.mean(dim=0))
        # print("neural_points.local_neural_points form: ", self.neural_points.local_neural_points.shape)
        
        surface_points_from_sdf = filter_surface_points(self.neural_points.local_neural_points, sdf_pred, epsilon = 0.03)
                        
        # change the points to local frame (based on the initial pose)
        # initial pose is the pose of the previous frame (inverse the init_pose)
        
        surface_points_from_sdf = transform_torch(surface_points_from_sdf, torch.inverse(init_pose))
        
        print("surface full points shape: ", surface_points_from_sdf.shape)
        
        # leave only the points that are included in the area of source_points
        
        max_x = source_points[:, 0].max()
        min_x = source_points[:, 0].min()
        max_y = source_points[:, 1].max()
        min_y = source_points[:, 1].min()
        max_z = source_points[:, 2].max()
        min_z = source_points[:, 2].min()
        
        mask_source = (surface_points_from_sdf[:, 0] < max_x) & (surface_points_from_sdf[:, 0] > min_x) & (surface_points_from_sdf[:, 1] < max_y) & (surface_points_from_sdf[:, 1] > min_y) & (surface_points_from_sdf[:, 2] < max_z) & (surface_points_from_sdf[:, 2] > min_z)
        surface_points_from_sdf = surface_points_from_sdf[mask_source]
        
        print("front surface points shape: ", surface_points_from_sdf.shape)
        # print("surface point center: ", surface_points_from_sdf.mean(dim=0))   
        # print("surface point x min: ", surface_points_from_sdf[:, 0].min())
        # print("surface point x max: ", surface_points_from_sdf[:, 0].max())
        # print("surface point y min: ", surface_points_from_sdf[:, 1].min())
        # print("surface point y max: ", surface_points_from_sdf[:, 1].max())
        # print("surface point z min: ", surface_points_from_sdf[:, 2].min())
        # print("surface point z max: ", surface_points_from_sdf[:, 2].max())
        
        print("source points shape: ", source_points.shape)
        # print("source points center: ", source_points.mean(dim=0))
        # print("source point x min: ", source_points[:, 0].min())
        # print("source point x max: ", source_points[:, 0].max())
        # print("source point y min: ", source_points[:, 1].min())
        # print("source point y max: ", source_points[:, 1].max())
        # print("source point z min: ", source_points[:, 2].min())
        # print("source point z max: ", source_points[:, 2].max())
        
        ndt_voxel_size = 1
        # start measuring the time
        aT0 = get_time()
        local_ndt = create_ndt_from_pointcloud_fast(surface_points_from_sdf, ndt_voxel_size)
        aT1 = get_time()
        source_ndt = create_ndt_from_pointcloud_fast(source_points, ndt_voxel_size)
        aT2 = get_time()
        
        # PRINT the size of ndt
        print("local ndt size: ", len(local_ndt))
        print("local ndt time: ", (aT1 - aT0) * 1e3)
        print("source ndt size: ", len(source_ndt))
        print("source ndt time: ", (aT2 - aT1) * 1e3)
        
        
        
        
        
        
        
        
        
        
        

        return T, cov_mat, weight_point_cloud, valid_flag

    def query_source_points(
        self,
        coord,
        bs,
        query_sdf=True,
        query_sdf_grad=True,
        query_color=False,
        query_color_grad=False,
        query_radar=False,
        query_radar_grad=False,
        query_sem=False,
        query_mask=True,
        query_certainty=True,
        query_locally=True,
        mask_min_nn_count: int = 4,
    ):
        """query the sdf value, semantic label and marching cubes mask for points
        Args:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            bs: batch size for the inference
        Returns:
            sdf_pred: Ndim torch tenosr, signed distance value (scaled) at each query point
            sem_pred: Ndim torch tenosr, semantic label prediction at each query point
            mc_mask:  Ndim torch tenosr, marching cubes mask at each query point
        """

        sample_count = coord.shape[0]
        iter_n = math.ceil(sample_count / bs)

        if query_sdf:
            sdf_pred = torch.zeros(sample_count, device=coord.device)
            sdf_std = torch.zeros(sample_count, device=coord.device)
        else:
            sdf_pred = None
            sdf_std = None
        if query_sem:
            sem_pred = torch.zeros(sample_count, device=coord.device)
        else:
            sem_pred = None
        if query_color:
            color_pred = torch.zeros(
                (sample_count, self.config.color_channel), device=coord.device
            )
        else:
            color_pred = None
        if query_radar:
            radar_pred = torch.zeros(sample_count, 1, device=coord.device) # only for rcs (test)
        else:
            radar_pred = None
        if query_mask:
            mc_mask = torch.zeros(sample_count, device=coord.device, dtype=torch.bool)
        else:
            mc_mask = None
        if query_sdf_grad:
            sdf_grad = torch.zeros((sample_count, 3), device=coord.device)
        else:
            sdf_grad = None
        if query_color_grad:
            color_grad = torch.zeros(
                (sample_count, self.config.color_channel, 3), device=coord.device
            )
        else:
            color_grad = None
        if query_radar_grad:
            radar_grad = torch.zeros(
                (sample_count, 1, 3), device=coord.device
            ) # size of second term is the additional channel such as intensity. just add rcs for test
        else:
            radar_grad = None
        if query_certainty:
            certainty = torch.zeros(sample_count, device=coord.device)
        else:
            certainty = None

        for n in range(iter_n):
            head = n * bs
            tail = min((n + 1) * bs, sample_count)
            batch_coord = coord[head:tail, :]
            if query_sdf_grad or query_color_grad or query_radar_grad:
                batch_coord.requires_grad_(True)

            (
                batch_geo_feature,
                batch_color_feature,
                batch_radar_feature,
                weight_knn,
                nn_count,
                batch_certainty,
            ) = self.neural_points.query_feature(
                batch_coord,
                training_mode=False,
                query_locally=query_locally,
                query_color_feature=query_color,
                query_radar_feature=query_radar,
            )  # inference mode

            # print(weight_knn)
            if query_sdf:
                batch_sdf = self.geo_decoder.sdf(batch_geo_feature)
                if not self.config.weighted_first:
                    # batch_sdf = torch.sum(batch_sdf * weight_knn, dim=1).squeeze(1)
                    # print(batch_sdf.squeeze(-1))

                    batch_sdf_mean = torch.sum(batch_sdf * weight_knn, dim=1)  # N, 1
                    batch_sdf_var = torch.sum(
                        (weight_knn * (batch_sdf - batch_sdf_mean.unsqueeze(-1)) ** 2),
                        dim=1,
                    )
                    batch_sdf_std = torch.sqrt(batch_sdf_var).squeeze(1)
                    batch_sdf = batch_sdf_mean.squeeze(1)
                    sdf_std[
                        head:tail
                    ] = (
                        batch_sdf_std.detach()
                    )  # the std is a bit too large, figure out why

                if query_sdf_grad:
                    batch_sdf_grad = get_gradient(
                        batch_coord, batch_sdf
                    )  # use analytical gradient in tracking
                    sdf_grad[head:tail, :] = batch_sdf_grad.detach()
                sdf_pred[head:tail] = batch_sdf.detach()
            if query_sem:
                batch_sem_prob = self.sem_decoder.sem_label_prob(batch_geo_feature)
                if not self.config.weighted_first:
                    batch_sem_prob = torch.sum(batch_sem_prob * weight_knn, dim=1)
                batch_sem = torch.argmax(batch_sem_prob, dim=1)
                sem_pred[head:tail] = batch_sem.detach()
            if query_color:
                batch_color = self.color_decoder.regress_color(batch_color_feature)
                if not self.config.weighted_first:
                    batch_color = torch.sum(batch_color * weight_knn, dim=1)  # N, C
                if query_color_grad:
                    for i in range(self.config.color_channel):
                        batch_color_grad = get_gradient(batch_coord, batch_color[:, i])
                        color_grad[head:tail, i, :] = batch_color_grad.detach()
                color_pred[head:tail] = batch_color.detach()
            if query_radar:
                batch_radar = self.radar_decoder.regress_radar(batch_radar_feature)
                if not self.config.weighted_first:
                    batch_radar = torch.sum(batch_radar * weight_knn, dim=1) # N, 1
                if query_radar_grad:
                    batch_radar_grad = get_gradient(batch_coord, batch_radar)
                    radar_grad[head:tail, 0, :] = batch_radar_grad.detach()
                radar_pred[head:tail] = batch_radar.detach()
            if query_mask:
                mc_mask[head:tail] = nn_count >= mask_min_nn_count
            if query_certainty:
                certainty[head:tail] = batch_certainty.detach()

        return (
            sdf_pred,
            sdf_grad,
            color_pred,
            color_grad,
            radar_pred,
            radar_grad,
            sem_pred,
            mc_mask,
            certainty,
            sdf_std,
        )

    def registration_step(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        sdf_labels: torch.Tensor,
        colors: torch.Tensor,
        radar: torch.Tensor,
        min_grad_norm,
        max_grad_norm,
        GM_dist=None,
        GM_grad=None,
        lm_lambda=0.0,
        vis_weight_pc=False,
    ):  # if lm_lambda = 0, then it's Gaussian Newton Optimization

        T0 = get_time()

        colors_on = colors is not None and self.config.color_on
        photo_loss_on = self.config.photometric_loss_on and colors_on
        
        radar_on = radar is not None and self.config.use_radar_intensity
        radar_loss_on = self.config.radar_loss_on and radar_on
        
        (
            sdf_pred,
            sdf_grad,
            color_pred,
            color_grad,
            radar_pred,
            radar_grad,
            _,
            mask,
            certainty,
            sdf_std,
        ) = self.query_source_points(
            points,
            self.config.infer_bs,
            True,
            True,
            colors_on,
            photo_loss_on,
            radar_on,
            radar_loss_on,
            query_locally=self.reg_local_map,
            mask_min_nn_count=self.config.track_mask_query_nn_k,
        )  # fixme

        T1 = get_time()

        grad_norm = sdf_grad.norm(dim=-1, keepdim=True).squeeze()  # unit: m

        grad_unit = sdf_grad / grad_norm.unsqueeze(-1)

        min_certainty = 5.0
        sdf_pred_abs = torch.abs(sdf_pred)

        max_sdf = self.config.surface_sample_range_m * self.config.max_sdf_ratio
        max_sdf_std = self.config.surface_sample_range_m * self.config.max_sdf_std_ratio

        valid_idx = (
            mask
            & (grad_norm < max_grad_norm)
            & (grad_norm > min_grad_norm)
            & (sdf_std < max_sdf_std)
            # & (sdf_pred_abs < max_sdf)
        )
        
        # check how each condition affects the valid_idx
        # give number of invalid points that are filtered out from each condition
        # print("Number of invalid points filtered out by each condition:")
        # print("mask: ", torch.sum(~mask))
        # print("grad_norm < max_grad_norm: ", torch.sum(~(grad_norm < max_grad_norm)))
        # print("grad_norm > min_grad_norm: ", torch.sum(~(grad_norm > min_grad_norm)))
        # print("sdf_std < max_sdf_std: ", torch.sum(~(sdf_std < max_sdf_std)))

        valid_points = points[valid_idx]
        valid_point_count = valid_points.shape[0]

        if valid_point_count < 10:
            T = torch.eye(4, device=points.device, dtype=torch.float64)
            return T, None, None, None, valid_points, 0.0, 0.0
        if vis_weight_pc:
            invalid_points = points[~valid_idx]

        grad_norm = grad_norm[valid_idx]
        sdf_pred = sdf_pred[valid_idx]
        sdf_grad = sdf_grad[valid_idx]
        sdf_labels = sdf_labels[valid_idx]

        # certainty not used here
        # certainty = certainty[valid_idx]
        # std also not used
        # sdf_std = sdf_std[valid_idx]
        # std_mean = sdf_std.mean()

        valid_grad_unit = grad_unit[valid_idx]
        invalid_grad_unit = grad_unit[~valid_idx]

        if normals is not None:
            valid_normals = normals[valid_idx]

        grad_anomaly = grad_norm - 1.0  # relative to 1
        if (
            self.config.reg_dist_div_grad_norm
        ):  # fix the overshot, as wiesmann2023ral (not enabled)
            sdf_pred = sdf_pred / grad_norm

        sdf_residual = sdf_pred - sdf_labels

        sdf_residual_mean_cm = torch.mean(torch.abs(sdf_residual)).item() * 100.0

        # print("\nOdometry residual (cm):", sdf_residual_mean_cm)
        # print("Valid point count:", valid_point_count)

        weight_point_cloud = None

        # calculate the weights
        # we use the Geman-McClure robust weight here (https://arxiv.org/pdf/1810.01474.pdf)
        # note that there's a mistake that in this paper, the author multipy an additional k at the numerator
        w_grad = (
            1.0
            if GM_grad is None
            else ((GM_grad / (GM_grad + grad_anomaly**2)) ** 2).unsqueeze(1)
        )
        w_res = (
            1.0
            if GM_dist is None
            else ((GM_dist / (GM_dist + sdf_residual**2)) ** 2).unsqueeze(1)
        )

        w_normal = (
            1.0
            if normals is None
            else (
                0.5 + torch.abs((valid_normals * valid_grad_unit).sum(dim=1))
            ).unsqueeze(1)
        )

        w_certainty = 1.0

        w_color = 1.0
        if colors_on:  # how do you know the channel number
            colors = colors[valid_idx, : self.config.color_channel]  # fix channel
            color_pred = color_pred[valid_idx, : self.config.color_channel]

            if self.config.color_channel == 3 and self.config.color_on:
                colors = color_to_intensity(colors)
                color_pred = color_to_intensity(color_pred)

            if (
                photo_loss_on
            ):  # if color already in loss, we do not need the color weight
                color_grad = color_grad[valid_idx, : self.config.color_channel]

                if self.config.color_channel == 3 and self.config.color_on:
                    color_grad = color_to_intensity(color_grad)

            elif self.config.consist_wieght_on:  # color (intensity) consistency weight
                w_color = torch.exp(
                    -torch.mean(torch.abs(colors - color_pred), dim=-1)
                ).unsqueeze(
                    1
                )  # color in [0,1]
                # w_color[colors==0] = 1.
        
        w_radar = 1.0
        if radar_on: # 여기서 radar point의 어떤 channel을 활용할지 정하면 됨. (use_radar_intensity version)
            # first, just test for the rcs value...
            # radar (N, 4) -> doppler, rcs,  doppler_uncertainty, cos_similarity
            
            radar = radar[valid_idx, 1].unsqueeze(1) # test for rcs value
            radar_pred = radar_pred[valid_idx]
            
            if radar_loss_on:
                radar_grad = radar_grad[valid_idx]
                
                # print(radar_grad.shape)
            elif self.config.consist_wieght_on:
                w_radar = torch.exp(
                    -torch.mean(torch.abs(radar - radar_pred), dim=-1)
                ).unsqueeze(
                    1
                ) # radar in [0,1]
                # w_radar[radar==0] = 1.

        # sdf standard deviation as the weight (not used)
        # w_std = (std_mean / sdf_std).unsqueeze(1)
        w_std = 1.0

        # print(w_color)
        # w = w_res * w_grad * w_normal * w_color * w_certainty * w_std
        w = w_res * w_grad * w_normal * w_color * w_radar * w_certainty * w_std
        if not isinstance(w, (float)):
            w /= 2.0 * torch.mean(w)  # normalize weight for visualization

        T2 = get_time()

        color_residual_mean = None
        radar_residual_mean = None
        if photo_loss_on:
            color_residual = color_pred - colors
            color_residual_mean = torch.mean(torch.abs(color_residual)).item()
            T = implicit_color_reg(
                valid_points,
                sdf_grad,
                sdf_residual,
                colors,
                color_grad,
                color_residual,
                w,
                w_photo_loss=self.config.photometric_loss_weight,
                lm_lambda=lm_lambda,
            )
            cov_mat = None
            eigenvalues = None
        elif radar_loss_on:
            radar_residual = radar_pred - radar
            radar_residual_mean = torch.mean(torch.abs(radar_residual)).item()
            T = implicit_radar_reg(
                valid_points,
                sdf_grad,
                sdf_residual,
                radar,
                radar_grad,
                radar_residual,
                w,
                w_radar_loss=self.config.radar_loss_weight,
                lm_lambda=lm_lambda,
            )
            cov_mat = None
            eigenvalues = None
        else:
            T, cov_mat, eigenvalues = implicit_reg(
                valid_points,
                sdf_grad,
                sdf_residual,
                w,
                lm_lambda=lm_lambda,
                require_cov=vis_weight_pc,
                require_eigen=vis_weight_pc,
            )  # only get metrics for the last iter

        T3 = get_time()

        if vis_weight_pc:  # only for the visualization
            # visualize the filtered points and also the weights
            valid_points_numpy = valid_points.detach().cpu().numpy()
            invalid_points_numpy = invalid_points.detach().cpu().numpy()
            points_numpy = np.vstack((valid_points_numpy, invalid_points_numpy)).astype(
                np.float64
            )  # for faster Vector3dVector

            weight_point_cloud = o3d.geometry.PointCloud()
            weight_point_cloud.points = o3d.utility.Vector3dVector(points_numpy)

            # w /= torch.max(w) # normalize to [0-1]

            weight_numpy = w.squeeze(1).detach().cpu().numpy()
            weight_colors = np.zeros_like(valid_points_numpy)
            weight_colors[:, 0] = weight_numpy  # set as the red channel
            invalid_colors = np.zeros_like(invalid_points_numpy)
            invalid_colors[:, 2] = 1.0
            colors_numpy = np.vstack((weight_colors, invalid_colors)).astype(
                np.float64
            )  # for faster Vector3dVector
            weight_point_cloud.colors = o3d.utility.Vector3dVector(colors_numpy)

            valid_normal_numpy = valid_grad_unit.detach().cpu().numpy()
            invalid_normal_numpy = invalid_grad_unit.detach().cpu().numpy()
            normal_numpy = np.vstack((valid_normal_numpy, invalid_normal_numpy)).astype(
                np.float64
            )

            # normal_numpy = normals.detach().cpu().numpy().astype(np.float64)

            weight_point_cloud.normals = o3d.utility.Vector3dVector(normal_numpy)

            # print("\n# Valid source point: ", valid_point_count)
            # print("Odometry residual (cm):", sdf_residual_mean_cm)
            # if photo_loss_on:
            #     print("Photometric residual:", color_residual_mean)

        T4 = get_time()

        # print("time for querying        :", (T1-T0) * 1e3) # time mainly spent here
        # print("time for weight          :", (T2-T1) * 1e3) # kind of fast
        # print("time for registration    :", (T3-T2) * 1e3) # kind of fast
        # print("time for vis             :", (T4-T3) * 1e3) # negligible

        return (
            T,
            cov_mat,
            eigenvalues,
            weight_point_cloud,
            valid_points,
            sdf_residual_mean_cm,
            color_residual_mean,
            radar_residual_mean,
        )


# function adapted from LocNDF by Louis Wiesmann
def implicit_reg(
    points,
    sdf_grad,
    sdf_residual,
    weight,
    lm_lambda=0.0,
    require_cov=False,
    require_eigen=False,
):
    """
    One step point-to-implicit model registration using LM optimization.

    Args:
        points (`torch.tensor'):
            Current transformed source points in the coordinate system of the implicit distance field
            with the shape of [N, 3]
        sdf_grad (`torch.tensor'):
            The gradient of predicted SDF
            with the shape of [N, 3]
        sdf_residual (`torch.tensor'):
            SDF predictions at the positions of the points
            with the shape of [N, 1]
        weight (`torch.tensor'):
            Point-wise weight for the optimization
            with the shape of [N, 1]
        lm_lambda: (`float`):
            Lambda damping factor for LM optimization

    Returns:
        T_mat (`torch.tensor'):
            4 by 4 transformation matrix of this iteration of the registration
        cov_mat (`torch.tensor'):
            6 by 6 covariance matrix for the registration
        eigenvalues (`torch.tensor'):
            3 dim translation part of the eigenvalues for the registration degerancy check
    """
    # print("implicit_reg part")
    # print(points.shape)
    # print(sdf_grad.shape)
    cross = torch.cross(points, sdf_grad, dim=-1)  # N,3 x N,3
    J_mat = torch.cat(
        [cross, sdf_grad], -1
    )  # The Jacobian matrix # first rotation, then translation # N, 6
    # print(J_mat.shape)
    # print(weight.shape)
    N_mat = J_mat.T @ (
        weight * J_mat
    )  # approximate Hessian matrix # first rot, then tran # 6, 6

    if require_cov or require_eigen:
        N_mat_raw = N_mat.clone()

    # use LM optimization
    N_mat += lm_lambda * torch.diag(torch.diag(N_mat))
    # N += lm_lambda * 1e3 * torch.eye(6, device=points.device)

    # about lambda
    # If the lambda parameter is large, it implies that the algorithm is relying more on the gradient descent component of the optimization. This can lead to slower convergence as the steps are smaller, but it may improve stability and robustness, especially in the presence of noisy or ill-conditioned data.
    # If the lambda parameter is small, it implies that the algorithm is relying more on the Gauss-Newton component, which can make convergence faster. However, if the problem is ill-conditioned, setting lambda too small might result in numerical instability or divergence.

    g_vec = -(J_mat * weight).T @ sdf_residual

    t_vec = torch.linalg.inv(N_mat.to(dtype=torch.float64)) @ g_vec.to(
        dtype=torch.float64
    )  # 6dof tran parameters

    T_mat = torch.eye(4, device=points.device, dtype=torch.float64)
    T_mat[:3, :3] = expmap(t_vec[:3])  # rotation part
    T_mat[:3, 3] = t_vec[3:]  # translation part

    eigenvalues = (
        None  # the weight are also included, we need to normalize the weight part
    )
    if require_eigen:
        N_mat_raw_tran_part = N_mat_raw[3:, 3:]
        eigenvalues = torch.linalg.eigvals(N_mat_raw_tran_part).real
        # we need to set a threshold for the minimum eigenvalue for degerancy determination

    cov_mat = None
    if require_cov:
        # Compute the covariance matrix (using a scaling factor)
        mse = torch.mean(weight.squeeze(1) * sdf_residual**2)
        cov_mat = torch.linalg.inv(N_mat_raw) * mse  # rotation , translation

    return T_mat, cov_mat, eigenvalues


# functions
def implicit_color_reg(
    points,
    sdf_grad,
    sdf_residual,
    colors,
    color_grad,
    color_residual,
    weight,
    w_photo_loss=0.1,
    lm_lambda=0.0,
):

    geo_cross = torch.cross(points, sdf_grad)
    J_geo = torch.cat([geo_cross, sdf_grad], -1)  # first rotation, then translation
    N_geo = J_geo.T @ (weight * J_geo)
    g_geo = -(J_geo * weight).T @ sdf_residual

    N = N_geo
    g = g_geo

    color_channel = colors.shape[1]
    for i in range(
        color_channel
    ):  # we have converted color to intensity, so there's only one channel here
        color_cross_channel = torch.cross(
            points, color_grad[:, i, :]
        )  # first rotation, then translation
        J_color_channel = torch.cat([color_cross_channel, color_grad[:, i]], -1)
        N_color_channel = J_color_channel.T @ (weight * J_color_channel)
        g_color_channel = -(J_color_channel * weight).T @ color_residual[:, i]
        N += w_photo_loss * N_color_channel
        g += w_photo_loss * g_color_channel

    # use LM optimization
    # N += lm_lambda * torch.eye(6, device=points.device)
    N += lm_lambda * torch.diag(torch.diag(N))

    t = torch.linalg.inv(N.to(dtype=torch.float64)) @ g.to(dtype=torch.float64)  # 6dof

    T = torch.eye(4, device=points.device, dtype=torch.float64)
    T[:3, :3] = expmap(t[:3])  # rotation part
    T[:3, 3] = t[3:]  # translation part

    # TODO: add cov

    return T

# function for radar registration
def implicit_radar_reg(
    points,
    sdf_grad,
    sdf_residual,
    radar,
    radar_grad,
    radar_residual,
    weight,
    w_radar_loss=0.1,
    lm_lambda=0.0,
):
    geo_cross = torch.cross(points, sdf_grad)
    J_geo = torch.cat([geo_cross, sdf_grad], -1)  # first rotation, then translation
    N_geo = J_geo.T @ (weight * J_geo)
    g_geo = -(J_geo * weight).T @ sdf_residual
    
    N = N_geo
    g = g_geo
    
    radar_channel = radar.shape[1] # currently, its only 1 channel (rcs)
    for i in range(
        radar_channel
    ):
        radar_cross_channel = torch.cross(
            points, radar_grad[:, i, :]
        )  # first rotation, then translation
        J_radar_channel = torch.cat([radar_cross_channel, radar_grad[:, i]], -1)
        N_radar_channel = J_radar_channel.T @ (weight * J_radar_channel)
        g_radar_channel = -(J_radar_channel * weight).T @ radar_residual[:, i]
        N += w_radar_loss * N_radar_channel
        g += w_radar_loss * g_radar_channel
    
    N += lm_lambda * torch.diag(torch.diag(N))
    
    t = torch.linalg.inv(N.to(dtype=torch.float64)) @ g.to(dtype=torch.float64)  # 6dof
    
    T = torch.eye(4, device=points.device, dtype=torch.float64)
    T[:3, :3] = expmap(t[:3])  # rotation part
    T[:3, 3] = t[3:]  # translation part
    
    return T
    

# continous time registration (motion undistortion deskew is not needed then)
# point-wise timestamp required
# we regard the robot motion as uniform velocity in intervals (control poses)
# then each points transformation can be interpolated using the control poses
# we estimate poses of the control points
# we also need to enforce the conherent smoothness of the control poses
# and solve the non-linear optimization problem (TODO, not implemented yet)
def ct_registration_step(
    self,
    points: torch.Tensor,
    ts: torch.Tensor,
    normals: torch.Tensor,
    sdf_labels: torch.Tensor,
    colors: torch.Tensor,
    cur_ts,
    min_grad_norm,
    max_grad_norm,
    GM_dist=None,
    GM_grad=None,
    lm_lambda=0.0,
    vis_weight_pc=False,
):
    return


# math tools
def skew(v):
    S = torch.zeros(3, 3, device=v.device, dtype=v.dtype)
    S[0, 1] = -v[2]
    S[0, 2] = v[1]
    S[1, 2] = -v[0]
    return S - S.T


def expmap(axis_angle: torch.Tensor):

    angle = axis_angle.norm()
    axis = axis_angle / angle
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    S = skew(axis)
    R = eye + S * torch.sin(angle) + (S @ S) * (1.0 - torch.cos(angle))

    # print(R @ torch.linalg.inv(R))
    return R


def rotation_matrix_to_axis_angle(R):
    # epsilon = 1e-8  # A small value to handle numerical precision issues
    # Ensure the input matrix is a valid rotation matrix
    assert torch.is_tensor(R) and R.shape == (3, 3), "Invalid rotation matrix"
    # Compute the trace of the rotation matrix
    trace = torch.trace(R)
    # Compute the angle of rotation
    angle = torch.acos((trace - 1) / 2)

    return angle  # rad







### code for NDT odometry

def filter_surface_points(points, sdf_values, epsilon=0.05):
    """
    Filter points near the surface based on SDF values.

    Args:
        points (torch.Tensor): Input point cloud (Nx3).
        sdf_values (torch.Tensor): Corresponding SDF values (N,).
        epsilon (float): Threshold for surface points.

    Returns:
        surface_points (torch.Tensor): Points near the surface (Mx3).
    """
    surface_mask = torch.abs(sdf_values) < epsilon
    surface_points = points[surface_mask]
    return surface_points

def create_ndt_from_pointcloud_fast(surface_points, voxel_size):
    """
    Fast NDT map creation from surface points using scatter for parallelization.

    Args:
        surface_points (torch.Tensor): Points near the surface (Mx3).
        voxel_size (float): Size of each voxel.

    Returns:
        ndt_map (dict): NDT map with voxel indices as keys and (mean, covariance) as values.
    """
    # Compute voxel indices
    voxel_indices = torch.floor(surface_points / voxel_size).to(torch.int32)  # (M, 3)

    # Flatten voxel indices to unique keys (1D)
    voxel_keys = (voxel_indices * torch.tensor([1, 1000, 1000000], device=voxel_indices.device)).sum(dim=1)

    # Find unique voxel keys and assign points to them
    unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)

    # Initialize structures for mean and covariance computation
    num_voxels = unique_keys.size(0)
    ndt_means = torch.zeros((num_voxels, 3), device=surface_points.device)
    ndt_covariances = torch.zeros((num_voxels, 3, 3), device=surface_points.device)
    voxel_counts = torch.zeros(num_voxels, device=surface_points.device)

    # Step 1: Calculate mean using scatter
    ndt_means = ndt_means.index_add_(0, inverse_indices, surface_points)
    voxel_counts = voxel_counts.index_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32))
    ndt_means /= voxel_counts.unsqueeze(1)

    # Step 2: Calculate covariance using scatter
    diffs = surface_points - ndt_means[inverse_indices]
    cov_updates = diffs.unsqueeze(2) @ diffs.unsqueeze(1)  # (M, 3, 3)
    ndt_covariances.index_add_(0, inverse_indices, cov_updates)
    ndt_covariances /= voxel_counts.unsqueeze(1).unsqueeze(2)

    # Step 3: Build the NDT map
    ndt_map = {}
    for i, key in enumerate(unique_keys):
        if voxel_counts[i] > 1:  # Exclude single-point voxels
            voxel_idx = (key % 1000, (key // 1000) % 1000, key // 1000000)
            ndt_map[voxel_idx] = (ndt_means[i], ndt_covariances[i])

    return ndt_map
