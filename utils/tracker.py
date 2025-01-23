#!/usr/bin/env python3
# @file      tracker.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import math

import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt

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
        
        self.number = 0
        self.test_T = torch.eye(4, dtype=torch.float64, device=self.device)

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
            self.test_T = torch.eye(4, dtype=torch.float64, device=self.device)
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
        
        # only for check
        T_prev = T

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
        
        # print("surface full points shape: ", surface_points_from_sdf.shape)
        
        # leave only the points that are included in the area of source_points
        
        max_x = source_points[:, 0].max()
        min_x = source_points[:, 0].min()
        max_y = source_points[:, 1].max()
        min_y = source_points[:, 1].min()
        max_z = source_points[:, 2].max()
        min_z = source_points[:, 2].min()
        
        mask_source = (surface_points_from_sdf[:, 0] < max_x) & (surface_points_from_sdf[:, 0] > min_x) & (surface_points_from_sdf[:, 1] < max_y) & (surface_points_from_sdf[:, 1] > min_y) & (surface_points_from_sdf[:, 2] < max_z) & (surface_points_from_sdf[:, 2] > min_z)
        surface_points_from_sdf = surface_points_from_sdf[mask_source]
        
        # print("front surface points shape: ", surface_points_from_sdf.shape)
        # print("surface point center: ", surface_points_from_sdf.mean(dim=0))   
        # print("surface point x min: ", surface_points_from_sdf[:, 0].min())
        # print("surface point x max: ", surface_points_from_sdf[:, 0].max())
        # print("surface point y min: ", surface_points_from_sdf[:, 1].min())
        # print("surface point y max: ", surface_points_from_sdf[:, 1].max())
        # print("surface point z min: ", surface_points_from_sdf[:, 2].min())
        # print("surface point z max: ", surface_points_from_sdf[:, 2].max())
        
        # print("source points shape: ", source_points.shape)
        # print("source points center: ", source_points.mean(dim=0))
        # print("source point x min: ", source_points[:, 0].min())
        # print("source point x max: ", source_points[:, 0].max())
        # print("source point y min: ", source_points[:, 1].min())
        # print("source point y max: ", source_points[:, 1].max())
        # print("source point z min: ", source_points[:, 2].min())
        # print("source point z max: ", source_points[:, 2].max())
        
        ndt_voxel_size = 2
        ndt_min_points = 3
        
        # ndt_boundary = ((0, 100), (-25, 25), (-5, 20))
        ndt_boundary = None
        
        # start measuring the time
        aT0 = get_time()
        local_ndt_idx, local_ndt_mean, local_ndt_cov = create_ndt_overlap_fast_tensor(surface_points_from_sdf, ndt_voxel_size, ndt_min_points, voxel_bounds = ndt_boundary)
        aT1 = get_time()
        source_ndt_idx, source_ndt_mean, source_ndt_cov = create_ndt_overlap_fast_tensor(source_points, ndt_voxel_size, ndt_min_points, voxel_bounds = ndt_boundary)
        aT2 = get_time()
        
        print("local ndt size: ", len(local_ndt_mean))
        # print("local ndt time: ", (aT1 - aT0) * 1e3)
        print("source ndt size: ", len(source_ndt_mean))
        # print("source ndt time: ", (aT2 - aT1) * 1e3)
        
                        
        # start the odometry calculation
        ndt_odometry = register_ndt_exponential_gn_LM_allpairs((source_ndt_idx, source_ndt_mean, source_ndt_cov), (local_ndt_idx, local_ndt_mean, local_ndt_cov), device=self.device, dtype=self.dtype)
        aT3 = get_time()
        
        print("ndt odometry time: ", (aT3 - aT2) * 1e3)
        
        # file_name_test = "/home/irap/sdf_map/ndt_test/ndt_test_" + str(self.number) + ".png"
        # save_ndt_2d_bird_eye_tensor((local_ndt_idx, local_ndt_mean, local_ndt_cov), (source_ndt_idx, source_ndt_mean, source_ndt_cov), surface_points_from_sdf, source_points, file_name_test, xlim=(0, 200), ylim=(-50, 50))
        
        # file_name_test2 = "/home/irap/sdf_map/ndt_test_vertical/ndt_test_vertical_" + str(self.number) + ".png"
        # save_ndt_2d_side_view_tensor((local_ndt_idx, local_ndt_mean, local_ndt_cov), (source_ndt_idx, source_ndt_mean, source_ndt_cov), surface_points_from_sdf, source_points, file_name_test2, xlim=(0, 200), zlim=(-50, 50))
        # self.number += 1
        
        ndt_R, ndt_t = ndt_odometry
        ndt_delta_T = torch.eye(4, dtype=torch.float64, device=self.device)
        ndt_delta_T[:3, :3] = ndt_R
        ndt_delta_T[:3, 3] = ndt_t.squeeze()
        
        self.test_T = ndt_delta_T @ self.test_T
        
        print("ndt odometry result: ", self.test_T)
        print("original odometry result: ", T)
        # print("ndt odometry delta: ", ndt_delta_T)
        # print("original dometry delta: ", T @ torch.inverse(T_prev))
        
        # only for fast testing... not good way
        # T = ndt_delta_T @ T_prev

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

def filter_surface_points(points, sdf_values, epsilon=0.03):
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


def create_ndt_overlap_fast_tensor(
    surface_points: torch.Tensor,
    voxel_size: int = 5,
    min_num: int = 10,
    epsilon: float = 1e-2,
    use_half: bool = False,
    voxel_bounds=None,
):
    """
    Overlap NDT를 PyTorch에서 '고도 최적화' 기법으로 구현 (배치/벡터화).
    최종 결과를 dict 대신, 텐서 형태로 반환하여
    후속 registration 등에서 훨씬 빠른 연산 가능.

    Args:
        surface_points (torch.Tensor): (M,3) [GPU], float32/float64 권장
        voxel_size (int): 기본 5 (슬라이딩 폭)
        min_num (int): voxel 내 점이 이 개수 이상이어야 채택
        epsilon (float): 공분산 regularization
        use_half (bool): True면 half 내부 연산. 단, 고유값 분해는 float32로 
        voxel_bounds (tuple or None):
            ((x_min, x_max), (y_min, y_max), (z_min, z_max)) 형태
            voxel 시작점(vx,vy,vz)이 이 범위 안일 때만 채택
            None이면 제한 없음

    Returns:
        voxel_idxs_3d (torch.Tensor): (N,3), int32 or int64
        voxel_means   (torch.Tensor): (N,3), float32/float16/...
        voxel_covs    (torch.Tensor): (N,3,3), same dtype as voxel_means
        * N은 최종 남은 voxel 개수(count>=min_num)
    """
    device = surface_points.device
    orig_dtype = surface_points.dtype

    # -------------------------------
    # 0) Half precision 변환(옵션)
    # -------------------------------
    if use_half and surface_points.dtype in (torch.float32, torch.float64):
        surface_points = surface_points.half()

    dtype = surface_points.dtype  # float16 or float32 or float64
    M = surface_points.size(0)

    # -------------------------------
    # 1) 각 점의 floor 좌표
    # -------------------------------
    p_floor = torch.floor(surface_points).to(torch.int32)  # (M,3)
    px_floor = p_floor[:, 0]
    py_floor = p_floor[:, 1]
    pz_floor = p_floor[:, 2]

    # -------------------------------
    # 2) 3D meshgrid
    # -------------------------------
    offsets_1d = torch.arange(voxel_size, device=device, dtype=torch.int32)  
    xgrid, ygrid, zgrid = torch.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing='ij')

    # -------------------------------
    # 3) (vx,vy,vz) + 마스킹
    # -------------------------------
    VX = px_floor.view(M,1,1,1) - xgrid.view(1,voxel_size,voxel_size,voxel_size)
    VY = py_floor.view(M,1,1,1) - ygrid.view(1,voxel_size,voxel_size,voxel_size)
    VZ = pz_floor.view(M,1,1,1) - zgrid.view(1,voxel_size,voxel_size,voxel_size)

    PX = surface_points[:,0].view(M,1,1,1).expand(M,voxel_size,voxel_size,voxel_size).contiguous()
    PY = surface_points[:,1].view(M,1,1,1).expand(M,voxel_size,voxel_size,voxel_size).contiguous()
    PZ = surface_points[:,2].view(M,1,1,1).expand(M,voxel_size,voxel_size,voxel_size).contiguous()

    valid_mask = (
        (VX <= PX) & (PX < (VX + voxel_size)) &
        (VY <= PY) & (PY < (VY + voxel_size)) &
        (VZ <= PZ) & (PZ < (VZ + voxel_size))
    )

    valid_mask_flat = valid_mask.view(-1)
    valid_idx = torch.nonzero(valid_mask_flat, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        print("[Warning] No valid overlap found. Return empty.")
        return (torch.empty((0,3),dtype=torch.int32,device=device),
                torch.empty((0,3),dtype=dtype,device=device),
                torch.empty((0,3,3),dtype=dtype,device=device))

    # Flatten + gather
    VXf = VX.view(-1)
    VYf = VY.view(-1)
    VZf = VZ.view(-1)
    PXf = PX.view(-1)
    PYf = PY.view(-1)
    PZf = PZ.view(-1)

    vx_valid = VXf[valid_idx]
    vy_valid = VYf[valid_idx]
    vz_valid = VZf[valid_idx]
    px_valid = PXf[valid_idx]
    py_valid = PYf[valid_idx]
    pz_valid = PZf[valid_idx]

    points_valid = torch.stack((px_valid, py_valid, pz_valid), dim=1)  # (K,3)

    # -------------------------------
    # 5) voxel_bounds 적용
    # -------------------------------
    if voxel_bounds is not None:
        (x_lim, y_lim, z_lim) = voxel_bounds

        def range_mask(vals, rng):
            mn, mx = rng
            mask_ = torch.ones_like(vals, dtype=torch.bool)
            if mn is not None:
                mask_ &= (vals >= mn)
            if mx is not None:
                mask_ &= (vals < mx)
            return mask_

        mask_x = range_mask(vx_valid, x_lim)
        mask_y = range_mask(vy_valid, y_lim)
        mask_z = range_mask(vz_valid, z_lim)

        in_bound_mask = mask_x & mask_y & mask_z
        if not in_bound_mask.any():
            print("[Warning] All valid voxels are out of voxel_bounds.")
            return (torch.empty((0,3),dtype=torch.int32,device=device),
                    torch.empty((0,3),dtype=dtype,device=device),
                    torch.empty((0,3,3),dtype=dtype,device=device))

        in_bound_idx = torch.nonzero(in_bound_mask, as_tuple=False).squeeze(1)
        vx_valid = vx_valid[in_bound_idx]
        vy_valid = vy_valid[in_bound_idx]
        vz_valid = vz_valid[in_bound_idx]
        points_valid = points_valid[in_bound_idx]

    # -------------------------------
    # 6) voxel key + unique + inverse
    # -------------------------------
    offset_val = 500
    vx_off = (vx_valid + offset_val).to(torch.int32)
    vy_off = (vy_valid + offset_val).to(torch.int32)
    vz_off = (vz_valid + offset_val).to(torch.int32)

    voxel_keys = vx_off + vy_off*1000 + vz_off*1000000
    unique_keys, inv_idx = torch.unique(voxel_keys, return_inverse=True)
    num_voxels = unique_keys.size(0)

    # -------------------------------
    # 7) Two-Pass 누적
    # -------------------------------
    counts = torch.zeros(num_voxels, device=device, dtype=dtype)
    counts.index_add_(0, inv_idx, torch.ones_like(inv_idx, dtype=dtype))

    sum_xyz = torch.zeros((num_voxels,3), device=device, dtype=dtype)
    sum_xyz.index_add_(0, inv_idx, points_valid)

    x_valid = points_valid[:,0]
    y_valid = points_valid[:,1]
    z_valid = points_valid[:,2]

    sum_xx = torch.zeros(num_voxels, device=device, dtype=dtype)
    sum_xy = torch.zeros(num_voxels, device=device, dtype=dtype)
    sum_xz = torch.zeros(num_voxels, device=device, dtype=dtype)
    sum_yy = torch.zeros(num_voxels, device=device, dtype=dtype)
    sum_yz = torch.zeros(num_voxels, device=device, dtype=dtype)
    sum_zz = torch.zeros(num_voxels, device=device, dtype=dtype)

    sum_xx.index_add_(0, inv_idx, x_valid*x_valid)
    sum_xy.index_add_(0, inv_idx, x_valid*y_valid)
    sum_xz.index_add_(0, inv_idx, x_valid*z_valid)
    sum_yy.index_add_(0, inv_idx, y_valid*y_valid)
    sum_yz.index_add_(0, inv_idx, y_valid*z_valid)
    sum_zz.index_add_(0, inv_idx, z_valid*z_valid)

    # -------------------------------
    # 8) 공분산 + regularization
    # -------------------------------
    means = sum_xyz / counts.unsqueeze(1).clamp_min(1e-9)

    Exx = sum_xx / counts.clamp_min(1e-9)
    Exy = sum_xy / counts.clamp_min(1e-9)
    Exz = sum_xz / counts.clamp_min(1e-9)
    Eyy = sum_yy / counts.clamp_min(1e-9)
    Eyz = sum_yz / counts.clamp_min(1e-9)
    Ezz = sum_zz / counts.clamp_min(1e-9)

    mx = means[:,0]
    my = means[:,1]
    mz = means[:,2]

    covs = torch.zeros((num_voxels,3,3), device=device, dtype=dtype)
    cov00 = Exx - mx*mx
    cov01 = Exy - mx*my
    cov02 = Exz - mx*mz
    cov11 = Eyy - my*my
    cov12 = Eyz - my*mz
    cov22 = Ezz - mz*mz

    covs[:,0,0] = cov00
    covs[:,0,1] = cov01; covs[:,1,0] = cov01
    covs[:,0,2] = cov02; covs[:,2,0] = cov02
    covs[:,1,1] = cov11
    covs[:,1,2] = cov12; covs[:,2,1] = cov12
    covs[:,2,2] = cov22

    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    covs = 0.5*(covs + covs.transpose(1,2)) + epsilon*I

    # Half -> float32 변환하여 eigh
    need_recast = (use_half and (dtype==torch.float16))
    if need_recast:
        cov_f32 = covs.float()
        eigvals_f32, eigvecs_f32 = torch.linalg.eigh(cov_f32)
        eigvals_f32 = torch.clamp(eigvals_f32, min=epsilon)
        cov_recon_f32 = eigvecs_f32 @ torch.diag_embed(eigvals_f32) @ eigvecs_f32.transpose(-1,-2)
        cov_recon_f32 = 0.5*(cov_recon_f32 + cov_recon_f32.transpose(1,2))
        covs = cov_recon_f32.half()
    else:
        eigvals, eigvecs = torch.linalg.eigh(covs)
        eigvals = torch.clamp(eigvals, min=epsilon)
        covs = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1,-2)
        covs = 0.5*(covs + covs.transpose(1,2))

    # -------------------------------
    # 9) counts >= min_num 필터
    # -------------------------------
    keep_mask = (counts >= min_num)
    if not keep_mask.any():
        print("[Info] All voxels are below min_num. Return empty.")
        return (torch.empty((0,3),dtype=torch.int32,device=device),
                torch.empty((0,3),dtype=dtype,device=device),
                torch.empty((0,3,3),dtype=dtype,device=device))

    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    covs = covs[keep_idx]
    means = means[keep_idx]
    # unique_keys -> (vx,vy,vz)
    uk = unique_keys[keep_idx]
    vx_ = (uk % 1000)
    vy_ = ((uk // 1000) % 1000)
    vz_ = ((uk // 1000000) % 1000)
    vx_ = vx_ - offset_val
    vy_ = vy_ - offset_val
    vz_ = vz_ - offset_val

    voxel_idxs_3d = torch.stack([vx_, vy_, vz_], dim=1)  # (N,3), int32

    return voxel_idxs_3d, means, covs







def compute_grad_hess_allpairs(
    ndtA, ndtB, R, t
):
    """
    모든 voxel 쌍 (K*M) 비교:
    cost = sum( 1 - factor_ij * exp(-0.5 * diff_ij^T invC_ij diff_ij ) ).
    => batch dimension = K*M

    param= (omega(3), trans(3))
    Returns: grad(6,), hess(6,6), cost (float)
    """

    device = R.device
    # ndtA=(voxel_idx_A, meansA(K,3), covA(K,3,3))
    # ndtB=(voxel_idx_B, meansB(M,3), covB(M,3,3))
    vxA, meansA, covA = ndtA
    vxB, meansB, covB = ndtB

    K = meansA.size(0)
    M = meansB.size(0)

    # cast
    muA = meansA.to(device=device, dtype=R.dtype)  # (K,3)
    SigA= covA.to(device=device, dtype=R.dtype)    # (K,3,3)
    muB = meansB.to(device=device, dtype=R.dtype)  # (M,3)
    SigB= covB.to(device=device, dtype=R.dtype)    # (M,3,3)

    # 1) expand -> shape(K,M,3) for diff
    # muA shape= (K,3)-> (K,1,3)
    muA_exp = muA.unsqueeze(1)      # (K,1,3)
    # muB shape= (M,3)->(1,M,3)
    muB_exp = muB.unsqueeze(0)      # (1,M,3)
    # => (K,M,3)
    # transform: R muA + t => we do batch for muA only => shape(K,3)
    muA_rot = muA @ R.transpose(-1,-2) + t.view(1,3)   # (K,3)
    muA_rot_exp= muA_rot.unsqueeze(1)                  # (K,1,3)

    # broadcast -> diff = (K,M,3)
    diff= muA_rot_exp - muB_exp
    # flatten => (K*M,3)
    diff_2d= diff.view(-1,3)  # flatten

    # 2) cov transform
    # R covA R^T => (K,3,3)
    R_ = R.unsqueeze(0)              # (1,3,3)
    R_expand= R_.expand(K,3,3)       # (K,3,3)
    covA_rot= R_expand.bmm(SigA).bmm(R_expand.transpose(1,2))  # (K,3,3)

    # now we want "covA_rot[i] + covB[j]" => shape(K,M,3,3). flatten =>(K*M,3,3)
    # expand covA_rot => (K,1,3,3)
    covA_rot_exp= covA_rot.unsqueeze(1)   # (K,1,3,3)
    # expand covB => (1,M,3,3)
    covB_exp= SigB.unsqueeze(0)          # (1,M,3,3)
    # broadcast => (K,M,3,3)
    comb= covA_rot_exp + covB_exp
    comb_2d= comb.view(-1,3,3)  # shape (K*M,3,3)

    # 3) det, inv
    detC= torch.linalg.det(comb_2d).clamp_min(1e-12)  # (K*M,)
    invC= torch.linalg.inv(comb_2d)                   # (K*M,3,3)

    # factor= 1 / sqrt((2pi)^3* detC)
    two_pi_3half= (2*math.pi)**1.5
    factor_= 1.0/( two_pi_3half* torch.sqrt(detC))    # (K*M,)

    # 4) x_i= diff^T invC diff => shape(K*M)
    # we can do: (K*M,3) -> unsqueeze(1)->(K*M,1,3), mm->(K*M,1,3)-> mm->(K*M,1,1)
    diff_3= diff_2d.unsqueeze(1)  # (K*M,1,3)
    left= torch.bmm(diff_3, invC) # (K*M,1,3)
    quad_= torch.bmm(left, diff_2d.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (K*M,)
    exponent_= torch.exp(-0.5* quad_)  # (K*M,)

    val_= factor_* exponent_            # (K*M,)
    # cost= sum( 1- val_i ), shape => (K*M,)
    cost_each= (1.0- val_).clamp_min(0.0)
    cost_val= cost_each.sum().item()

    # 5) gradient
    # partial( cost_i )= - partial(val_i ), val_i= factor_i*exp(...)
    # ignoring partial factor wrt param => partial val_i= val_i*(-0.5)* grad_x_i
    # cost_i=1- val_i => partial= - partial(val_i )= +0.5 val_i* grad_x
    # grad_x= 2 invC diff
    # => grad_x_i shape (K*M,3)
    # do batch:
    # grad_x= 2.0* (invC( (K*M,3,3 )) x diff_2d( (K*M,3) ) )
    # let's do a bmm with shape (K*M,3,3)*(K*M,3,1)->(K*M,3,1)...

    diff_3_again= diff_2d.view(-1,3,1) # (K*M,3,1)
    grad_x_ = 2.0* torch.bmm(invC, diff_3_again).squeeze(-1)  # (K*M,3)

    # build J_diff => shape(K*M, 3,6)
    # we need -R skew(muA[i]) for each i => but now i -> i,j => we must do " - R skew( muA[i] ) " depends only on i, not j
    # Actually, we must do an approach:
    #   skew( muA[i] ) for i => shape(K,3,3), expand along M => shape(K,M,3,3)
    #   then flatten => shape(K*M,3,3). multiply by R => shape(K*M,3,3)
    # let's do that approach:
    
    # skew( muA ) => (K,3,3)
    # broadcast => (K,M,3,3), flatten => (K*M,3,3)
    # then multiply by -R for each i => but R doesn't depend on i => we do once for i => we do it for all pairs
    # Actually simpler: muA[i], for each i, broadcast along j => same approach as for covA. Then flatten => (K*M,3).
    # Then do skew in (K*M,3) => (K*M,3,3)...

    # create big MuA2d => shape(K*M,3)
    muA_expand= muA.unsqueeze(1).expand(K,M,3).contiguous().view(-1,3)  # (K*M,3)
    # skew them => we can do direct formula
    skew_mat= muA_expand.new_zeros(K*M,3,3)
    skew_mat[:,0,1]= -muA_expand[:,2]
    skew_mat[:,0,2]=  muA_expand[:,1]
    skew_mat[:,1,0]=  muA_expand[:,2]
    skew_mat[:,1,2]= -muA_expand[:,0]
    skew_mat[:,2,0]= -muA_expand[:,1]
    skew_mat[:,2,1]=  muA_expand[:,0]

    # dRmu => -(R@ skew_mat[i]) => we do shape(1,3,3)->(K*M,3,3)? Actually R is (3,3) => expand => multiply
    R_expand2= R.unsqueeze(0).expand(K*M,3,3)
    dRmu_2d= - torch.bmm(R_expand2, skew_mat)  # (K*M,3,3)

    # I3 => shape(K*M,3,3)
    I3_2d= torch.eye(3, device=device, dtype=R.dtype).unsqueeze(0).expand(K*M,3,3)
    # cat => shape(K*M,3,6)
    J_diff_2d= torch.cat([dRmu_2d, I3_2d], dim=2)

    # grad_x_i => shape(K*M,6)
    grad_x_3= grad_x_.unsqueeze(1)  # (K*M,1,3)
    # matmul =>(K*M,1,6)->(K*M,6)
    grad_x_i= torch.bmm( grad_x_3, J_diff_2d ).squeeze(1)

    # partial( cost_i )= val_i* 0.5 * grad_x_i
    # => shape(K*M,6)
    val_2d= val_.unsqueeze(1)  # (K*M,1)
    grad_term= 0.5* val_2d* grad_x_i

    # sum => shape(6)
    grad_out= grad_term.sum(dim=0)

    # hessian => shape(K*M,6,6). we do outer product => sum
    gt_expand1= grad_term.unsqueeze(2)  # (K*M,6,1)
    gt_expand2= grad_term.unsqueeze(1)  # (K*M,1,6)
    hess_batch= gt_expand1* gt_expand2  # (K*M,6,6)
    hess_out= hess_batch.sum(dim=0)

    return grad_out, hess_out, cost_val


def register_ndt_exponential_gn_LM_allpairs(
    ndtA, ndtB,
    max_iter=30,
    init_rotation=None,
    init_translation=None,
    tolerance=1e-6,
    device=None,
    dtype=None
):
    """
    LM approach + all-pairs approach => (K*M) batch
    cost = sum( 1 - factor * exp(-0.5 x) ), ignoring partial factor wrt param
    We'll store "old_cost" to refer to the last accepted cost.
    """
    if device is None:
        device = ndtA[1].device
    if dtype is None:
        dtype = ndtA[1].dtype

    # 초기 파라미터 설정
    if init_rotation is None:
        R = torch.eye(3, device=device, dtype=dtype)
    else:
        R = init_rotation.to(device=device, dtype=dtype)
    if init_translation is None:
        t = torch.zeros((3,1), device=device, dtype=dtype)
    else:
        t = init_translation.to(device=device, dtype=dtype)

    lm_lambda = 0.01

    # ----------------------------
    # (1) 초기 cost 계산 -> old_cost
    # ----------------------------
    grad_init, hess_init, old_cost = compute_grad_hess_allpairs(ndtA, ndtB, R, t)

    for step in range(max_iter):
        # 현재 (R,t)에 대한 grad, hess, cost 계산
        grad, hess, cost = compute_grad_hess_allpairs(ndtA, ndtB, R, t)

        rank = torch.linalg.matrix_rank(hess)
        if rank < 6:
            print(f"[Info] Hessian rank < 6 at step={step}, break.")
            break

        # LM 뎀핑
        damp_I = lm_lambda * torch.eye(6, device=device, dtype=dtype)
        hess_damped = hess + damp_I

        try:
            delta = torch.linalg.solve(hess_damped, -grad)
        except RuntimeError:
            # singular => lambda 증가 후 재시도
            lm_lambda *= 10
            continue
                
        d_omega = delta[:3]
        d_trans = delta[3:]
        R_up = expmap(d_omega)  # external 'expmap'
        R_new = R_up @ R
        t_new = t + d_trans.view(3,1)

        # 후보 파라미터 (R_new, t_new)에 대한 cost 계산
        grad2, hess2, new_cost = compute_grad_hess_allpairs(ndtA, ndtB, R_new, t_new)
        
        print("lm_lambda: ", lm_lambda)
        print(f"old_cost= {old_cost}, cost= {cost}, new_cost= {new_cost}")

        # ----------------------------
        # (2) 개선 여부 판단
        # ----------------------------
        if new_cost < old_cost:
            # improvement => accept
            R = R_new
            t = t_new

            # old_cost 갱신
            old_cost = new_cost

            lm_lambda *= 0.5

            # 수렴조건
            if abs(old_cost - new_cost) < tolerance:
                print(f"[Info] Converged at step={step}, cost={new_cost}")
                break
        else:
            # revert => (R,t) 그대로, lambda만 증가
            lm_lambda *= 5

    return R, t









def save_ndt_2d_bird_eye_tensor(
    ndt_blue, ndt_red,
    point_cloud1, point_cloud2,
    save_path, xlim, ylim,
    alpha=0.1, scale=1.5, figsize=(30, 30)
):
    """
    Save a 2D bird's-eye (X-Y) view of two NDT maps (in Tensor form) and point clouds as an image.

    Args:
        ndt_blue (tuple): (voxel_idxs_3d, means, covs) for the first NDT, each a torch.Tensor.
            - voxel_idxs_3d.shape == (N,3)
            - means.shape == (N,3)
            - covs.shape == (N,3,3)
        ndt_red (tuple): Same format as ndt_blue.
        point_cloud1, point_cloud2: (Nx3 or Nx2), torch.Tensor or np.ndarray
        save_path (str): Path to save the image file
        xlim (tuple): (x_min, x_max) for X axis
        ylim (tuple): (y_min, y_max) for Y axis
        alpha (float): Transparency of ellipses
        scale (float): Factor to scale ellipse size
        figsize (tuple): Figure size in inches
    """
    # Ensure xlim, ylim are numpy
    if isinstance(xlim[0], torch.Tensor):
        xlim = (xlim[0].cpu().numpy(), xlim[1].cpu().numpy())
    if isinstance(ylim[0], torch.Tensor):
        ylim = (ylim[0].cpu().numpy(), ylim[1].cpu().numpy())

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("2D Bird's-Eye View of NDT (Tensor) and Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')  # Ensure equal axis resolution

    def plot_ndt_tensors(ndt_tensors, color):
        """Plot ellipse for each voxel's mean/cov in X-Y plane."""
        voxel_idxs_3d, means, covs = ndt_tensors

        # move data to CPU if needed
        if means.is_cuda:
            means = means.cpu()
        if covs.is_cuda:
            covs = covs.cpu()

        means_np = means.numpy()
        covs_np = covs.numpy()

        # For each voxel
        for i in range(means_np.shape[0]):
            mean_3d = means_np[i]       # e.g. [mx, my, mz]
            cov_3x3 = covs_np[i]       # shape (3,3)

            # 2D projection: XY plane
            mean_2d = mean_3d[:2]
            cov_2d = cov_3x3[:2, :2]

            # Eigen decomposition for ellipse
            eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
            # rotate ellipse
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            # ellipse axes
            width, height = scale * np.sqrt(eigenvalues)

            # Draw ellipse
            ellipse = plt.matplotlib.patches.Ellipse(
                xy=mean_2d, width=width, height=height,
                angle=np.degrees(angle),
                edgecolor=color, facecolor=color, alpha=alpha
            )
            ax.add_patch(ellipse)

    # Plot Point Clouds
    def to_xy(pc):
        """Ensure we have Nx2 for plotting (taking XY from Nx3)."""
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        if pc.shape[1] == 3:
            pc = pc[:, :2]
        return pc

    pc1_xy = to_xy(point_cloud1)
    pc2_xy = to_xy(point_cloud2)
    ax.scatter(pc1_xy[:, 0], pc1_xy[:, 1], c='green', s=10, label='Local Cloud')
    ax.scatter(pc2_xy[:, 0], pc2_xy[:, 1], c='yellow', s=10, label='Source Cloud')

    # Plot NDTs
    plot_ndt_tensors(ndt_blue, color='blue')
    plot_ndt_tensors(ndt_red, color='red')

    ax.legend(loc='upper right')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def save_ndt_2d_side_view_tensor(
    ndt_blue, ndt_red,
    point_cloud1, point_cloud2,
    save_path, xlim, zlim,
    alpha=0.2, scale=1.5, figsize=(30, 30)
):
    """
    Save a 2D side view (X-Z plane) of two NDT maps (in Tensor form) and point clouds as an image.

    Args:
        ndt_blue (tuple): (voxel_idxs_3d, means, covs) for the first NDT
        ndt_red (tuple): (voxel_idxs_3d, means, covs) for the second NDT
        point_cloud1, point_cloud2: Nx3 or Nx2, torch.Tensor or np.ndarray
        save_path (str): path to save
        xlim (tuple): (x_min, x_max)
        zlim (tuple): (z_min, z_max)
        alpha (float): ellipse transparency
        scale (float): ellipse size scale
        figsize (tuple): figure size
    """
    if isinstance(xlim[0], torch.Tensor):
        xlim = (xlim[0].cpu().numpy(), xlim[1].cpu().numpy())
    if isinstance(zlim[0], torch.Tensor):
        zlim = (zlim[0].cpu().numpy(), zlim[1].cpu().numpy())

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_title("2D Side View (X-Z) of NDT (Tensor) and Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_aspect('equal')

    def plot_ndt_tensors(ndt_tensors, color):
        voxel_idxs_3d, means, covs = ndt_tensors

        if means.is_cuda:
            means = means.cpu()
        if covs.is_cuda:
            covs = covs.cpu()

        means_np = means.numpy()
        covs_np = covs.numpy()

        # For each voxel
        for i in range(means_np.shape[0]):
            mean_3d = means_np[i]
            cov_3x3 = covs_np[i]

            # X-Z plane
            mean_xz = mean_3d[[0, 2]]
            cov_xz = cov_3x3[np.ix_([0,2],[0,2])]  # shape (2,2)

            eigenvalues, eigenvectors = np.linalg.eigh(cov_xz)
            angle = np.arctan2(eigenvectors[1,0], eigenvectors[0,0])
            width, height = scale * np.sqrt(eigenvalues)

            ellipse = plt.matplotlib.patches.Ellipse(
                xy=mean_xz, width=width, height=height,
                angle=np.degrees(angle),
                edgecolor=color, facecolor=color, alpha=alpha
            )
            ax.add_patch(ellipse)

    # Plot point clouds in X-Z
    def to_xz(pc):
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        if pc.shape[1] == 3:
            return pc[:, [0,2]]
        return pc  # if it's Nx2 already

    pc1_xz = to_xz(point_cloud1)
    pc2_xz = to_xz(point_cloud2)
    ax.scatter(pc1_xz[:,0], pc1_xz[:,1], c='green', s=10, label='Local Cloud')
    ax.scatter(pc2_xz[:,0], pc2_xz[:,1], c='yellow', s=10, label='Source Cloud')

    # Plot NDTs
    plot_ndt_tensors(ndt_blue, color='blue')
    plot_ndt_tensors(ndt_red, color='red')

    ax.legend(loc='upper right')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)








# non-verlap ndt generation code

# def create_ndt_3d(
#     points: torch.Tensor,
#     voxel_size: float = 5.0,
#     min_points: int = 10,
#     epsilon: float = 1e-3,
# ) -> tuple:
#     """
#     - points: (N,3) shape, float
#     - voxel_size: 예) 5.0 (5m)
#     - min_points: 보셀 내 점이 이 개수보다 적으면 제외
#     - epsilon: 공분산 행렬 regularization
#     반환: (voxel_idxs, means, covs)
#       - voxel_idxs: (K,3), 각 보셀의 int 인덱스 (vx,vy,vz)
#       - means: (K,3) float
#       - covs: (K,3,3)
#     """
#     device = points.device
#     dtype = points.dtype
#     N = points.shape[0]

#     # 1) 보셀 인덱스화: floor(x/voxel_size)
#     voxel_idx = torch.floor(points / voxel_size).to(torch.int64)  # (N,3)

#     # 2) unique, inverse
#     #   편의상 offset 따로 안 쓰고, (vx,vy,vz)를 튜플로 모아 key로 써도 됨.
#     #   여기서는 큰 포인트 클라우드에서도 빠르게 하기 위해 아래처럼 1D key화
#     offset_val = 10_000  # voxel 인덱스가 음수가 있을 수 있으니 offset
#     vx_off = voxel_idx[:,0] + offset_val
#     vy_off = voxel_idx[:,1] + offset_val
#     vz_off = voxel_idx[:,2] + offset_val

#     key_1d = vx_off + vy_off*100000 + vz_off*100000**2  # 예: 최대 10만 범위라고 가정
#     unique_keys, inv_idx = torch.unique(key_1d, return_inverse=True)
#     K = unique_keys.shape[0]

#     # 3) 보셀 별로 (count, sum, sum of squares) 누적
#     ones_ = torch.ones(N, device=device, dtype=dtype)
#     counts = torch.zeros(K, device=device, dtype=dtype)
#     counts.index_add_(0, inv_idx, ones_)

#     sum_xyz = torch.zeros(K, 3, device=device, dtype=dtype)
#     sum_xyz.index_add_(0, inv_idx, points)

#     sum_x2 = torch.zeros(K, device=device, dtype=dtype)
#     sum_y2 = torch.zeros(K, device=device, dtype=dtype)
#     sum_z2 = torch.zeros(K, device=device, dtype=dtype)
#     sum_xy = torch.zeros(K, device=device, dtype=dtype)
#     sum_xz = torch.zeros(K, device=device, dtype=dtype)
#     sum_yz = torch.zeros(K, device=device, dtype=dtype)

#     sum_x2.index_add_(0, inv_idx, points[:,0]*points[:,0])
#     sum_y2.index_add_(0, inv_idx, points[:,1]*points[:,1])
#     sum_z2.index_add_(0, inv_idx, points[:,2]*points[:,2])
#     sum_xy.index_add_(0, inv_idx, points[:,0]*points[:,1])
#     sum_xz.index_add_(0, inv_idx, points[:,0]*points[:,2])
#     sum_yz.index_add_(0, inv_idx, points[:,1]*points[:,2])

#     # 4) 평균, 공분산
#     counts_clamped = counts.clamp_min(1e-9)
#     means = sum_xyz / counts_clamped.unsqueeze(-1)  # (K,3)
#     Exx = sum_x2 / counts_clamped
#     Eyy = sum_y2 / counts_clamped
#     Ezz = sum_z2 / counts_clamped
#     Exy = sum_xy / counts_clamped
#     Exz = sum_xz / counts_clamped
#     Eyz = sum_yz / counts_clamped

#     mx = means[:,0]
#     my = means[:,1]
#     mz = means[:,2]

#     covs = torch.zeros(K,3,3, device=device, dtype=dtype)
#     covs[:,0,0] = Exx - mx*mx
#     covs[:,1,1] = Eyy - my*my
#     covs[:,2,2] = Ezz - mz*mz
#     covs[:,0,1] = Exy - mx*my; covs[:,1,0] = covs[:,0,1]
#     covs[:,0,2] = Exz - mx*mz; covs[:,2,0] = covs[:,0,2]
#     covs[:,1,2] = Eyz - my*mz; covs[:,2,1] = covs[:,1,2]

#     # Symmetrize + regularize
#     I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
#     covs = 0.5*(covs + covs.transpose(1,2)) + epsilon*I3

#     # 5) min_points 이상인 보셀만 필터
#     mask = (counts >= min_points)
#     if not mask.any():
#         print("[Warning] No voxel meets min_points => return empty.")
#         return (torch.empty((0,3), dtype=torch.int64, device=device),
#                 torch.empty((0,3), dtype=dtype, device=device),
#                 torch.empty((0,3,3), dtype=dtype, device=device))

#     keep_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
#     covs = covs[keep_idx]
#     means = means[keep_idx]
#     good_keys = unique_keys[keep_idx]

#     # 복원: vx,vy,vz
#     vx_final = (good_keys % 100000) - offset_val
#     vy_final = ((good_keys // 100000) % 100000) - offset_val
#     vz_final = ((good_keys // (100000**2)) % 100000) - offset_val
#     voxel_idxs = torch.stack([vx_final, vy_final, vz_final], dim=1).to(torch.int32)

#     return voxel_idxs, means, covs


# overlap ndt only register code

# def find_ndt_overlap_idx(ndtA, ndtB):
#     """
#     ndtA=(vxA, meansA, covA), ndtB=(vxB, meansB, covB)
#     정수 voxel idx가 동일한 것만 추출 => Overlap
#     """
#     vxA, _, _ = ndtA
#     vxB, _, _ = ndtB
#     offset = 500
#     keyA = (vxA[:,0]+offset) + (vxA[:,1]+offset)*1000 + (vxA[:,2]+offset)*1000000
#     keyB = (vxB[:,0]+offset) + (vxB[:,1]+offset)*1000 + (vxB[:,2]+offset)*1000000

#     dictB = {}
#     for i in range(keyB.size(0)):
#         dictB[keyB[i].item()] = i

#     A_list, B_list = [], []
#     for i in range(keyA.size(0)):
#         k = keyA[i].item()
#         if k in dictB:
#             A_list.append(i)
#             B_list.append(dictB[k])

#     if len(A_list)==0:
#         return None, None

#     device = vxA.device
#     A_idx = torch.tensor(A_list, device=device, dtype=torch.long)
#     B_idx = torch.tensor(B_list, device=device, dtype=torch.long)
#     return A_idx, B_idx


# def compute_grad_hess_quadratic_batch(ndtA, ndtB, R, t):
#     """
#     Quadratic cost:
#       cost = sum_i ( diff_i^T W_i diff_i ),
#       W_i= (R covA[i] R^T + covB[i])^-1
#       diff_i= R muA[i] + t - muB[i]

#     Gauss-Newton 근사 => grad, hess in batch
#     - cost_i >=0, sum => cost >=0
#     - partial( cost_i )= 2 diff^T W partial diff => shape (1,3)* J_diff(3,6)= (1,6)
#     - sum across i => (6,)
#     - Hess => sum_i( outer(grad_i, grad_i) ) (GN 근사).
#     """
#     device = R.device
#     vxA, meansA, covA= ndtA
#     vxB, meansB, covB= ndtB

#     # 1) Overlap
#     A_idx, B_idx= find_ndt_overlap_idx(ndtA, ndtB)
#     if A_idx is None:
#         grad= torch.zeros(6, device=device, dtype=R.dtype)
#         hess= torch.zeros((6,6), device=device, dtype=R.dtype)
#         return grad, hess, 0.0

#     # 2) gather data
#     muA = meansA[A_idx].to(device=device, dtype=R.dtype)   # (K,3)
#     SigA= covA[A_idx].to(device=device, dtype=R.dtype)     # (K,3,3)
#     muB = meansB[B_idx].to(device=device, dtype=R.dtype)   # (K,3)
#     SigB= covB[B_idx].to(device=device, dtype=R.dtype)     # (K,3,3)
#     K= muA.size(0)

#     # 3) diff= (R muA + t - muB)
#     R_ = R.unsqueeze(0)           # shape(1,3,3)
#     muA_rot= muA @ R.transpose(-1,-2) + t.view(1,3)  # (K,3)
#     diff= muA_rot - muB                               # (K,3)

#     # 4) comb= R SigA R^T + SigB => shape(K,3,3)
#     R_expand= R_.expand(K,3,3)
#     covA_rot= R_expand.bmm(SigA).bmm(R_expand.transpose(1,2))  # (K,3,3)
#     comb= covA_rot + SigB                                      # (K,3,3)

#     # invC => W
#     invC= torch.linalg.inv(comb)  # (K,3,3)

#     # 5) cost= sum_i( diff^T W diff ) => do batch
#     # diff shape => (K,3), W => (K,3,3)
#     # => x_i= diff^T W diff => flatten => (K,)
#     diff_ = diff.unsqueeze(1)   # (K,1,3)
#     left  = torch.bmm(diff_, invC) # (K,1,3)
#     quad_ = torch.bmm(left, diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (K,)
#     cost_val= quad_.sum().item()

#     # 6) gradient
#     # partial( cost_i )= 2 diff^T W partial( diff )
#     # => grad_x= 2 invC diff => shape(K,3)
#     diff3= diff.unsqueeze(-1)  # (K,3,1)
#     grad_x= 2.0* torch.bmm(invC, diff3).squeeze(-1)  # (K,3)

#     # partial( diff )/ partial( param ) => J_diff= [ -R skew(muA ),  I ]
#     # build in batch
#     # skew( muA ) => shape(K,3,3)
#     skew_mat= muA.new_zeros(K,3,3)
#     skew_mat[:,0,1]= -muA[:,2]
#     skew_mat[:,0,2]=  muA[:,1]
#     skew_mat[:,1,0]=  muA[:,2]
#     skew_mat[:,1,2]= -muA[:,0]
#     skew_mat[:,2,0]= -muA[:,1]
#     skew_mat[:,2,1]=  muA[:,0]

#     dRmu= - torch.bmm(R_expand, skew_mat)  # (K,3,3)
#     I3_batch= torch.eye(3, device=device, dtype=R.dtype).expand(K,3,3)
#     # cat => (K,3,6)
#     J_diff= torch.cat([dRmu, I3_batch], dim=2)

#     # => grad_x shape(K,3), convert => (K,1,3)
#     grad_x_ = grad_x.unsqueeze(1)  # (K,1,3)
#     # => grad_x_i= (K,1,3) x (K,3,6)= (K,1,6)->(K,6)
#     grad_x_i= torch.bmm(grad_x_, J_diff).squeeze(1)  # (K,6)

#     # sum => shape(6)
#     grad_out= grad_x_i.sum(dim=0)

#     # 7) Hessian => sum_i( outer( grad_x_i, grad_x_i ) )
#     # => shape(K,6,6)-> sum->(6,6)
#     g1= grad_x_i.unsqueeze(2)  # (K,6,1)
#     g2= grad_x_i.unsqueeze(1)  # (K,1,6)
#     hess_batch= g1* g2         # (K,6,6)
#     hess_out= hess_batch.sum(dim=0)

#     return grad_out, hess_out, cost_val


# def register_ndt_quadratic_gn_LM(
#     ndtA, ndtB,
#     max_iter=30,
#     init_rotation=None,
#     init_translation=None,
#     tolerance=1e-6,
#     device=None,
#     dtype=None
# ):
#     """
#     Levenberg-Marquardt for Quadratic cost:
#       cost = sum_i diff_i^T [inv(R covA R^T + covB)] diff_i
#     Overlap-based approach => only idx match
#     """
#     if device is None:
#         device = ndtA[1].device
#     if dtype is None:
#         dtype = ndtA[1].dtype

#     # 초기 파라미터 설정
#     if init_rotation is None:
#         R = torch.eye(3, device=device, dtype=dtype)
#     else:
#         R = init_rotation.to(device=device, dtype=dtype)
#     if init_translation is None:
#         t = torch.zeros((3,1), device=device, dtype=dtype)
#     else:
#         t = init_translation.to(device=device, dtype=dtype)

#     lm_lambda = 0.01

#     # ----------------------------
#     # 1) 초기 cost 계산 -> old_cost
#     # ----------------------------
#     grad_init, hess_init, old_cost = compute_grad_hess_quadratic_batch(ndtA, ndtB, R, t)

#     for step in range(max_iter):
#         # 현재 파라미터에 대한 grad, hess, cost
#         grad, hess, cost = compute_grad_hess_quadratic_batch(ndtA, ndtB, R, t)

#         rank = torch.linalg.matrix_rank(hess)
#         if rank < 6:
#             print(f"[Info] Hessian rank < 6 at step={step}, break.")
#             break

#         # LM 뎀핑
#         dampI = lm_lambda * torch.eye(6, device=device, dtype=dtype)
#         hess_damped = hess + dampI

#         try:
#             delta = torch.linalg.solve(hess_damped, -grad)
#         except RuntimeError:
#             # singular => lambda 증가 후 재시도
#             lm_lambda *= 10
#             continue

#         d_omega = delta[:3]
#         d_trans = delta[3:]
#         R_up = expmap(d_omega)  # external 'expmap' 함수
#         R_new = R_up @ R
#         t_new = t + d_trans.view(3,1)

#         # 새 파라미터에 대한 cost
#         _, _, new_cost = compute_grad_hess_quadratic_batch(ndtA, ndtB, R_new, t_new)

#         print("lm_lambda:", lm_lambda)
#         print(f"old_cost = {old_cost}, cost(this param)={cost}, new_cost={new_cost}")

#         # ----------------------------
#         # 2) 개선 여부 판단
#         # ----------------------------
#         if new_cost < old_cost:
#             # 개선 => accept 파라미터
#             R = R_new
#             t = t_new

#             # (중요) old_cost 갱신
#             old_cost = new_cost

#             # lambda 감소
#             lm_lambda *= 0.5

#             # 수렴 여부 확인
#             # 여기서는 old_cost vs new_cost가 거의 같은지로 판단해도 되고,
#             # 혹은 cost 차이를 비교해도 됨
#             if abs(new_cost - cost) < tolerance:
#                 print(f"[Info] Converged at step={step}, cost={new_cost}")
#                 break
#         else:
#             # revert => R,t는 그대로. lambda 증가
#             lm_lambda *= 5

#     return R, t




# def find_ndt_overlap_idx(ndtA, ndtB):
#     vxA, _, _ = ndtA
#     vxB, _, _ = ndtB
#     offset = 500
#     keyA = (vxA[:,0]+offset) + (vxA[:,1]+offset)*1000 + (vxA[:,2]+offset)*1000000
#     keyB = (vxB[:,0]+offset) + (vxB[:,1]+offset)*1000 + (vxB[:,2]+offset)*1000000

#     dictB = {}
#     for i in range(keyB.size(0)):
#         dictB[keyB[i].item()] = i

#     A_list, B_list = [], []
#     for i in range(keyA.size(0)):
#         k = keyA[i].item()
#         if k in dictB:
#             A_list.append(i)
#             B_list.append(dictB[k])

#     if len(A_list)==0:
#         return None, None

#     device = vxA.device
#     A_idx = torch.tensor(A_list, device=device, dtype=torch.long)
#     B_idx = torch.tensor(B_list, device=device, dtype=torch.long)
#     return A_idx, B_idx


# def compute_grad_hess_exp_LM_batch(
#     ndtA, ndtB, R, t
# ):
#     """
#     Batch-based version, no Python for-loop over K.
#     cost = sum( 1 - val_i ), val_i= factor_i * exp(-0.5 x_i)
#     => grad = sum( partial( cost_i ) ), hess= sum( outer(grad_i,grad_i) ) in GN approx
#     """
#     device= R.device
#     vxA, meansA, covA= ndtA
#     vxB, meansB, covB= ndtB

#     A_idx, B_idx= find_ndt_overlap_idx(ndtA, ndtB)
#     if A_idx is None:
#         grad= torch.zeros(6, device=device, dtype=R.dtype)
#         hess= torch.zeros((6,6), device=device, dtype=R.dtype)
#         return grad, hess, 0.0

#     muA = meansA[A_idx].to(device=device, dtype=R.dtype)   # (K,3)
#     SigA= covA[A_idx].to(device=device, dtype=R.dtype)     # (K,3,3)
#     muB = meansB[B_idx].to(device=device, dtype=R.dtype)   # (K,3)
#     SigB= covB[B_idx].to(device=device, dtype=R.dtype)     # (K,3,3)

#     K= muA.size(0)

#     # 1) batch transform
#     R_ = R.unsqueeze(0)              # (1,3,3)
#     R_expand= R_.expand(K,3,3)       # (K,3,3)

#     # diff
#     muA_trans= muA @ R.transpose(-1,-2) + t.view(1,3)  # (K,3)
#     diff= muA_trans - muB                              # (K,3)

#     # comb= R SigmaA R^T + SigB
#     covA_trans= R_expand.bmm(SigA).bmm(R_expand.transpose(1,2))  # (K,3,3)
#     comb= covA_trans+ SigB                                        # (K,3,3)

#     detC= torch.linalg.det(comb).clamp_min(1e-12)   # (K,)
#     invC= torch.linalg.inv(comb)                    # (K,3,3)

#     two_pi_3half= (2*math.pi)**1.5
#     factor_= 1.0/( two_pi_3half* torch.sqrt(detC))  # (K,)

#     # x_i= diff^T invC diff => (K,)
#     diff_ = diff.unsqueeze(1)   # (K,1,3)
#     left  = torch.bmm(diff_, invC)  # (K,1,3)
#     quad_ = torch.bmm(left, diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (K,)
#     exponent_ = torch.exp(-0.5* quad_)  # (K,)

#     val_ = factor_ * exponent_          # (K,)
#     # cost_i= 1- val_i => cost= sum( cost_i )
#     cost_ = (1.0- val_).clamp_min(0.0)  # optionally clamp min=0
#     cost_val= cost_.sum().item()

#     # 2) gradient
#     # partial( cost_i )= - partial( val_i ) => val_i= factor_i * p_i => ignoring partial factor wrt param
#     # partial( p_i )= p_i( -0.5 ) grad_x => grad_x= 2 invC diff => => p_i(-0.5)(2 invC diff)
#     # => partial( val_i )= factor_i * partial( p_i ) => factor_i is constant wrt param => so val_i * -0.5 => 
#     # => grad_term_i= - partial(val_i )= - [val_i*( -0.5)* grad_x_i ]= val_i*0.5*grad_x_i
#     # Actually let's define directly:
#     # val_i= fac_i * p_i => partial( val_i )= fac_i*[ p_i*( -0.5)* grad_x_i ] => => val_i*( -0.5)* grad_x_i
#     # cost_i= (1- val_i ) => partial( cost_i )= - partial( val_i ) => => +0.5 val_i * grad_x_i
#     # sign carefully => watch code

#     # compute grad_x= 2 invC diff => shape(K,3)
#     grad_x = 2.0* torch.einsum('kij,kj->ki', invC, diff)   # (K,3)
#     # Alternatively:
#     # grad_x= (invC * diff). We can do bmm: shape(K,3,3)* (K,3,1)= (K,3,1)...

#     # build J_diff= [-R skew(muA), I3], shape(K,3,6)
#     # skew(muA) => (K,3,3)
#     # RExpand bmm skew => shape(K,3,3) => negative
#     # We do everything in one batch
#     # 1) skew_mA= skew( muA[i] ) for each i => we can do a direct formula or a gather approach

#     # We do 'skew' in batch => we can do a small custom kernel or do it by gather:
#     # We'll do: skew_mA= torch.zeros(K,3,3, device=..., dtype=...) and fill with muA. 
#     skew_mat= muA.new_zeros(K,3,3)
#     # fill
#     # approach => 
#     skew_mat[:,0,1]= -muA[:,2]
#     skew_mat[:,0,2]=  muA[:,1]
#     skew_mat[:,1,0]=  muA[:,2]
#     skew_mat[:,1,2]= -muA[:,0]
#     skew_mat[:,2,0]= -muA[:,1]
#     skew_mat[:,2,1]=  muA[:,0]

#     dRmu_batch= - torch.bmm(R_expand, skew_mat)  # (K,3,3)
#     # I3_batch => shape(K,3,3)= repeated Identity
#     I3_batch= torch.eye(3, device=device, dtype=R.dtype).expand(K,3,3)
#     # J_diff => cat along last dimension => shape(K,3,6)
#     J_diff= torch.cat([dRmu_batch, I3_batch], dim=2)

#     # grad_x_i => shape(K,6):
#     # grad_x( K,3 ) => (K,1,3)
#     grad_x_ = grad_x.unsqueeze(1)          # (K,1,3)
#     # (K,1,3) x (K,3,6)= (K,1,6)
#     grad_x_i= torch.bmm( grad_x_, J_diff ).squeeze(1)   # (K,6)

#     # cost_i= 1- val_i => partial( cost_i )= - partial( val_i ). 
#     # partial( val_i )= val_i*( -0.5)* grad_x_i => => => let's define sign carefully:
#     # => grad_term= partial( cost_i )= - [ val_i*( -0.5)* grad_x_i ]= val_i*0.5* grad_x_i
#     # 
#     # So let's define:
#     grad_term= 0.5* ( val_.unsqueeze(1)* grad_x_i )  # shape(K,6)

#     # sum => shape(6)
#     grad_out= grad_term.sum(dim=0)

#     # 3) Hessian => for each i => outer( grad_term_i, grad_term_i ) => (K,6,6). sum => (6,6)
#     # we can do: hess= torch.einsum('ki,kj->ij', grad_term, grad_term)
#     # or do an expand + multiply:
#     # shape => (K,6,1)*(K,1,6)= (K,6,6). sum across K => (6,6)
#     # but watch memory usage => we said memory is not an issue => let's do it
#     # grad_term => (K,6)
#     gt_expand1= grad_term.unsqueeze(2)   # (K,6,1)
#     gt_expand2= grad_term.unsqueeze(1)   # (K,1,6)
#     hess_batch= gt_expand1* gt_expand2   # (K,6,6)
#     hess_out= hess_batch.sum(dim=0)      # (6,6)

#     return grad_out, hess_out, cost_val


# def register_ndt_exponential_LM_fast(
#     ndtA, ndtB,
#     max_iter=30,
#     init_rotation=None,
#     init_translation=None,
#     tolerance=1e-6,
#     device=None,
#     dtype=None
# ):
#     """
#     LM approach + full batch vectorization for speed
#     cost = sum( 1 - factor_i * exp(-0.5 x_i) ), partial derivatives in batch
#     => fewer python loops => faster
#     """
#     if device is None:
#         device= ndtA[1].device
#     if dtype is None:
#         dtype= ndtA[1].dtype

#     if init_rotation is None:
#         R= torch.eye(3, device=device, dtype=dtype)
#     else:
#         R= init_rotation.to(device=device, dtype=dtype)
#     if init_translation is None:
#         t= torch.zeros((3,1), device=device, dtype=dtype)
#     else:
#         t= init_translation.to(device=device, dtype=dtype)

#     lm_lambda= 1.0
#     prev_cost= float('inf')
#     for step in range(max_iter):
#         grad, hess, cost= compute_grad_hess_exp_LM_batch(ndtA, ndtB, R, t)

#         # LM damping
#         rank= torch.linalg.matrix_rank(hess)
#         if rank<6:
#             break

#         damp_I= lm_lambda* torch.eye(6, device=device, dtype=dtype)
#         hess_damped= hess+ damp_I
#         try:
#             delta= torch.linalg.solve(hess_damped, -grad)
#         except RuntimeError:
#             lm_lambda*=10
#             continue

#         # update param
#         d_omega= delta[:3]
#         d_trans= delta[3:]
#         R_up= expmap(d_omega)  # external 'expmap'
#         R_new= R_up@ R
#         t_new= t+ d_trans.view(3,1)

#         # check cost improvement
#         grad2, hess2, new_cost= compute_grad_hess_exp_LM_batch(ndtA, ndtB, R_new, t_new)
        
#         # print("prev_cost:", prev_cost, "new_cost:", new_cost)
        
#         if new_cost< cost:
#             # improvement => accept
#             R= R_new
#             t= t_new
#             lm_lambda*=0.5
#             if abs(prev_cost- new_cost)< tolerance:
#                 break
#             prev_cost= new_cost
#         else:
#             # revert, up lambda
#             lm_lambda*=5

#     return R, t



























# # L²-based objective function
# def compute_objective(ndt1, ndt2, R, t, device, dtype):
#     loss = torch.tensor(0.0, device=device, dtype=dtype)
#     overlapping_keys = set(ndt1.keys()) & set(ndt2.keys())

#     for key in overlapping_keys:
#         mean1, cov1 = ndt1[key]
#         mean2, cov2 = ndt2[key]

#         mean1 = mean1.unsqueeze(1)  # [3, 1]
#         transformed_mean1 = R @ mean1 + t.view(3, 1)  # [3, 1]
#         transformed_cov1 = R @ cov1 @ R.T  # [3, 3]

#         diff = transformed_mean1 - mean2.unsqueeze(1)  # [3, 1]
#         combined_cov = transformed_cov1 + cov2
#         inv_combined_cov = torch.linalg.inv(combined_cov)

#         exponent = -0.5 * diff.T @ inv_combined_cov @ diff  # Scalar
#         loss += torch.exp(exponent).squeeze()  # Accumulate loss

#     return -loss  # Negative similarity

# # Jacobian, residual, and weights
# def compute_jacobian_and_residual(ndt1, ndt2, R, t):
#     jacobians = []
#     residuals = []
#     weights = []
#     overlapping_keys = set(ndt1.keys()) & set(ndt2.keys())

#     for key in overlapping_keys:
#         mean1, cov1 = ndt1[key]
#         mean2, cov2 = ndt2[key]

#         mean1_trans = R @ mean1.unsqueeze(1) + t.view(3, 1)
#         cov1_trans = R @ cov1 @ R.T
#         residual = (mean1_trans - mean2.unsqueeze(1)).squeeze()

#         skew_mean1 = torch.tensor([[0, -mean1[2], mean1[1]],
#                                     [mean1[2], 0, -mean1[0]],
#                                     [-mean1[1], mean1[0], 0]], device=mean1.device, dtype=mean1.dtype)
#         J_rot = -R @ skew_mean1
#         J_trans = torch.eye(3, device=mean1.device, dtype=mean1.dtype)
#         jacobian = torch.cat([J_rot, J_trans], dim=1)

#         combined_cov = cov1_trans + cov2
#         inv_combined_cov = torch.linalg.inv(combined_cov)

#         jacobians.append(jacobian.unsqueeze(0))
#         residuals.append(residual.unsqueeze(0))
#         weights.append(inv_combined_cov)  # 2D로 저장

#     jacobians = torch.cat(jacobians, dim=0)
#     residuals = torch.cat(residuals, dim=0).view(-1, 1)

#     # 2D로 저장된 weights를 block_diag로 결합
#     weights = torch.block_diag(*weights)

#     jacobians = jacobians.view(-1, 6)

#     return jacobians, residuals, weights


# # NDT registration
# def register_ndts(ndt1, ndt2, device, dtype, max_iter=100, lm_lambda=1.0,
#                   init_rotation=None, init_translation=None, tolerance=1e-6):
#     R = init_rotation if init_rotation is not None else torch.eye(3, device=device, dtype=dtype)
#     t = init_translation if init_translation is not None else torch.zeros((3, 1), device=device, dtype=dtype)

#     prev_loss = float('inf')
#     for step in range(max_iter):
#         jacobians, residuals, weights = compute_jacobian_and_residual(ndt1, ndt2, R, t)
#         N_mat = jacobians.T @ weights @ jacobians
#         g_vec = -(jacobians.T @ weights @ residuals).squeeze()
#         N_mat += lm_lambda * torch.eye(N_mat.size(0), device=device, dtype=dtype)

#         try:
#             t_vec = torch.linalg.solve(N_mat, g_vec)
#         except RuntimeError:
#             lm_lambda *= 10
#             continue

#         R_update = expmap(t_vec[:3])
#         R = R_update @ R
#         t += t_vec[3:].view(3, 1)

#         loss = compute_objective(ndt1, ndt2, R, t, device, dtype)
#         if step > 0 and abs(prev_loss - loss) < tolerance:
#             break

#         lm_lambda = lm_lambda * 0.1 if loss < prev_loss else lm_lambda * 10
#         prev_loss = loss

#     return R, t

























# def create_ndt_from_pointcloud_fast(surface_points, voxel_size=3, min_num=10, epsilon=1e-2):
#     """
#     Optimized NDT map creation with improved stability for covariance matrices.

#     Args:
#         surface_points (torch.Tensor): Points near the surface (Mx3).
#         voxel_size (float): Size of each voxel.
#         epsilon (float): Regularization factor for covariance matrices.

#     Returns:
#         ndt_map (dict): NDT map with voxel indices as keys and (mean, covariance) as values.
#     """
#     device = surface_points.device
#     offset = 500  # Offset to handle negative coordinates safely

#     # Compute voxel indices and keys with offset
#     voxel_indices = torch.floor(surface_points / voxel_size).to(torch.int32)  # (M, 3)
#     voxel_keys = ((voxel_indices + offset) * torch.tensor([1, 1000, 1000000], device=device)).sum(dim=1)

#     # Unique voxel keys and inverse mapping
#     unique_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)
#     num_voxels = unique_keys.size(0)

#     # Compute voxel means and counts
#     ndt_means = torch.zeros((num_voxels, 3), device=device)
#     voxel_counts = torch.zeros(num_voxels, device=device)

#     ndt_means.index_add_(0, inverse_indices, surface_points)
#     voxel_counts.index_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32))
#     ndt_means /= voxel_counts.unsqueeze(1)

#     # Compute covariance in batches
#     diffs = surface_points - ndt_means[inverse_indices]
#     cov_updates = diffs.unsqueeze(2) @ diffs.unsqueeze(1)  # (M, 3, 3)
#     ndt_covariances = torch.zeros((num_voxels, 3, 3), device=device)
#     ndt_covariances.index_add_(0, inverse_indices, cov_updates)
#     ndt_covariances /= voxel_counts.view(-1, 1, 1)
    
#     # Regularize covariance matrices in batches
#     I = torch.eye(3, device=device).unsqueeze(0)  # (1, 3, 3)
#     ndt_covariances = 0.5 * (ndt_covariances + ndt_covariances.transpose(1, 2))  # Ensure symmetry
#     eigvals, eigvecs = torch.linalg.eigh(ndt_covariances + epsilon * I)  # Add epsilon regularization

#     # Check Eigenvalues for each covariance matrix
#     for i, voxel_key in enumerate(unique_keys.tolist()):
#         if torch.any(eigvals[i] < 0):
#             print(f"Warning: Negative eigenvalue detected in voxel {voxel_key}: {eigvals[i]}")

#     eigvals = torch.clamp(eigvals, min=epsilon)  # Clamp eigenvalues directly
#     ndt_covariances = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(1, 2)
#     ndt_covariances = 0.5 * (ndt_covariances + ndt_covariances.transpose(1, 2))  # Re-symmetrize

#     # Build NDT map
#     ndt_map = {}
#     voxel_indices_3d = torch.stack([
#         unique_keys % 1000 - offset,
#         (unique_keys // 1000) % 1000 - offset,
#         (unique_keys // 1000000) % 1000 - offset
#     ], dim=1)

#     for i, voxel_idx in enumerate(voxel_indices_3d.tolist()):
#         if voxel_counts[i] >= min_num:
#             ndt_map[tuple(voxel_idx)] = (ndt_means[i], ndt_covariances[i])

#     return ndt_map

# def create_ndt_overlap_ultrafast(surface_points: torch.Tensor,
#                                  voxel_size: int = 5,
#                                  min_num: int = 10,
#                                  epsilon: float = 1e-2):
#     """
#     Overlap NDT를 빠르게 생성하는 예시 코드.
#     - 각 점이 (x,y,z)축에서 voxel_size=5만큼 슬라이딩(격자간격=1)으로 
#       최대 5개씩 후보 voxel 시작점을 갖는다.
#     - 즉 한 점당 최대 125개의 voxel에 중복 삽입.
#     - Python dict 루프 없이, PyTorch GPU 연산(index_add_)으로 평균/공분산 계산.

#     Args:
#         surface_points (torch.Tensor): (M,3) 형태의 입력 점들 (GPU상).
#         voxel_size (int): 슬라이딩 윈도우 크기 (기본 5).
#         min_num (int): 각 voxel에 최소 얼마나 많은 점이 있어야 NDT로 쓸지.
#         epsilon (float): 공분산 정규화용 파라미터.

#     Returns:
#         ndt_map (dict):
#             key = (vx,vy,vz) (정수 좌표; voxel 시작점)
#             value = (mean (3,), covariance(3,3)) 로 구성
#     """

#     device = surface_points.device
#     dtype = surface_points.dtype

#     # -----------------------------
#     # 1) 각 점의 바닥(floor) 정수 좌표 계산
#     #    ex) p=(10.2, 3.7, -2.1) -> floor=(10, 3, -3)
#     # -----------------------------
#     p_floor = torch.floor(surface_points).to(torch.int32)  # (M,3)
#     px_floor = p_floor[:, 0]  # (M,)
#     py_floor = p_floor[:, 1]
#     pz_floor = p_floor[:, 2]

#     M = surface_points.size(0)

#     # 오프셋(예: 500). 음수 좌표나 큰 좌표를 양수로 매핑하기 위함
#     # (x+offset, y+offset, z+offset)이 최대 0~999 범위 내에 들어갈지 여부는
#     # 사용자가 점 범위에 맞춰 적절히 조정해야 함
#     offset = 500

#     # -----------------------------
#     # 2) (0..voxel_size-1) 범위의 오프셋들로 3D meshgrid를 구성
#     #    xgrid, ygrid, zgrid  -> shape [voxel_size, voxel_size, voxel_size]
#     # -----------------------------
#     offsets_1d = torch.arange(voxel_size, device=device, dtype=torch.int32)  # [0,1,2,3,4] (기본 5개)
#     xgrid, ygrid, zgrid = torch.meshgrid(
#         offsets_1d, offsets_1d, offsets_1d, indexing='ij'
#     )
#     # xgrid.shape == (voxel_size, voxel_size, voxel_size) 등등

#     # -----------------------------
#     # 3) 각 점에 대해 슬라이딩된 voxel 시작점 (vx,vy,vz) 계산
#     #
#     #   vx = floor(px) - xgrid
#     #   vy = floor(py) - ygrid
#     #   vz = floor(pz) - zgrid
#     #
#     #   => shape: (M, voxel_size, voxel_size, voxel_size)
#     # -----------------------------
#     # 아래처럼 broadcast 연산:
#     #   px_floor.view(M,1,1,1) : (M,1,1,1)
#     #   xgrid.view(1,v,v,v)    : (1,5,5,5)
#     # => 결과: (M,5,5,5)
#     VX = px_floor.view(M, 1, 1, 1) - xgrid.view(1, voxel_size, voxel_size, voxel_size)
#     VY = py_floor.view(M, 1, 1, 1) - ygrid.view(1, voxel_size, voxel_size, voxel_size)
#     VZ = pz_floor.view(M, 1, 1, 1) - zgrid.view(1, voxel_size, voxel_size, voxel_size)

#     # flatten -> (M*voxel_size^3,)
#     VX = VX.view(-1)  # [M*v^3]
#     VY = VY.view(-1)
#     VZ = VZ.view(-1)

#     # offset 적용
#     voxel_x = VX + offset
#     voxel_y = VY + offset
#     voxel_z = VZ + offset

#     # 1D key = x + y*1000 + z*1000000
#     # (원하는 해시 공식에 맞춰서 조정 가능)
#     voxel_keys = voxel_x + voxel_y * 1000 + voxel_z * 1000000  # (M*v^3,)

#     # -----------------------------
#     # 4) 중복 확장된 점 좌표 준비
#     #
#     #   surface_points: (M,3)
#     #   -> expand/broadcast -> (M, voxel_size, voxel_size, voxel_size, 3)
#     #   -> flatten -> (M*voxel_size^3, 3)
#     # -----------------------------
#     expanded_points = (
#         surface_points
#         .view(M, 1, 1, 1, 3)
#         .expand(-1, voxel_size, voxel_size, voxel_size, -1)
#         .contiguous()
#         .view(-1, 3)
#     )

#     # -----------------------------
#     # 5) unique + inverse_indices 로 voxel 단위로 점들을 그룹화
#     #    -> index_add_로 mean, cov 계산
#     # -----------------------------
#     unique_keys, inverse = torch.unique(voxel_keys, return_inverse=True)
#     num_voxels = unique_keys.size(0)

#     # (a) point count
#     counts = torch.zeros(num_voxels, device=device, dtype=dtype)
#     counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=dtype))

#     # (b) sum of coords -> mean
#     sums = torch.zeros((num_voxels, 3), device=device, dtype=dtype)
#     sums.index_add_(0, inverse, expanded_points)
#     means = sums / counts.unsqueeze(1).clamp_min(1e-9)  # (num_voxels,3)

#     # (c) sum of outer products -> covariance
#     diffs = expanded_points - means[inverse]  # (M*v^3, 3)
#     cov_updates = diffs.unsqueeze(2) * diffs.unsqueeze(1)  # (M*v^3, 3,3)

#     covariances = torch.zeros((num_voxels, 3, 3), device=device, dtype=dtype)
#     covariances.index_add_(0, inverse, cov_updates)
#     covariances /= counts.view(-1, 1, 1).clamp_min(1e-9)

#     # -----------------------------
#     # 6) 공분산 정규화(대칭화, eigen값 clamp 등)
#     # -----------------------------
#     I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
#     covariances = 0.5 * (covariances + covariances.transpose(1, 2))
#     eigvals, eigvecs = torch.linalg.eigh(covariances + epsilon * I)
#     eigvals = torch.clamp(eigvals, min=epsilon)
#     covariances = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
#     covariances = 0.5 * (covariances + covariances.transpose(1, 2))

#     # -----------------------------
#     # 7) ndt_map(dict)으로 정리
#     #    counts < min_num인 voxel은 버림
#     # -----------------------------
#     ndt_map = {}

#     vx_ = (unique_keys % 1000)       # (num_voxels,)
#     vy_ = (unique_keys // 1000) % 1000
#     vz_ = (unique_keys // 1000000) % 1000

#     voxel_idxs_3d = torch.stack([vx_, vy_, vz_], dim=1) - offset  # (num_voxels,3)
    
#     for i in range(num_voxels):
#         if counts[i] < min_num:
#             continue
#         key_3d = tuple(voxel_idxs_3d[i].tolist())  # (vx,vy,vz)
#         ndt_map[key_3d] = (means[i], covariances[i])

#     return ndt_map




# # Optimized rotation matrix calculation
# def rotation_matrix(euler_angles, device, dtype):
#     rx, ry, rz = euler_angles
#     # Precompute sines and cosines
#     sin_rx, cos_rx = torch.sin(rx), torch.cos(rx)
#     sin_ry, cos_ry = torch.sin(ry), torch.cos(ry)
#     sin_rz, cos_rz = torch.sin(rz), torch.cos(rz)

#     Rx = torch.tensor([
#         [1, 0, 0],
#         [0, cos_rx, -sin_rx],
#         [0, sin_rx, cos_rx]
#     ], dtype=dtype, device=device)

#     Ry = torch.tensor([
#         [cos_ry, 0, sin_ry],
#         [0, 1, 0],
#         [-sin_ry, 0, cos_ry]
#     ], dtype=dtype, device=device)

#     Rz = torch.tensor([
#         [cos_rz, -sin_rz, 0],
#         [sin_rz, cos_rz, 0],
#         [0, 0, 1]
#     ], dtype=dtype, device=device)

#     return Rz @ Ry @ Rx

# # Optimized L2 distance with Cholesky decomposition
# def gaussian_l2_distance(mean1, cov1, mean2, cov2):
#     diff_mean = mean1 - mean2  # [3, 1]
#     combined_cov = cov1 + cov2  # [3, 3]
    
#     # Cholesky decomposition for determinant and inverse
#     L = torch.linalg.cholesky(combined_cov)
#     inv_cov = torch.cholesky_inverse(L)
#     log_det = 2 * torch.sum(torch.log(torch.diag(L)))  # log(det)

#     # Mahalanobis distance exponent
#     exponent = -0.5 * diff_mean.T @ inv_cov @ diff_mean  # Scalar tensor
#     norm_factor = torch.exp(-0.5 * log_det)
#     return (torch.exp(exponent) / norm_factor).squeeze()

# # Optimized compute_objective with batched processing
# def compute_objective(ndt1, ndt2, transformation, device, dtype):
#     R, t = transformation
#     loss = torch.tensor(0.0, device=device, dtype=dtype)
    
#     # Compute overlapping keys
#     overlapping_keys = set(ndt1.keys()) & set(ndt2.keys())
    
#     # Compare only overlapping NDT cells
#     for key in overlapping_keys:
#         mean1, cov1 = ndt1[key]
#         mean2, cov2 = ndt2[key]

#         # Transform mean1 and cov1
#         mean1 = mean1.unsqueeze(1)
#         t = t.view(3, 1)
#         transformed_mean1 = R @ mean1 + t
#         transformed_cov1 = R @ cov1 @ R.T

#         # Compute Gaussian L2 distance and accumulate loss
#         mean2 = mean2.unsqueeze(1)
#         loss += gaussian_l2_distance(transformed_mean1, transformed_cov1, mean2, cov2)
    
#     return -loss  # Minimize negative similarity

    
#     return -loss  # Minimize negative similarity

# from torch.optim.lr_scheduler import StepLR

# def register_ndts(
#     ndt1, ndt2, device, dtype, max_iter=100, initial_lr=0.01, 
#     init_params=None, tolerance=1e-6
# ):
#     # Initialize transformation parameters (translation + Euler angles)
#     if init_params is None:
#         init_params = torch.zeros(6, dtype=dtype, device=device)
#     else:
#         init_params = torch.tensor(init_params, dtype=dtype, device=device)
    
#     params = init_params.clone().detach().requires_grad_(True)
#     optimizer = torch.optim.Adam([params], lr=initial_lr)
#     scheduler = StepLR(optimizer, step_size=10, gamma=0.8)  # Decay LR by 0.8 every 10 steps

#     prev_loss = float('inf')  # Initialize previous loss to infinity
#     for step in range(max_iter):
#         optimizer.zero_grad()

#         # Rotation and translation
#         R = rotation_matrix(params[3:], device, dtype)
#         t = params[:3].view(3, 1)

#         # Compute loss
#         loss = compute_objective(ndt1, ndt2, (R, t), device, dtype)

#         # Early stopping check
#         if abs(prev_loss - loss.item()) < tolerance:
#             # print(f"Early stopping at step {step + 1}, Loss: {loss.item():.6f}")
#             break
#         prev_loss = loss.item()

#         # Backpropagation and optimization step
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#         # Logging
#         # print(f"Step {step + 1}/{max_iter}, Loss: {loss.item():.6f}")
#         # print(f"Parameters: {params.detach().cpu().numpy()}")

#     # Final transformation
#     R_final = rotation_matrix(params[3:], device, dtype)
#     t_final = params[:3]
#     return R_final, t_final








# def d2d_registration(source_ndt, target_ndt, device, dtype, max_iterations=20, tolerance=1e-5):
#     """
#     Perform D2D registration using NDT models with specified device and dtype.

#     Args:
#         source_ndt (dict): Source NDT {key: (mean, covariance)}.
#         target_ndt (dict): Target NDT {key: (mean, covariance)}.
#         device (torch.device): The device to perform computations on ('cpu' or 'cuda').
#         dtype (torch.dtype): The data type for computations (e.g., torch.float32).
#         max_iterations (int): Maximum number of iterations for optimization.
#         tolerance (float): Convergence threshold for transformation updates.

#     Returns:
#         torch.Tensor: Final transformation matrix (4x4).
#     """
#     # Initialize transformation matrix on the specified device and dtype
#     T = torch.eye(4, device=device, dtype=dtype)  # Initial transformation

#     def transform_gaussian(mean, covariance, T):
#         """Apply rigid transformation to Gaussian components."""
#         R = T[:3, :3]
#         t = T[:3, 3]
#         transformed_mean = (R @ mean.T).T + t
#         transformed_covariance = R @ covariance @ R.T
#         return transformed_mean, transformed_covariance

#     def compute_l2_distance(source_ndt, target_ndt, T):
#         """Compute L2 distance between two NDT models under a given transformation."""
#         l2_distance = 0
#         gradient = torch.zeros(6, device=device, dtype=dtype)
#         hessian = torch.zeros((6, 6), device=device, dtype=dtype)

#         for source_key, (source_mean, source_cov) in source_ndt.items():
#             # Transform source Gaussian
#             transformed_mean, transformed_cov = transform_gaussian(source_mean, source_cov, T)

#             if source_key in target_ndt:
#                 target_mean, target_cov = target_ndt[source_key]
#                 combined_cov = transformed_cov + target_cov
#                 combined_cov_inv = torch.linalg.pinv(combined_cov)

#                 diff = transformed_mean - target_mean
#                 exp_term = torch.exp(-0.5 * diff @ combined_cov_inv @ diff.T)

#                 # Update L2 distance
#                 l2_distance += exp_term.item()

#                 # Compute gradient and Hessian
#                 jacobian = combined_cov_inv @ diff
#                 gradient[:3] += jacobian[:3]  # Translation part
#                 # TODO: Compute rotational gradients

#         return l2_distance, gradient, hessian
    
#     # Optimization Loop
#     for iteration in range(max_iterations):
#         l2_dist, gradient, hessian = compute_l2_distance(source_ndt, target_ndt, T)

#         try:
#             delta = torch.linalg.solve(hessian, -gradient)
#         except RuntimeError:
#             print("failed")
#             hessian += 1e-6 * torch.eye(6, device=device, dtype=dtype)
#             delta = torch.linalg.solve(hessian, -gradient)

#         # Convert delta to SE(3) transformation
#         delta_T = torch.eye(4, device=device, dtype=dtype)
#         delta_T[:3, :3] = expmap(delta[:3])  # Rotation from exponential map
#         delta_T[:3, 3] = delta[3:]          # Translation
#         T = delta_T @ T

#         # Convergence check
#         if torch.norm(delta) < tolerance:
#             print(f"Converged in {iteration + 1} iterations.")
#             break

#     return T







# def d2d_registration(source_ndt, local_ndt, device, dtype, max_iterations=20, tolerance=1e-5):
#     """
#     Perform Distribution-to-Distribution (D2D) registration using NDT models.

#     Args:
#         source_points (torch.Tensor): Source point cloud (Nx3).
#         target_points (torch.Tensor): Target point cloud (Mx3).
#         voxel_size (float): Size of each voxel.
#         max_iterations (int): Maximum number of iterations for optimization.
#         tolerance (float): Convergence threshold for transformation updates.

#     Returns:
#         T (torch.Tensor): Final transformation matrix (4x4).
#     """
#     # Step 1: Initialize transformation matrix
#     T = torch.eye(4, device=device, dtype=dtype)  # Initial transformation

#     def transform_gaussian(mean, covariance, T):
#         """Apply rigid transformation to Gaussian components."""
#         R = T[:3, :3]
#         t = T[:3, 3]
#         transformed_mean = (R @ mean.T).T + t
#         transformed_covariance = R @ covariance @ R.T
#         return transformed_mean, transformed_covariance

#     def compute_l2_distance(source_ndt, target_ndt, T):
#         """Compute L2 distance between two NDT models under a given transformation."""
#         l2_distance = 0
#         gradient = torch.zeros(6, device=device, dtype=dtype)
#         hessian = torch.zeros((6, 6), device=device, dtype=dtype)

#         for source_key, (source_mean, source_cov) in source_ndt.items():
#             transformed_mean, transformed_cov = transform_gaussian(source_mean, source_cov, T)

#             if source_key in target_ndt:
#                 target_mean, target_cov = target_ndt[source_key]
#                 combined_cov = transformed_cov + target_cov
#                 combined_cov_inv = torch.linalg.pinv(combined_cov)

#                 diff = transformed_mean - target_mean
#                 exp_term = torch.exp(-0.5 * diff @ combined_cov_inv @ diff.T)

#                 # Update L2 distance
#                 l2_distance += exp_term.item()

#                 # Compute gradient and Hessian (simplified for this example)
#                 jacobian = combined_cov_inv @ diff
#                 gradient[:3] += jacobian[:3]  # Translation part
#                 # TODO: Add rotational gradient updates if necessary

#         return l2_distance, gradient, hessian

#     # Step 2: Optimization loop
#     for iteration in range(max_iterations):
#         l2_dist, gradient, hessian = compute_l2_distance(source_ndt, local_ndt, T)

#         try:
#             delta = torch.linalg.solve(hessian, -gradient)
#         except RuntimeError:
#             hessian += 1e-6 * torch.eye(6, device=device, dtype=dtype)
#             delta = torch.linalg.solve(hessian, -gradient)

#         # Convert delta to SE(3) transformation
#         delta_T = torch.eye(4, device=device, dtype=dtype)
#         delta_T[:3, :3] = expmap(delta[:3])  # Rotation from exponential map
#         delta_T[:3, 3] = delta[3:]          # Translation
#         T = delta_T @ T

#         # Convergence check
#         if torch.norm(delta) < tolerance:
#             print(f"Converged in {iteration + 1} iterations.")
#             break

#     return T



# def save_ndt_2d_bird_eye(
#     ndt_blue, ndt_red, point_cloud1, point_cloud2, save_path, xlim, ylim, alpha=0.5, scale=1, figsize=(30, 30)
# ):
#     """
#     Save a 2D bird's-eye view of two NDT maps and a point cloud as an image.

#     Args:
#         ndt_blue (dict): First NDT map to be visualized in blue.
#         ndt_red (dict): Second NDT map to be visualized in red.
#         point_cloud1 (torch.Tensor or np.ndarray): Point cloud data (Nx3 or Nx2).
#         point_cloud2 (torch.Tensor or np.ndarray): Point cloud data (Nx3 or Nx2).
#         save_path (str): Path to save the image file (e.g., 'ndt_view.png').
#         xlim (tuple): Limits for the X axis (can be torch.Tensor).
#         ylim (tuple): Limits for the Y axis (can be torch.Tensor).
#         alpha (float): Transparency of the ellipses.
#         scale (float): Scaling factor for ellipse size (default is 10 for larger ellipses).
#         figsize (tuple): Size of the output figure in inches (default is (12, 12)).
#     """
#     # Ensure xlim and ylim are numpy arrays
#     if isinstance(xlim[0], torch.Tensor):
#         xlim = (xlim[0].cpu().numpy(), xlim[1].cpu().numpy())
#     if isinstance(ylim[0], torch.Tensor):
#         ylim = (ylim[0].cpu().numpy(), ylim[1].cpu().numpy())

#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     ax.set_title("2D Bird's-Eye View of NDTs and Point Cloud")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_aspect('equal')  # Ensure equal axis resolution

#     def plot_ndt(ndt, color):
#         for mean, cov in ndt.values():
#             # Extract 2D projection (XY plane)
#             if isinstance(mean, torch.Tensor):
#                 mean = mean.cpu().numpy()
#             if isinstance(cov, torch.Tensor):
#                 cov = cov.cpu().numpy()

#             mean_2d = mean[:2]
#             cov_2d = cov[:2, :2]

#             # Eigen decomposition for ellipse parameters
#             eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
#             angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])  # Rotation angle
#             width, height = scale * np.sqrt(eigenvalues)  # Scale factor applied to ellipse size

#             # Draw ellipse
#             ellipse = plt.matplotlib.patches.Ellipse(
#                 xy=mean_2d, width=width, height=height, angle=np.degrees(angle),
#                 edgecolor=color, facecolor=color, alpha=alpha
#             )
#             ax.add_patch(ellipse)


#     # Plot Point Cloud
#     if isinstance(point_cloud1, torch.Tensor):
#         point_cloud1 = point_cloud1.cpu().numpy()
#     if point_cloud1.shape[1] == 3:  # Use only XY for 2D visualization
#         point_cloud1 = point_cloud1[:, :2]
#     ax.scatter(point_cloud1[:, 0], point_cloud1[:, 1], c='green', s=10, label='Local Point Cloud')
    
#     if isinstance(point_cloud2, torch.Tensor):
#         point_cloud2 = point_cloud2.cpu().numpy()
#     if point_cloud2.shape[1] == 3:  # Use only XY for 2D visualization
#         point_cloud2 = point_cloud2[:, :2]
#     ax.scatter(point_cloud2[:, 0], point_cloud2[:, 1], c='yellow', s=10, label='Source Point Cloud')
    
#     # Plot NDTs
#     plot_ndt(ndt_blue, color='blue')  # First NDT in blue
#     plot_ndt(ndt_red, color='red')    # Second NDT in red
    
#     # Add legend
#     ax.legend(loc='upper right')

#     # Save the plot as an image
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close(fig)


# def save_ndt_2d_side_view(
#     ndt_blue, ndt_red, point_cloud1, point_cloud2, save_path, xlim, zlim, alpha=0.5, scale=3, figsize=(30, 30)
# ):
#     """
#     Save a 2D side view (X-Z plane) of two NDT maps and a point cloud as an image.

#     Args:
#         ndt_blue (dict): First NDT map to be visualized in blue.
#         ndt_red (dict): Second NDT map to be visualized in red.
#         point_cloud1 (torch.Tensor or np.ndarray): Point cloud data (Nx3).
#         point_cloud2 (torch.Tensor or np.ndarray): Point cloud data (Nx3).
#         save_path (str): Path to save the image file (e.g., 'ndt_side_view.png').
#         xlim (tuple): Limits for the X axis (can be torch.Tensor).
#         zlim (tuple): Limits for the Z axis (can be torch.Tensor).
#         alpha (float): Transparency of the ellipses.
#         scale (float): Scaling factor for ellipse size (default is 10 for larger ellipses).
#         figsize (tuple): Size of the output figure in inches (default is (30, 30)).
#     """
#     # Ensure xlim and zlim are numpy arrays
#     if isinstance(xlim[0], torch.Tensor):
#         xlim = (xlim[0].cpu().numpy(), xlim[1].cpu().numpy())
#     if isinstance(zlim[0], torch.Tensor):
#         zlim = (zlim[0].cpu().numpy(), zlim[1].cpu().numpy())

#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_xlim(xlim)
#     ax.set_ylim(zlim)
#     ax.set_title("2D Side View (X-Z) of NDTs and Point Cloud")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Z")
#     ax.set_aspect('equal')  # Ensure equal axis resolution

#     def plot_ndt(ndt, color):
#         for mean, cov in ndt.values():
#             # Extract 2D projection (XZ plane)
#             if isinstance(mean, torch.Tensor):
#                 mean = mean.cpu().numpy()
#             if isinstance(cov, torch.Tensor):
#                 cov = cov.cpu().numpy()

#             mean_2d = mean[[0, 2]]  # Use X and Z for side view
#             cov_2d = cov[np.ix_([0, 2], [0, 2])]  # Covariance for XZ plane

#             # Eigen decomposition for ellipse parameters
#             eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
#             angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])  # Rotation angle
#             width, height = scale * np.sqrt(eigenvalues)  # Scale factor applied to ellipse size

#             # Draw ellipse
#             ellipse = plt.matplotlib.patches.Ellipse(
#                 xy=mean_2d, width=width, height=height, angle=np.degrees(angle),
#                 edgecolor=color, facecolor=color, alpha=alpha
#             )
#             ax.add_patch(ellipse)

#     # Plot Point Cloud
#     if isinstance(point_cloud1, torch.Tensor):
#         point_cloud1 = point_cloud1.cpu().numpy()
#     if point_cloud1.shape[1] == 3:  # Use XZ for side view
#         point_cloud1 = point_cloud1[:, [0, 2]]
#     ax.scatter(point_cloud1[:, 0], point_cloud1[:, 1], c='green', s=10, label='Local Point Cloud')

#     if isinstance(point_cloud2, torch.Tensor):
#         point_cloud2 = point_cloud2.cpu().numpy()
#     if point_cloud2.shape[1] == 3:  # Use XZ for side view
#         point_cloud2 = point_cloud2[:, [0, 2]]
#     ax.scatter(point_cloud2[:, 0], point_cloud2[:, 1], c='yellow', s=10, label='Source Point Cloud')

#     # Plot NDTs
#     plot_ndt(ndt_blue, color='blue')  # First NDT in blue
#     plot_ndt(ndt_red, color='red')    # Second NDT in red

#     # Add legend
#     ax.legend(loc='upper right')

#     # Save the plot as an image
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close(fig)






# def d2p_registration(source_points, ndt_map, voxel_size, max_iterations=20, tolerance=1e-5):
#     device = source_points.device
#     dtype = source_points.dtype
#     T = torch.eye(4, device=device, dtype=dtype)  # Initial transformation matrix

#     def transform_points(points, transform):
#         R = transform[:3, :3]
#         t = transform[:3, 3]
#         return points @ R.T + t

#     def compute_score_and_gradient(points):
#         score = 0
#         gradient = torch.zeros(6, device=device, dtype=dtype)
#         hessian = torch.zeros((6, 6), device=device, dtype=dtype)

#         for point in points:
#             voxel_idx = tuple(torch.floor(point / voxel_size).to(torch.int32).tolist())
#             if voxel_idx not in ndt_map:
#                 continue

#             mean, covariance = ndt_map[voxel_idx]
#             diff = point - mean
#             inv_covariance = torch.linalg.pinv(covariance)

#             # Compute Gaussian evaluation
#             exp_term = torch.exp(-0.5 * (diff @ inv_covariance @ diff))
#             score += exp_term

#             # Compute Jacobian components
#             J_r = -exp_term * inv_covariance @ skew(diff)
#             J_t = exp_term * inv_covariance

#             # Combine into a single Jacobian for this point
#             jacobian = torch.cat([J_r, J_t], dim=1)
#             gradient += jacobian.sum(dim=0)

#             # Update Hessian
#             hessian += jacobian.T @ jacobian

#         return score, gradient, hessian

#     for iteration in range(max_iterations):
#         transformed_points = transform_points(source_points, T)
#         _, gradient, hessian = compute_score_and_gradient(transformed_points)

#         # Newton update
#         try:
#             delta = torch.linalg.solve(hessian, -gradient)
#         except RuntimeError:
#             print("Singular Hessian, regularizing...")
#             hessian += 1e-6 * torch.eye(6, device=device, dtype=dtype)
#             delta = torch.linalg.solve(hessian, -gradient)

#         # Update transformation
#         delta_T = torch.eye(4, device=device, dtype=dtype)
#         delta_T[:3, :3] = expmap(delta[:3])
#         delta_T[:3, 3] = delta[3:]
#         T = delta_T @ T

#         # Check convergence
#         if torch.norm(delta) < tolerance:
#             print(f"Converged in {iteration + 1} iterations.")
#             break

#     return T





