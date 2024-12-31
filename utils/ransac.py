#!/usr/bin/env python3
# @file      ransac.py
# @author    Sangwoo Jung    [dan0130@snu.ac.kr]
# Copyright (c) 2024 Sangwoo Jung, all rights reserved

import numpy as np
import cv2
import random
import sys

class Ransac:
    def __init__(self, iterations=30, threshold=0.2, sample_size=3):
        # RANSAC parameters
        self.iterations = iterations
        self.threshold = threshold
        self.sample_size = sample_size        
        
        # Results
        self.dynamic_points = None  # To store dynamic points
        self.static_points = None   # To store static points
        self.ego_vel = None         # To store ego velocity from static points
        
    def process_pointcloud(self, raw_pointcloud: np.ndarray):
        """
        Process 4D pointcloud data to separate static and dynamic objects.
        raw_pointcloud: numpy.ndarray of shape (n, 5), where each row is (x, y, z, doppler, rcs)
        """
        if raw_pointcloud.shape[1] != 5:
            sys.exit("Input pointcloud must have shape (n, 5).")
        
        # print("Number of points before filtering:", len(raw_pointcloud))
        
        raw_pointcloud = self.filter_pointcloud_by_azimuth(raw_pointcloud)
        
        # print("Number of points after nearest_filtering:", len(raw_pointcloud))
        
        xyz = raw_pointcloud[:, :3]
        vr = raw_pointcloud[:, 3]
        
        dist = np.linalg.norm(xyz, axis=1).reshape(-1, 1)
        A = xyz / dist
        B = vr.reshape(-1, 1)
        
        max_cnt = 0
        best_model = None
        
        for _ in range(self.iterations):
            # Random sampling
            indices = random.sample(range(len(vr)), self.sample_size)
            AA = A[indices, :]
            BB = B[indices, :]
            
            try:
                X = np.linalg.lstsq(AA, BB, rcond=None)[0]
                residuals = np.abs(B - A @ X)
                cnt = np.sum(residuals < self.threshold)
                
                if cnt > max_cnt:
                    best_model = X
                    max_cnt = cnt
            except np.linalg.LinAlgError:
                continue
        
        if best_model is not None:
            residuals = np.abs(A @ best_model - B).flatten()
            inliers = residuals < self.threshold
            outliers = ~inliers
            
            self.static_points = raw_pointcloud[inliers]
            self.dynamic_points = raw_pointcloud[outliers]
            self.ego_vel = self.cal_ego_vel()
        
            # print("Number of static points after RANSAC:", len(self.static_points))
            # print("Number of static points before filtering:", len(self.static_points))
            
            # test_static = self.filter_pointcloud_by_azimuth(self.static_points)
            # self.static_points = test_static
            
        # print("shape of static: ", self.static_points.shape)
        # print("shape of dynamic:", self.dynamic_points.shape)
        
        # print(self.ego_vel)
        # self.draw_result()
    
    def draw_result(self):
        """
        Visualize the results.
        """
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        interval = 200

        if self.static_points is not None:
            for point in self.static_points:
                try:
                    # Calculate x and y coordinates
                    pt_x = int((np.arctan(point[1] / point[0]) + np.pi / 2) * interval)
                    temp_vel = point[3] * np.sqrt(point[0]**2 + point[1]**2) / np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
                    pt_y = int((temp_vel + 10) * 30.0)
                    cv2.circle(img, (pt_x, pt_y), 3, (0, 255, 0), -1)  # Green circles
                except ZeroDivisionError:
                    # Handle division by zero (point[0] == 0)
                    continue

        # Draw dynamic points in red
        if self.dynamic_points is not None:
            for point in self.dynamic_points:
                try:
                    # Calculate x and y coordinates
                    pt_x = int((np.arctan(point[1] / point[0]) + np.pi / 2) * interval)
                    temp_vel = point[3] * np.sqrt(point[0]**2 + point[1]**2) / np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
                    pt_y = int((temp_vel + 10) * 30.0)
                    cv2.circle(img, (pt_x, pt_y), 3, (0, 0, 255), -1)  # Red circles
                except ZeroDivisionError:
                    # Handle division by zero (point[0] == 0)
                    continue

        cv2.imshow("Result Ransac", img)
        cv2.waitKey(1)
    
    def cal_ego_vel(self):
        A = self.static_points[:, :3]
        B = - self.static_points[:, 3]
        B = B.reshape(-1, 1)
        
        norm = np.linalg.norm(A, axis=1, keepdims=True)
        A_normalized = A / norm
        
        A_pseudo_inverse = np.linalg.pinv(A_normalized)
        v = A_pseudo_inverse @ B

        return v.flatten()
    
    def clean(self):
        self.dynamic_points = None
        self.static_points = None
        self.ego_vel = None

    def cart2spherical(self, x, y, z):
        # Radius
        r = np.sqrt(x**2 + y**2 + z**2)
        # Azimuth
        az = np.degrees(np.arctan2(y, x))
        az[az < 0] += 360
        # Elevation
        el = np.degrees(np.arcsin(z / r))
        return r, az, el
    
    # def filter_pointcloud_by_azimuth(self, pointcloud, az_res_deg=0.175):
    #     """
    #     pointcloud shape: (N, 4)
    #     Each row is [x, y, z, vr]

    #     Filter by azimuth bins only, without separate binning for elevation.
    #     In other words, create a key from (az_bin, el). If there are multiple points 
    #     with the same az_bin and the same el value, keep only the one that is closest.
    #     If el is different (which is almost always the case), keep them all.
    #     """

    #     xyz = pointcloud[:, :3]
    #     x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    #     r, az, el = self.cart2spherical(x, y, z)
        
    #     az_bin = (az // az_res_deg).astype(int)

    #     bin_dict = {}
    #     for i in range(len(pointcloud)):
    #         key = (az_bin[i], el[i])  # Create a key from az_bin and el
    #         dist = r[i]
    #         if key not in bin_dict:
    #             bin_dict[key] = (dist, i)
    #         else:
    #             # If there is already a point with the same az_bin, el value,
    #             # keep the one that is closer
    #             if dist < bin_dict[key][0]:
    #                 bin_dict[key] = (dist, i)

    #     selected_indices = [val[1] for val in bin_dict.values()]
    #     filtered_pointcloud = pointcloud[selected_indices]

    #     return filtered_pointcloud
    
    def filter_pointcloud_by_azimuth(self, pointcloud, az_res_deg=0.175):
        """
        pointcloud shape: (N, 5)
        Each row is [x, y, z, doppler, rcs]

        Filter by azimuth bins only, keeping only one point per bin if they have the same elevation.
        If elevations differ, all are kept.

        Uses NumPy sorting and unique for performance instead of a dictionary.
        """
        xyz = pointcloud[:, :3]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r, az, el = self.cart2spherical(x, y, z)
        
        az_bin = (az // az_res_deg).astype(int)
        
        # Sort by (az_bin, el, r) and keep only the closest point in each (az_bin, el) group
        data = np.column_stack((az_bin, el, r, np.arange(len(pointcloud))))
        
        # Sort data by az_bin -> el -> r
        data_sorted = data[np.lexsort((data[:,2], data[:,1], data[:,0]))]
        
        # The first occurrence of each (az_bin, el) after sorting is the point with minimum r
        _, unique_indices = np.unique(data_sorted[:, :2], axis=0, return_index=True)
        
        selected = data_sorted[unique_indices]
        original_indices = selected[:, 3].astype(int)

        filtered_pointcloud = pointcloud[original_indices]

        return filtered_pointcloud
    
    def rad_vel_uncertainty(self, method=0, alpha=1.0, beta=1.0, epsilon=1e-6):
        """
        Calculate radial velocity from ego velocity and compare it with the estimated radial velocity (Doppler).
        Compute a similarity score for each point.

        Parameters
        ----------
        method : int
            0 -> Linear + Clipping
            1 -> Exponential (Gaussian-like)
            2 -> Logistic (Sigmoid-like)
        alpha : float
            Sensitivity parameter for method=1,2
            (Controls how quickly the similarity drops to 0 when there is a large difference in velocity)
        beta : float
            The midpoint parameter for the logistic (sigmoid) function when method=2
            (If difference exceeds beta, the similarity drops rapidly)
        epsilon : float
            A small value to prevent division by zero
        """

        # Check if ego_vel and static_points are well defined
        if self.ego_vel is None or self.static_points is None:
            raise ValueError("Ego velocity or static points are not calculated yet.")
        
        # Extract positions and Doppler velocities
        positions = self.static_points[:, :3]
        doppler_vel = self.static_points[:, 3]
        
        # Normalize positions to calculate direction vectors
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        directions = - positions / norms  # Direction vectors for each point
        
        # Calculate radial velocities using ego velocity
        ego_vel_radial = np.sum(directions * self.ego_vel, axis=1)
        
        # Method-specific similarity calculation
        if method == 0:
            # (1) Linear + Clipping: similarity = 1 - (|ego - doppler| / (|doppler| + epsilon))
            diff = np.abs(ego_vel_radial - doppler_vel)
            denom = np.abs(doppler_vel) + epsilon
            similarity = 1 - (diff / denom)
            similarity = np.clip(similarity, 0.0, 1.0)
        
        elif method == 1:
            # (2) Exponential (Gaussian-like): similarity = exp(-alpha * |ego - doppler|)
            diff = np.abs(ego_vel_radial - doppler_vel)
            similarity = np.exp(-alpha * diff)
            # similarity = np.clip(similarity, 0.0, 1.0)  # optional
        
        elif method == 2:
            # (3) Logistic (sigmoid): similarity = 1 / (1 + exp(alpha * (|ego - doppler| - beta)))
            diff = np.abs(ego_vel_radial - doppler_vel)
            similarity = 1.0 / (1.0 + np.exp(alpha * (diff - beta)))
            # similarity = np.clip(similarity, 0.0, 1.0)  # optional
        
        else:
            raise ValueError("method must be 0, 1, or 2.")
        
        # print(f"Similarity score (method={method}):", similarity)
        return similarity
    
    def vel_cos_similarity(self):
        """
        Calculate cosine similarity between ego velocity and the direction of radial velocity
        for each static point.
        """
        if self.ego_vel is None:
            return ValueError("Ego velocity or static points are not calculated yet.")
        
        # Extract positions
        positions = self.static_points[:, :3]  # (x, y, z)

        # Normalize positions to calculate direction vectors
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        
        zero_mask = (norms < 1e-6)
        norms[zero_mask] = 1e-6
        
        directions = positions / norms  # Shape: (n, 3)
        
        ego_vel_norm_val = np.linalg.norm(self.ego_vel)
        if ego_vel_norm_val < 1e-6:
            return np.zeros(len(positions))        

        # Normalize ego velocity
        ego_vel_norm = self.ego_vel / ego_vel_norm_val

        # Calculate cosine similarity for each point
        cosine_similarities = np.sum(directions * ego_vel_norm, axis=1)
        
        # print("Cosine similarities:", cosine_similarities)
        # print("min cosine similarity:", np.min(cosine_similarities)

        return cosine_similarities
