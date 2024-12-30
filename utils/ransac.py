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
        # Ransac Parameters
        self.iterations = iterations
        self.threshold = threshold
        self.sample_size = sample_size        
        
        # Results
        self.dynamic_points = None  # To store dynamic points
        self.static_points = None  # To store static points
        self.ego_vel = None # To store ego-vel from static points
        
    def process_pointcloud(self, raw_pointcloud: np.ndarray):
        """
        Process 4D pointcloud data to separate static and dynamic object. 
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
        # radius
        r = np.sqrt(x**2 + y**2 + z**2)
        # azimuth
        az = np.degrees(np.arctan2(y, x))
        az[az < 0] += 360
        # elevation
        el = np.degrees(np.arcsin(z / r))
        return r, az, el
    
    # def filter_pointcloud_by_azimuth(self, pointcloud, az_res_deg=0.175):
    #     """
    #     pointcloud shape: (N, 4)
    #     각 row는 [x, y, z, vr]

    #     azimuth bin 단위로만 필터링하되, elevation에 대해서는 별도 binning 없이 그대로 사용.
    #     즉, (az_bin, el)의 조합으로 key를 만들고, 같은 az_bin과 동일한 el 값을 가지는 점이 있다면
    #     그 중 가장 가까운 점 하나만 유지. el이 다르면(거의 항상 다를 것) 모두 유지.
    #     """

    #     xyz = pointcloud[:, :3]
    #     x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    #     r, az, el = self.cart2spherical(x, y, z)
        
    #     az_bin = (az // az_res_deg).astype(int)

    #     bin_dict = {}
    #     for i in range(len(pointcloud)):
    #         key = (az_bin[i], el[i])  # az_bin과 el값의 조합으로 key를 만든다
    #         dist = r[i]
    #         if key not in bin_dict:
    #             bin_dict[key] = (dist, i)
    #         else:
    #             # 같은 az_bin, 같은 el 값일 경우 더 가까운 점 선택
    #             # (이 경우 거의 동일한 el을 가진 포인트는 드물지만 혹시 몰라서 처리)
    #             if dist < bin_dict[key][0]:
    #                 bin_dict[key] = (dist, i)

    #     selected_indices = [val[1] for val in bin_dict.values()]
    #     filtered_pointcloud = pointcloud[selected_indices]

    #     return filtered_pointcloud
    
    def filter_pointcloud_by_azimuth(self, pointcloud, az_res_deg=0.175):
        """
        pointcloud shape: (N, 5)
        각 row는 [x, y, z, doppler, rcs]

        azimuth bin 단위로만 필터링하되,
        elevation 값이 동일한 경우에만 그 중 가장 가까운 점을 하나 선택.
        elevation 값이 다르면 모두 유지한다.

        성능 개선을 위해 dictionary 대신 NumPy 정렬 및 unique를 사용한다.
        """
        xyz = pointcloud[:, :3]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r, az, el = self.cart2spherical(x, y, z)
        
        az_bin = (az // az_res_deg).astype(int)
        
        # (az_bin, el) 그룹을 기준으로 정렬한 뒤, 같은 그룹 내에서 r이 최소인 포인트를 선택
        data = np.column_stack((az_bin, el, r, np.arange(len(pointcloud))))
        
        # data를 az_bin, el, r 순으로 정렬
        # az_bin 오름차순 -> 같은 az_bin 내 el 오름차순 -> 같은 (az_bin, el) 내 r 오름차순
        data_sorted = data[np.lexsort((data[:,2], data[:,1], data[:,0]))]
        
        # (az_bin, el) 조합에서 처음 등장하는 인덱스가 r이 최소인 point
        _, unique_indices = np.unique(data_sorted[:, :2], axis=0, return_index=True)
        
        # unique_indices가 가리키는 행만 선택
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
            Method=1,2 에서 민감도를 조절하는 파라미터
            (지수/로지스틱 함수에서 차이가 커질 때 0으로 떨어지는 속도 조절)
        beta : float
            Method=2 에서 로지스틱 함수의 중심점(차이가 beta 이상이 되면 유사도 급감)
        epsilon : float
            분모가 0이 되는 것을 방지하는 소량 값
        """

        # 먼저 ego_vel, static_points가 잘 세팅되어 있는지 확인
        if self.ego_vel is None or self.static_points is None:
            raise ValueError("Ego velocity or static points are not calculated yet.")
        
        # Extract positions and doppler velocities
        positions = self.static_points[:, :3]
        doppler_vel = self.static_points[:, 3]
        
        # Normalize positions to calculate direction vectors
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        directions = - positions / norms  # 각 point의 방향벡터
        
        # Calculate radial velocities using ego velocity
        ego_vel_radial = np.sum(directions * self.ego_vel, axis=1)
        
        # print("Ego radial velocities:", ego_vel_radial)
        # print("Doppler velocities:", doppler_vel)
        
        # --- method 별 similarity 계산 로직 ---
        if method == 0:
            # (1) 선형 + 클리핑 : similarity = 1 - (|ego - doppler| / (|doppler| + epsilon))
            diff = np.abs(ego_vel_radial - doppler_vel)
            denom = np.abs(doppler_vel) + epsilon
            similarity = 1 - (diff / denom)
            # 0~1 범위로 clip
            similarity = np.clip(similarity, 0.0, 1.0)
        
        elif method == 1:
            # (2) 지수(가우시안) 방식 : similarity = exp(-alpha * |ego - doppler|)
            diff = np.abs(ego_vel_radial - doppler_vel)
            similarity = np.exp(-alpha * diff)
            # 지수함수 결과는 이미 0~1 범위이나, 안전차원에서 clip 가능
            # similarity = np.clip(similarity, 0.0, 1.0)
        
        elif method == 2:
            # (3) 로지스틱(시그모이드) 방식
            # similarity = 1 / (1 + exp(alpha * (|ego - doppler| - beta)))
            diff = np.abs(ego_vel_radial - doppler_vel)
            similarity = 1.0 / (1.0 + np.exp(alpha * (diff - beta)))
            # 이미 0~1 범위이지만, 필요시 clip 가능
            # similarity = np.clip(similarity, 0.0, 1.0)
        
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
        # print("min cosine similarity:", np.min(cosine_similarities))

        return cosine_similarities