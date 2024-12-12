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
        raw_pointcloud: numpy.ndarray of shape (n, 4), where each row is (x, y, z, vr)
        """
        if raw_pointcloud.shape[1] != 4:
            sys.exit("Input pointcloud must have shape (n, 4).")
                
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
        else:
            self.clean()
        
        print(self.ego_vel)
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
