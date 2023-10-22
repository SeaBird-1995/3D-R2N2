'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-04-07 14:41:05
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import numpy as np
import os
import os.path as osp
from matplotlib import pyplot as plt
from einops import (rearrange, reduce, repeat)
import cv2
import numpy as np


idx = 8
RT_fp = f"data/render_2/02691156/1a04e3eab45ca15dd86060f189eb133/cam_RT/{idx:02d}.txt"
K_fp = "data/render_2/02691156/1a04e3eab45ca15dd86060f189eb133/cam_K/00.txt"
pc_fp = "data/1a04e3eab45ca15dd86060f189eb133.xyz"
img_path = f"data/render_2/02691156/1a04e3eab45ca15dd86060f189eb133/rendering/{idx:02d}.png"

data_str = "data/render_2/02958343/15d6cd7877c6601d9ed410a010efa019"
RT_fp = f"{data_str}/pose/{idx:02d}.txt"
K_fp = f"{data_str}/intrinsic.txt"
pc_fp = "data/15d6cd7877c6601d9ed410a010efa019.xyz"
img_path = f"{data_str}/rendering/{idx:02d}.png"


idx = 21
RT_fp = f"./data/ShapeNetViPC-Dataset/ShapeNetViPC-View/02691156/1a04e3eab45ca15dd86060f189eb133/pose/{idx:02d}.txt"
K_fp = "./data/ShapeNetViPC-Dataset/ShapeNetViPC-View/02691156/1a04e3eab45ca15dd86060f189eb133/intrinsic/00.txt"
pc_fp = "data/1a04e3eab45ca15dd86060f189eb133.xyz"
img_path = f"./data/ShapeNetViPC-Dataset/ShapeNetViPC-View/02691156/1a04e3eab45ca15dd86060f189eb133/rendering/{idx:02d}.png"


cam_RT = np.loadtxt(RT_fp)
cam_K = np.loadtxt(K_fp)
pc_world = np.loadtxt(pc_fp) # (N, 3)


def draw_points(image, points, add_text=False):
    """Draw points on the image.

    Args:
        image (np.ndarray): input image
        points (np.ndarray): (N, 2) array of points
        add_text (bool, optional): _description_. Defaults to True.

    """
    idx = -1
    for pt in points:
        idx += 1
        pt = (int(pt[0]), int(pt[1]))
        cv2.circle(image, pt, 2, (255, 0, 0), -1)

        if add_text:
            cv2.putText(image, str(idx), pt, cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255))


def transform(pc, cam_RT, cam_K, method=1):
    """Project the points from the world coordinate to the image pixels

    Args:
        pc (np.ndarray): (N, 3)
        cam_RT (np.ndarray): (3, 4), the world space to camera space transform matrix
        cam_K (_type_): (3, 3)
        method (int, optional): _description_. Defaults to 1.

    Returns:
        (np.ndarray): (N, 2), the (u, v) in the image coordinate, u
    """
    pc_cam = pc @ cam_RT[:, :3].T + cam_RT[:, -1]
    
    if method == 1:
        x = pc_cam[:, 0]
        y = pc_cam[:, 1]
        z = pc_cam[:, 2]
        u = x * cam_K[0][0] / z + cam_K[0][2]
        v = y * cam_K[1][1] / z + cam_K[1][2]

        pc_pixel = np.vstack([u, v]).T
    elif method == 2:
        cam_K_new = np.eye(4)
        cam_K_new[:3, :3] = cam_K
        pc_cam_h = np.concatenate([pc_cam, np.ones_like(pc_cam[..., :1])], axis=-1)

        pc_pixel = cam_K_new @ pc_cam_h.T
        pc_pixel = pc_pixel.T
        pc_pixel = pc_pixel[:, :2] / pc_pixel[:, 2:3]
    elif method == 3:
        w2c_pose_new = np.eye(4)
        w2c_pose_new[:3, :4] = cam_RT

        cam_K_new = np.eye(4)
        cam_K_new[:3, :3] = cam_K

        P_matrix = cam_K_new @ w2c_pose_new  # (4, 4)
        pc_h = np.concatenate([pc, np.ones_like(pc_cam[..., :1])], axis=-1)  # to (N, 4)
        
        pc_pixel = P_matrix @ rearrange(pc_h, "N Dim4 -> Dim4 N")  # to (4, N)
        pc_pixel = pc_pixel / pc_pixel[2:3]
        pc_pixel = pc_pixel[:2]
        pc_pixel = rearrange(pc_pixel, "Dim2 N -> N Dim2")

    pc_cam = pc @ cam_RT[:, :3].T + cam_RT[:, -1]
    np.savetxt(f"pc_cam_{idx:03d}.xyz", pc_cam)
    return pc_pixel


pc_pixel = transform(pc_world, cam_RT, cam_K, method=3)

img_src = cv2.imread(img_path)
print(img_src.shape, type(img_src))
# pc_pixel = pc_pixel[:, ::-1]
# pc_pixel[:, -1] = 512 - 1 - pc_pixel[:, -1]
# draw_points(img_src, pc_pixel)
for p in pc_pixel:
    img_src[int(p[1]), int(p[0]), :] = (255, 0, 0)
cv2.imwrite(f"opencv_neww_{idx:03d}.jpg", img_src)


