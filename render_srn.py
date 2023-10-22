'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-04-27 21:17:27
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import sys
import os
import os.path as osp
sys.path.append(os.path.dirname(__file__))

import pickle
import numpy as np
import bpy
from mathutils import Matrix
import argparse
import blender_interface


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Blender renderer.')
    parser.add_argument("--dict", type=str, 
                        default="/home/zhanghm/Research/PointCloud/3D-R2N2/model_view_metadata_test/3_airplane.pkl",
                        help="model-view file for rendering.")
    args = parser.parse_args(argv)
    return args


def clear_mesh():
    """ clear all meshes in the secene

    """
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH' or obj.type == 'EMPTY':
            obj.select = True
    bpy.ops.object.delete()


if __name__ == "__main__":
    print("Start rendering srn dataset...")

    args = parse_args()

    result_list = pickle.load(open(args.dict, 'rb'))
    print(len(result_list), result_list[:3])

    renderer = blender_interface.ShapeNetRenderer()
    renderer.initialize(128, 128)

    output_dir = "./data/srn_airplanes/airplanes_test"
    os.makedirs(output_dir, exist_ok=True)

    for model in result_list:
        model_path = model[0]
        viewpoints = model[1]

        cat = model_path.split('/')[4]
        instance_dir = os.path.join(output_dir, cat)
        rgb_dir = osp.join(instance_dir, "rgb")
        pose_dir = osp.join(instance_dir, "pose")

        os.makedirs(instance_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)

        ## Start rendering this model
        renderer.clearModel()
        clear_mesh()
        renderer.loadModel(model_path)
        # rendering this model with different viewpoints
        for idx, vp in enumerate(viewpoints):
            image_path = os.path.join(rgb_dir, '%06d.png' % idx)
            if osp.exists(image_path):
                continue
            
            azim, elev, yaw, dist_ratio, fov = vp
            dist_ratio = 1.0 ## we keep the camera distance all the same
            renderer.setViewpoint(azim, elev, 0, dist_ratio, 25)
            
            renderer.render(image_path=image_path)

            # write the camera info
            K, RT = renderer.get_3x4_P_matrix_from_blender()
            cam_K_file = os.path.join(instance_dir, 'intrinsics.txt')
            np.savetxt(cam_K_file, K)
            cam_RT_path = os.path.join(pose_dir, '%06d.txt' % idx)
            np.savetxt(cam_RT_path, RT)