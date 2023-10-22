'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-04-07 15:12:40
Email: haimingzhang@link.cuhk.edu.cn
Description: Reder the ViPC dataset and obtaining the camera parameters.
'''

import time
import os
import os.path as osp
import sys
import contextlib
from math import radians
# from PIL import Image
import random
import numpy as np
import bpy
from mathutils import Matrix

DIR_RENDERING_PATH = './data/render_2'
RENDERING_MAX_CAMERA_DIST = 1.75
RENDERING_BLENDER_TMP_DIR = '/tmp/blender'
N_VIEWS = 24

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


class BaseRenderer:
    model_idx   = 0

    def __init__(self):
        # bpy.data.scenes['Scene'].render.engine = 'CYCLES'
        # bpy.context.scene.cycles.device = 'GPU'
        # bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        # bpy.context.user_preferences.system.compute_device = 'CUDA_1'

        # changing these values does affect the render.

        # remove the default cube
        bpy.ops.object.select_pattern(pattern="Cube")
        bpy.ops.object.delete()

        render_context = bpy.context.scene.render
        world  = bpy.context.scene.world
        camera = bpy.data.objects['Camera']
        light_1  = bpy.data.objects['Lamp']
        light_1.data.type = 'HEMI'

        # set the camera postion and orientation so that it is in
        # the front of the object
        camera.location       = (1, 0, 0)
        camera.rotation_mode  = 'ZXY'
        camera.rotation_euler = (0, radians(90), radians(90))

        # parent camera with a empty object at origin
        org_obj                = bpy.data.objects.new("RotCenter", None)
        org_obj.location       = (0, 0, 0)
        org_obj.rotation_euler = (0, 0, 0)
        bpy.context.scene.objects.link(org_obj)

        camera.parent = org_obj  # setup parenting

        # render setting
        render_context.resolution_percentage = 100
        world.horizon_color = (1, 1, 1)  # set background color to be white

        # set file name for storing rendering result
        self.result_fn = '%s/render_result_%d.png' % (DIR_RENDERING_PATH, os.getpid())
        bpy.context.scene.render.filepath = self.result_fn

        # new settings
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.use_zbuffer = True
        
        self.render_context = render_context
        self.org_obj = org_obj
        self.camera = camera
        self.light = light_1
        self._set_lighting()

    def initialize(self, viewport_size_x=256, viewport_size_y=256, models_fn=None):
        self.models_fn = models_fn
        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y

    def _set_lighting(self):
        pass

    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov):
        self.org_obj.rotation_euler = (0, 0, 0)
        self.light.location = (distance_ratio *
                               (RENDERING_MAX_CAMERA_DIST + 2), 0, 0)
        self.camera.location = (distance_ratio *
                                RENDERING_MAX_CAMERA_DIST, 0, 0)
        self.org_obj.rotation_euler = (radians(-yaw),
                                       radians(-altitude),
                                       radians(-azimuth))

    def setTransparency(self, transparency):
        """ transparency is either 'SKY', 'TRANSPARENT'
        If set 'SKY', render background using sky color."""
        self.render_context.alpha_mode = transparency

    def selectModel(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_pattern(pattern="RotCenter")
        bpy.ops.object.select_pattern(pattern="Lamp*")
        bpy.ops.object.select_pattern(pattern="Camera")
        bpy.ops.object.select_all(action='INVERT')

    def printSelection(self):
        print(bpy.context.selected_objects)

    def clearModel(self):
        self.selectModel()
        bpy.ops.object.delete()

        # The meshes still present after delete
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        for item in bpy.data.materials:
            bpy.data.materials.remove(item)

    def setModelIndex(self, model_idx):
        self.model_idx = model_idx

    def loadModel(self, file_path=None):
        if file_path is None:
            file_path = self.models_fn[self.model_idx]

        if file_path.endswith('obj'):
            bpy.ops.import_scene.obj(filepath=file_path)
        elif file_path.endswith('3ds'):
            bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
        elif file_path.endswith('dae'):
            # Must install OpenCollada. Please read README.md
            bpy.ops.wm.collada_import(filepath=file_path)
        else:
            raise Exception("Loading failed: %s Model loading for type %s not Implemented" %
                            (file_path, file_path[-4:]))

    def render(self, load_model=True, clear_model=True, resize_ratio=None,
               return_image=True, image_path=os.path.join(RENDERING_BLENDER_TMP_DIR, 'tmp.png')):
        """ Render the object """
        if load_model:
            self.loadModel()

        # resize object
        self.selectModel()
        if resize_ratio:
            bpy.ops.transform.resize(value=resize_ratio)

        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # save straight to file

        if resize_ratio:
            bpy.ops.transform.resize(value=(1/resize_ratio[0],
                1/resize_ratio[1], 1/resize_ratio[2]))

        if clear_model:
            self.clearModel()


class ShapeNetRenderer(BaseRenderer):

    def __init__(self):
        super().__init__()
        self.setTransparency('TRANSPARENT')

    def _set_lighting(self):
        # Create new lamp datablock
        light_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')

        # Create new object with our lamp datablock
        light_2 = bpy.data.objects.new(name="New Lamp", object_data=light_data)
        bpy.context.scene.objects.link(light_2)

        # put the light behind the camera. Reduce specular lighting
        self.light.location       = (0, -2, 2)
        self.light.rotation_mode  = 'ZXY'
        self.light.rotation_euler = (radians(45), 0, radians(90))
        self.light.data.energy = 0.7

        light_2.location       = (0, 2, 2)
        light_2.rotation_mode  = 'ZXY'
        light_2.rotation_euler = (-radians(45), 0, radians(90))
        light_2.data.energy = 0.7
    
    def get_3x4_P_matrix_from_blender(self):
        cam = self.camera
        K = get_calibration_matrix_K_from_blender(cam.data)
        RT = get_3x4_RT_matrix_from_blender(cam)
        return K, RT


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_blender2shapenet = Matrix(
        ((1, 0, 0),
         (0, 0, -1),
         (0, 1, 0)))

    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam*R_blender2shapenet
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT



def add_cam_params(args=None):
    renderer = ShapeNetRenderer()
    # file_paths = ["/data1/zhanghm/Datasets/ShapeNet/ShapeNetCore.v1/02691156/1a04e3eab45ca15dd86060f189eb133/model.obj"]
    file_paths = ["/data1/zhanghm/Datasets/ShapeNet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj"]

    renderer.initialize(137, 137, file_paths)

    view_root = "/home/zhanghm/Research/Completion/XMFnet/dataset/ShapeNetViPC-Dataset/ShapeNetViPC-View"
    cate_dir_list = sorted([cate_dir for cate_dir in os.listdir(view_root)
                            if os.path.isdir(osp.join(view_root, cate_dir))])
    print(len(cate_dir_list), cate_dir_list[:3])
    for cate_dir in cate_dir_list:
        model_dir_list = sorted(os.listdir(os.path.join(view_root, cate_dir)))
        for model_dir in model_dir_list:
            print(model_dir)
            if model_dir != "1a04e3eab45ca15dd86060f189eb133":
                continue
            model_dir_abs = osp.join(view_root, cate_dir, model_dir)
            rendering_dir = osp.join(view_root, cate_dir, model_dir, "rendering")
            rendering_meta_file = osp.join(rendering_dir, "rendering_metadata.txt")
            with open(rendering_meta_file, "r") as f:
                metadata_lines = f.readlines()
            
            cam_pose_dir = osp.join(view_root, cate_dir, model_dir, "pose_v2")
            cam_intrinsic_dir = osp.join(view_root, cate_dir, model_dir, "intrinsic_v2")
            os.makedirs(cam_pose_dir, exist_ok=True)
            os.makedirs(cam_intrinsic_dir, exist_ok=True)

            for i in range(24):
                # Get camera calibration.
                azim, elev, yaw, dist_ratio, fov = [
                    float(v) for v in metadata_lines[i].strip().split(" ")
                ]

                if i == 0:
                    load_model_flag = True
                else:
                    load_model_flag = False

                if i == N_VIEWS - 1:
                    clear_model_flag = True
                else:
                    clear_model_flag = False

                renderer.setModelIndex(0)
                renderer.setViewpoint(azim, elev, 0, dist_ratio, 25)
                image_path = os.path.join(model_dir_abs, 'rendering_v2', '%.2d.png' % i)
                renderer.render(load_model=load_model_flag, return_image=False,
                                clear_model=clear_model_flag, image_path=image_path)

                K, RT = renderer.get_3x4_P_matrix_from_blender()
                cam_K_file = os.path.join(cam_intrinsic_dir, '%.2d.txt' % i)
                np.savetxt(cam_K_file, K)
                cam_RT_path = os.path.join(cam_pose_dir, '%.2d.txt' % i)
                np.savetxt(cam_RT_path, RT)
            break
        break


def load_data_path(shapenet_path):
    target_obj_file = 'model_normalized.obj'
    data_path = []
    for root, dirs, files in os.walk(shapenet_path, topdown=True):
        if target_obj_file in files:
            obj_path = os.path.join(root, target_obj_file)
            data_path.append(obj_path)
    return data_path


def create_srn_like_dataset():
    renderer = ShapeNetRenderer()

    # load the models
    shapenet_normalized_path = "./data/ShapeNetCore.v2_normalized"
    all_objects = load_data_path(shapenet_normalized_path)
    # only process the airplane objects in the ShapeNetViPC dataset
    shapenet_vipc_dir = "/home/zhanghm/Research/PointCloud/3D-R2N2/data/ShapeNetViPC-Dataset/ShapeNetViPC-GT/02691156"
    all_airplanes = os.listdir(shapenet_vipc_dir)
    exit(0)

    # file_paths = ["/data1/zhanghm/Datasets/ShapeNet/ShapeNetCore.v1/02691156/1a04e3eab45ca15dd86060f189eb133/model.obj"]
    file_paths = ["/data1/zhanghm/Datasets/ShapeNet/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj"]

    renderer.initialize(137, 137, file_paths)

    view_root = "/home/zhanghm/Research/Completion/XMFnet/dataset/ShapeNetViPC-Dataset/ShapeNetViPC-View"
    cate_dir_list = sorted([cate_dir for cate_dir in os.listdir(view_root)
                            if os.path.isdir(osp.join(view_root, cate_dir))])
    print(len(cate_dir_list), cate_dir_list[:3])
    for cate_dir in cate_dir_list:
        model_dir_list = sorted(os.listdir(os.path.join(view_root, cate_dir)))
        for model_dir in model_dir_list:
            print(model_dir)
            if model_dir != "1a04e3eab45ca15dd86060f189eb133":
                continue
            model_dir_abs = osp.join(view_root, cate_dir, model_dir)
            rendering_dir = osp.join(view_root, cate_dir, model_dir, "rendering")
            rendering_meta_file = osp.join(rendering_dir, "rendering_metadata.txt")
            with open(rendering_meta_file, "r") as f:
                metadata_lines = f.readlines()
            
            cam_pose_dir = osp.join(view_root, cate_dir, model_dir, "pose_v2")
            cam_intrinsic_dir = osp.join(view_root, cate_dir, model_dir, "intrinsic_v2")
            os.makedirs(cam_pose_dir, exist_ok=True)
            os.makedirs(cam_intrinsic_dir, exist_ok=True)

            for i in range(24):
                # Get camera calibration.
                azim, elev, yaw, dist_ratio, fov = [
                    float(v) for v in metadata_lines[i].strip().split(" ")
                ]

                if i == 0:
                    load_model_flag = True
                else:
                    load_model_flag = False

                if i == N_VIEWS - 1:
                    clear_model_flag = True
                else:
                    clear_model_flag = False

                renderer.setModelIndex(0)
                renderer.setViewpoint(azim, elev, 0, dist_ratio, 25)
                image_path = os.path.join(model_dir_abs, 'rendering_v2', '%.2d.png' % i)
                renderer.render(load_model=load_model_flag, return_image=False,
                                clear_model=clear_model_flag, image_path=image_path)

                K, RT = renderer.get_3x4_P_matrix_from_blender()
                cam_K_file = os.path.join(cam_intrinsic_dir, '%.2d.txt' % i)
                np.savetxt(cam_K_file, K)
                cam_RT_path = os.path.join(cam_pose_dir, '%.2d.txt' % i)
                np.savetxt(cam_RT_path, RT)
            break
        break

if __name__ == "__main__":
    create_srn_like_dataset()