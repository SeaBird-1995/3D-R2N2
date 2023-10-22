'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-04-27 20:37:54
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os
import os.path as osp
from tqdm import tqdm
import pickle
import multiprocessing
import subprocess

g_blender_excutable_path = '/data1/zhanghm/Softwares/blender-2.79b-linux-glibc219-x86_64/blender'
g_shapenet_vipc_dir = "data/ShapeNetViPC-Dataset"
g_shapenet_vipc_view_dir = osp.join(g_shapenet_vipc_dir, "ShapeNetViPC-View")


def load_objects(mode="train", synset_str="02691156"):
    """We load the needed object category according to the splits of ShapeNet-ViPC
    dataset

    Args:
        mode (str, optional): _description_. Defaults to "train".
        synset_str (str, optional): _description_. Defaults to "02691156".

    Returns:
        _type_: _description_
    """
    vipc_split_file = osp.join(g_shapenet_vipc_dir, f"vipc_{mode}.txt")
    with open(vipc_split_file) as f:
        lines = [line.rstrip() for line in f.readlines()]
    lines = list(filter(lambda x: x.split(" ")[0] == synset_str, lines))
    return lines


def load_viewpoints(input_objects):
    all_objects_with_vp = []
    for key in tqdm(input_objects):
        synset, obj = key.split("/")[-4:-2]
        rendering_meta_file = osp.join(g_shapenet_vipc_view_dir, synset, obj, "rendering/rendering_metadata.txt")
        with open(rendering_meta_file, "r") as f:
            metadata_lines = f.readlines()
        all_viewpoints = []
        for line in metadata_lines:
            # Get camera calibration.
            azim, elev, yaw, dist_ratio, fov = [
                float(v) for v in line.strip().split(" ")
            ]
            all_viewpoints.append((azim, elev, yaw, dist_ratio, fov))
        all_objects_with_vp.append((key, all_viewpoints))
    return all_objects_with_vp


def create_srn_like_dataset(mode="train", synset_str="02691156"):
    # load the models
    shapenet_normalized_path = "./data/ShapeNetCore.v2_normalized"
    vipc_objects = load_objects(mode, synset_str)
    all_objects = [osp.join(shapenet_normalized_path, k.split()[0], k.split()[1], "models/model_normalized.obj") 
                   for k in vipc_objects]
    
    print(len(all_objects), all_objects[:3])
    all_objects_with_vp = load_viewpoints(all_objects)
    return all_objects_with_vp


def group_by_cpu(result_list, count):
    '''
    deploy model-view result to different kernels
    :param result_list: model-view result
    :param count: kernel count
    :return:
    '''
    num_per_batch = int(len(result_list)/count)
    result_list_by_group = []
    for batch_id in range(count):
        if batch_id != count-1:
            result_list_by_group.append(result_list[batch_id*num_per_batch: (batch_id+1)*num_per_batch])
        else:
            result_list_by_group.append(result_list[batch_id * num_per_batch:])
    return result_list_by_group


def render_cmd(result_dict):
    #render rgb
    command = [g_blender_excutable_path, '--background', '--python', 'render_srn.py', '--', result_dict]
    subprocess.run(command)


if __name__ == "__main__":
    result_list = create_srn_like_dataset(mode="test")
    
    count = 4

    result_list_by_group = group_by_cpu(result_list, count)
    model_view_dir = "model_view_metadata_test"
    os.makedirs(model_view_dir, exist_ok=True)

    model_view_paths = []
    for batch_id, result_per_group in zip(range(count), result_list_by_group):
        file_name = str(batch_id) + '_' + "airplane.pkl"
        single_model_view_path = os.path.join(model_view_dir, file_name)
        model_view_paths.append(single_model_view_path)

        with open(single_model_view_path, 'wb') as f:
            pickle.dump(result_per_group, f)
    
    # pool = multiprocessing.Pool(processes=count)
    # pool.map(render_cmd, model_view_paths)