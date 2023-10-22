'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-04-27 20:33:13
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import os


def load_data_path(shapenet_path):
    target_obj_file = 'model_normalized.obj'
    data_path = []
    for root, dirs, files in os.walk(shapenet_path, topdown=True):
        if target_obj_file in files:
            obj_path = os.path.join(root, target_obj_file)
            data_path.append(obj_path)
    return data_path