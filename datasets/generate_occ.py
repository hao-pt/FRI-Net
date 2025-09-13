import numpy as np
import os
import cv2
from multiprocessing import Process, Queue
import time
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import copy
import os
import matplotlib.pyplot as plt
from PIL import Image

import argparse


def resize_and_pad(img, target_size, pad_value=(255,255,255), interp=Image.BICUBIC, return_transform=False):
    """
    Resizes a NumPy image while preserving aspect ratio and then pads it to the target size.

    Args:
        img (numpy.ndarray): Input image as a NumPy array (H, W, C).
        target_size (tuple): Target size as (height, width).
        pad_value (int): Value to use for padding. Default is 0.
        interp (int): Interpolation method. Default is PIL.Image.BICUBIC.
        return_transform (bool): If True, returns transformation parameters. Default is False.

    Returns:
        numpy.ndarray: Resized and padded image as a NumPy array.
        dict (optional): Transformation parameters if return_transform is True.
    """
    img = np.array(img)
    h, w = img.shape[:2]

    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize the image
    resized_img = np.array(Image.fromarray(img).resize((new_w, new_h), interp))

    # Calculate padding
    pad_h, pad_w = target_size[0] - new_h, target_size[1] - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad the image
    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value
    )
    if return_transform:
        transform_params = {
            'scale': scale,
            'pad_left': left,
            'pad_top': top,
            'original_size': (h, w),
            'target_size': target_size
        }
        return padded_img, transform_params
    return padded_img


def update_room_corners(room_corners, transform_params):
    """
    Updates room corner coordinates using transformation parameters from resize_and_pad.
    
    Args:
        room_corners (numpy.ndarray): Original corner coordinates with shape (N, 2) in format [x, y]
        transform_params (dict): Transformation parameters from resize_and_pad
    
    Returns:
        numpy.ndarray: Updated corner coordinates with shape (N, 2)
    """
    room_corners = np.array(room_corners, dtype=np.float32)
    
    # Extract transformation parameters
    scale = transform_params['scale']
    pad_left = transform_params['pad_left']
    pad_top = transform_params['pad_top']
    
    # Scale the coordinates
    scaled_corners = room_corners * scale
    
    # Apply padding offset (room_corners are in [x, y] format)
    updated_corners = scaled_corners + np.array([pad_left, pad_top], dtype=np.float32)
    
    return updated_corners


def read_list(image_set, dataset="stru3d"):
    data_list = []
    file_path = f'{data_path}/{dataset}/{image_set}_list.txt'
    with open(file_path, 'r', encoding='utf-8') as infile:
        for name in infile:
            data_name = name.strip('\n').split()[0]
            data_list.append(data_name)
    return data_list


def generate_occ(dataset="stru3d"):
    save_folder = f"{data_path}/occ" 
    os.makedirs(save_folder, exist_ok=True)
    # if dataset == "stru3d":
    #     label_files = [f"{data_path}/stru3d/annotations/train.json", 
    #                 f"{data_path}/stru3d/annotations/val.json", 
    #                 f"{data_path}/stru3d/annotations/test.json"]
    # elif dataset == "r2g":
    #     label_files = [f"{data_path}/R2G_hr_dataset_processed_v1/annotations/train.json",
    #         f"{data_path}/R2G_hr_dataset_processed_v1/annotations/val.json",
    #         f"{data_path}/R2G_hr_dataset_processed_v1/annotations/test.json",
    #     ]

    image_roots = [
        f"{data_path}/train/",
        f"{data_path}/val/",
        f"{data_path}/test/",
    ]
    label_files = [
        f"{data_path}/annotations/train.json",
        f"{data_path}/annotations/val.json",
        f"{data_path}/annotations/test.json",
    ]

    for image_root, ann_file in zip(image_roots, label_files):
        coco = COCO(ann_file)
        ids = list(sorted(coco.imgs.keys()))
        for id in ids:
            ann_ids = coco.getAnnIds(imgIds=id)
            target = coco.loadAnns(ann_ids)
            target = [t for t in target if t['category_id'] not in [16, 17]]
            file_name = coco.loadImgs(id)[0]['file_name'].split('.')[0]
            image_path = os.path.join(image_root, file_name) + ".png"
            img = np.array(Image.open(image_path))
            _, transform_params = resize_and_pad(img, (256, 256), return_transform=True)
            
            occ_data = []
            for room_id, each_room in enumerate(target):
                room_seg = each_room['segmentation'][0]
                room_corners = np.array(room_seg).reshape(-1, 2).astype(np.float32)
                if len(room_corners) < 3: # skip window and door
                    continue
                room_corners = update_room_corners(room_corners, transform_params)
                spatial_query = np.mgrid[:256, :256]
                spatial_query = np.moveaxis(spatial_query, 0, -1)
                spatial_query = spatial_query.reshape(-1, 2).astype(np.float32)

                mask = np.zeros((256, 256))
                cv2.fillPoly(mask, [room_corners.astype(np.int32)], 1.)

                spatial_indices = np.round(spatial_query).astype(np.int64)
                spatial_occ = mask[spatial_indices[:, 1], spatial_indices[:, 0]]

                spatial_query = spatial_query.reshape(256, 256, 2)
                spatial_occ = spatial_occ.reshape(256, 256, 1)
                sub_row_indices, sub_col_indices = np.meshgrid(np.arange(1, 256, 4), np.arange(1, 256, 4),
                                                               indexing='ij')
                query = spatial_query[sub_row_indices, sub_col_indices]
                query = query.reshape(64 * 64, 2)
                occ = spatial_occ[sub_row_indices, sub_col_indices]
                occ = occ.reshape(64 * 64, 1)

                room_occ = dict()
                room_occ['query'] = query
                room_occ['occ'] = occ
                occ_data.append(room_occ)
            save_path = f'{save_folder}/{file_name}.npy'
            np.save(save_path, occ_data)
            print(f'generate occ: {save_path}')


def generate_input_img(dataset="stru3d"):
    if dataset == "stru3d":
        density_folder = f"{data_path}/stru3d/density"
        height_folder = f"{data_path}/stru3d/height"
    
    file_list = os.listdir(density_folder)
    file_list = sorted(file_list)
    
    save_folder = f"{data_path}/{dataset}/input"

    os.makedirs(save_folder, exist_ok=True)

    for file in file_list:
        print(f'file_name: {file}')
        density_path = f'{density_folder}/{file}'
        height_path = f'{height_folder}/{file}'
        if not os.path.exists(density_path) or not os.path.exists(height_path):
            continue
        density = cv2.imread(density_path)
        height = cv2.imread(height_path)
        # image_name = file.split('.')[0]
        # cv2.imshow('density', density)
        # cv2.imshow('height', height)
        # cv2.waitKey(0)
        print(f'generate: {file}')
        input_img = np.maximum(density, height)
        cv2.imwrite(f'{save_folder}/{file}', input_img)


if __name__ == '__main__':
    # generate_input_img(dataset="stru3d")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/d/projects/FRI-Net/FRI-Net/data", help='path to data folder')
    args = parser.parse_args()
    data_path = args.data_path
    generate_occ(dataset="stru3d")