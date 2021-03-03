# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .toy_setting import TSDataset
from .real_gates import RealGatesDS


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'toy_setting':
        if int(args.num_gates) == -1:
            return TSDataset(256, 256, black_and_white=(not args.colored))
        else:
            return TSDataset(256, 256, num_gates=int(args.num_gates),
                             rand_gate_number=True, black_and_white=(not args.colored))
    if args.dataset_file == 'real_gates':
        return RealGatesDS(args.real_gate_path, pkl_path=args.real_gate_pickle_path, image_set=image_set)
    raise ValueError(f'dataset {args.dataset_file} not supported')


def get_mask_rcnn_dataset(path: str, pkl_path: str):
    return RealGatesDS(dataset_path=path, pkl_path=pkl_path, image_set='train', mask_rcnn=True)
