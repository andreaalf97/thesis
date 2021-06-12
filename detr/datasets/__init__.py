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
        return TSDataset(256, 256, num_gates=args.num_gates, black_and_white=True, no_gate_chance=0.0, stroke=-1, num_corners=4, mask=False, clamp_gates=True, seq_order=args.seq_order, fix_gates=args.fix_gates)
    if args.dataset_file == 'real_gates':
        return RealGatesDS(args.real_gate_path, pkl_path=args.real_gate_pickle_path, image_set=image_set, seq_order=args.seq_order)
    raise ValueError(f'dataset {args.dataset_file} not supported')


def get_mask_rcnn_dataset(path: str, pkl_path: str, image_set='train'):
    return RealGatesDS(dataset_path=path, pkl_path=pkl_path, image_set=image_set, mask_rcnn=True)


def get_toy_setting_dataset():
    return TSDataset(256, 256, num_gates=5, black_and_white=True, no_gate_chance=0.0, stroke=-1, num_corners=4, mask=True)