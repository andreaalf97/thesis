# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
from scipy.optimize import linear_sum_assignment

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from shapely.geometry import MultiPoint
from shapely.errors import TopologicalError
import pandas as pd
from os.path import join


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        # coco_sample = torch.load("tmp/coco_sample_from_dataloader.pt")
        # samples, targets = torch.load("tmp/samples.pth"), torch.load("tmp/targets.pth")
        # show_coco_sample(samples, targets, s_num=0)

        """
        Each sample returned from the data loader is a tuple of size 2
            SAMPLES: is a NestedTensor (from util.misc)
                COCO samples have values in range (-2.101, 2.163)
                toy_setting samples have values in range (0, 1)
            TARGETS: is a tuple of length BATCH_SIZE
                Each label is a dict with keys:
                    boxes --> [num_gates, 8]
                        For COCO, boxes is [num_gates, 4], where each box is [center_x, center_y, width, height]
                    labels --> [num_gates]
                    image_id --> [1]
                    area --> [num_gates]
                    iscrowd --> [num_gates]
                    orig_size --> [2]
                    size --> [2]
                        This is Height, Width
        """
        samples = samples.to(device)
        # All values in the target dict are tensors so we move them to the selected device for computation (cuda/cpu)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Image values in range 0.0 : 1.0

        """
        OUTPUTS is a DICT with keys:
            pred_logits --> Tensor of shape [batch_size, 100, 21]
            pred_boxes --> Tensor of shape [batch_size, 100, 21]
            aux_outputs --> List of length 5 (repetitions of decoder - 1)
                Each aux_output[i] is again a DICT with keys ['pred_logits', 'pred_boxes']
        """
        outputs = model(samples)

        """
        LOSS_DICT is a dict with keys (all tensors of dim 1 or single items)
            loss_ce class_error loss_bbox loss_giou cardinality_error
            loss_ce_0 loss_bbox_0 loss_giou_0 cardinality_error_0
            loss_ce_1 loss_bbox_1 loss_giou_1 cardinality_error_1
            loss_ce_2 loss_bbox_2 loss_giou_2 cardinality_error_2
            loss_ce_3 loss_bbox_3 loss_giou_3 cardinality_error_3
            loss_ce_4 loss_bbox_4 loss_giou_4 cardinality_error_4
        """
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def plot_image_with_labels(samples: torch.Tensor, targets: torch.Tensor, num_sample=0, threshold=True):

    img = samples.tensors[num_sample]
    coords = targets[num_sample]["boxes"]

    h, w = list(img.shape)[-2:]

    print("IMAGE H:{}\nIMAGE W:{}".format(h, w))

    colors = [
        "blue",
        "red",
        "orange",
        "green",
        "yellow"
    ]

    plt.imshow(img.cpu().permute(1, 2, 0))

    for gate_coord, color in zip(coords, colors):
        if threshold:
            for i in range(len(gate_coord)):
                if gate_coord[i] >= 1.0:
                    gate_coord[i] = torch.tensor(0.99)
                if gate_coord[i] <= 0.0:
                    gate_coord[i] = torch.tensor(0.99)
        bl_x, bl_y = gate_coord[0], gate_coord[1]
        tl_x, tl_y = gate_coord[2], gate_coord[3]
        tr_x, tr_y = gate_coord[4], gate_coord[5]
        br_x, br_y = gate_coord[6], gate_coord[7]

        plt.scatter(bl_x.cpu() * w, bl_y.cpu() * h, c=color)
        plt.scatter(tl_x.cpu() * w, tl_y.cpu() * h, c=color)
        plt.scatter(tr_x.cpu() * w, tr_y.cpu() * h, c=color)
        plt.scatter(br_x.cpu() * w, br_y.cpu() * h, c=color)

    plt.show()

    print("COORDS SHAPE", coords.shape)


@torch.no_grad()
def plot_prediction(samples: utils.NestedTensor, outputs: dict, targets: tuple):

    logits_batch = outputs["pred_logits"]  # [batch_size, 100, 21]
    coords_batch = outputs["pred_boxes"]  # [batch_size, 100, 8]

    colors = [
        "tab:blue",
        "tab:red",
        "tab:orange",
        "tab:green",
        "tab:brown",
        "tab:purple",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
        "lime",
        "navy",
        "lightgray",
        "gold",
        "chocolate",
        "palegreen"
    ]

    colors = colors * 5

    for image, logits, coords, target in zip(samples.tensors, logits_batch, coords_batch, targets):

        num_predictions = 0

        h, w = list(image.shape)[-2:]
        plt.imshow(image.cpu().permute(1, 2, 0))

        for logit, coord, color in zip(logits, coords, colors):

            logit = torch.softmax(logit, 0)
            confidence, index = torch.max(logit, 0)

            if index.item() == 0:
                num_predictions += 1
                for i in range(len(coord)):
                    if coord[i] >= 1.0:
                        coord[i] = torch.tensor(0.99)
                    if coord[i] <= 0.0:
                        coord[i] = torch.tensor(0.99)
                bl_x, bl_y = coord[0], coord[1]
                tl_x, tl_y = coord[2], coord[3]
                tr_x, tr_y = coord[4], coord[5]
                br_x, br_y = coord[6], coord[7]

                plt.scatter(bl_x.cpu() * w, bl_y.cpu() * h, c=color)
                plt.scatter(tl_x.cpu() * w, tl_y.cpu() * h, c=color)
                plt.scatter(tr_x.cpu() * w, tr_y.cpu() * h, c=color)
                plt.scatter(br_x.cpu() * w, br_y.cpu() * h, c=color)

                plt.text(tl_x*w, tl_y*h, str(confidence.item()*100)[:5]+"%", color=color)

        plt.title(f"FOUND {num_predictions} GATES WITH A TOTAL OF {len(target['labels'])}")
        plt.show()


@torch.no_grad()
def plot_single_sample(sample: torch.Tensor, prediction_box: torch.Tensor, gt_box: torch.Tensor, title="", s=16) -> None:

    plt.imshow(sample.cpu().permute(1, 2, 0))
    h = list(sample.shape)[1]
    w = list(sample.shape)[2]

    x_s = [
        prediction_box[0] * w,
        prediction_box[2] * w,
        prediction_box[4] * w,
        prediction_box[6] * w
    ]

    y_s = [
        prediction_box[1] * h,
        prediction_box[3] * h,
        prediction_box[5] * h,
        prediction_box[7] * h
    ]

    x_s = [x.item() for x in x_s]
    y_s = [x.item() for x in y_s]

    plt.scatter(x_s, y_s, c='red', label="Prediction", s=s)

    x_s = [
        gt_box[0] * w,
        gt_box[2] * w,
        gt_box[4] * w,
        gt_box[6] * w
    ]

    y_s = [
        gt_box[1] * h,
        gt_box[3] * h,
        gt_box[5] * h,
        gt_box[7] * h
    ]

    x_s = [x.item() for x in x_s]
    y_s = [x.item() for x in y_s]

    plt.scatter(x_s, y_s, c='blue', label="Grund truth", s=s)

    plt.title(title)
    plt.legend()
    plt.show()


def show_coco_sample(samples, targets, s_num=0):
    img = samples.tensors[s_num]
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    boxes = targets[s_num]["boxes"]
    img_h, img_w = targets[s_num]["size"].cpu()
    labels = targets[s_num]["labels"]
    plt.imshow(img.cpu().permute(1, 2, 0))

    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    colors = [
        "blue",
        "red",
        "green",
        "yellow",
        "orange",
        "white"
    ]

    for box, color, label in zip(boxes, colors, labels):
        w, h = box[2].cpu() * img_w, box[3].cpu() * img_h

        plt.scatter(box[0].cpu() * img_w - (w / 2), box[1].cpu() * img_h - (h / 2), c=color, marker='+')
        plt.scatter(box[0].cpu() * img_w - (w / 2), box[1].cpu() * img_h + (h / 2), c=color, marker='+')
        plt.scatter(box[0].cpu() * img_w + (w / 2), box[1].cpu() * img_h + (h / 2), c=color, marker='+')
        plt.scatter(box[0].cpu() * img_w + (w / 2), box[1].cpu() * img_h - (h / 2), c=color, marker='+')
        plt.text(box[0].cpu() * img_w, box[1].cpu() * img_h, CLASSES[label], c=color)

    plt.title(f"Original shape of {list(img.shape)}")
    plt.show()


def print_confusion_matrix(matrix: dict):
    assert 'T' in matrix and 'F' in matrix
    assert 'T' in matrix['T'] and 'F' in matrix['T']
    assert 'T' in matrix['F'] and 'F' in matrix['F']

    TP = matrix['T']['T']
    TN = matrix['T']['F']
    FP = matrix['F']['T']
    FN = matrix['F']['F']

    print("__CONF___MATRIX__")
    print("x\tT\tF")
    print(f"T\t{TP}\t{FP}")
    print(f"F\t{FN}\t{TN}")
    print("-----------------")

    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("PRECISION: %.8f" % precision)
    print("RECALL: %.8f" % recall)
    print("F1 SCORE: %.8f" % f1)


def valid_box(box: torch.Tensor) -> bool:
    b = box.tolist()

    valid_bl = 0 < b[0] < 1 and 0 < b[1] < 1
    if valid_bl:
        return True
    valid_tl = 0 < b[2] < 1 and 0 < b[3] < 1
    if valid_tl:
        return True
    valid_tr = 0 < b[4] < 1 and 0 < b[5] < 1
    if valid_tr:
        return True
    valid_br = 0 < b[6] < 1 and 0 < b[7] < 1
    if valid_br:
        return True

    return False


def iou(prediction: torch.Tensor, gt: torch.Tensor) -> float:

    pred_bl = [prediction[0].item(), prediction[1].item()]
    pred_tl = [prediction[2].item(), prediction[3].item()]
    pred_tr = [prediction[4].item(), prediction[5].item()]
    pred_br = [prediction[6].item(), prediction[7].item()]

    gt_bl = [gt[0].item(), gt[1].item()]
    gt_tl = [gt[2].item(), gt[3].item()]
    gt_tr = [gt[4].item(), gt[5].item()]
    gt_br = [gt[6].item(), gt[7].item()]

    try:
        pred_poly = MultiPoint([
            pred_bl,
            pred_tl,
            pred_tr,
            pred_br
        ]).convex_hull

        gt_poly = MultiPoint([
            gt_bl,
            gt_tl,
            gt_tr,
            gt_br
        ]).convex_hull

        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.area + gt_poly.area - intersection
        # union = pred_poly.union(gt_poly).area

        return intersection/union
    except TopologicalError:
        return -1


@torch.no_grad()
def corner_distances(pred_box: torch.Tensor, real_box: torch.Tensor) -> tuple:

    bl = math.sqrt((pred_box[0] - real_box[0])**2 + (pred_box[1] - real_box[1])**2)
    tl = math.sqrt((pred_box[2] - real_box[2])**2 + (pred_box[3] - real_box[3])**2)
    tr = math.sqrt((pred_box[4] - real_box[4])**2 + (pred_box[5] - real_box[5])**2)
    br = math.sqrt((pred_box[6] - real_box[6])**2 + (pred_box[7] - real_box[7])**2)

    return bl, tl, tr, br


@torch.no_grad()
def plot_loss(output_file: str):
    with open(output_file, 'r') as file:

        epoch_list = []
        class_err_list = []
        loss_list = []
        loss_ce_list = []
        loss_bbox_list = []

        line = file.readline()
        while line:
            if "Epoch:" in line and "Total time: " not in line:
                epoch = int(line.split("Epoch: [", 1)[1].split("]", 1)[0])
                epoch_list.append(epoch)

                class_error = float(line.split("class_error: ", 1)[1].split(" loss:", 1)[0])
                class_err_list.append(class_error)

                loss = float(line.split("loss: ", 1)[1].split(" (", 1)[0])
                loss_list.append(loss)

                loss_ce = float(line.split("loss_ce: ", 1)[1].split(" (", 1)[0])
                loss_ce_list.append(loss_ce)

                loss_bbox = float(line.split("loss_bbox: ", 1)[1].split(" (", 1)[0])
                loss_bbox_list.append(loss_bbox)

            line = file.readline()

        epoch_list = np.array(epoch_list)
        class_err_list = np.array(class_err_list)
        loss_list = np.array(loss_list)
        loss_ce_list = np.array(loss_ce_list)
        loss_bbox_list = np.array(loss_bbox_list)

        window_size = int(math.floor(len(loss_bbox_list) / 100))
        print(window_size)
        if window_size % 2 == 0:
            window_size += 1

        loss_bbox_list = savgol_filter(loss_bbox_list, window_size, 3)

        plt.plot(range(len(loss_bbox_list)), loss_bbox_list, label="Avg of last 200 values is 0.33999")
        # plt.plot(range(len(loss_list_b), len(loss_list_b) + len(loss_list_a)), loss_list_a, label="Loss after LR drop")

        plt.title("Coordinates Loss")
        plt.legend()
        plt.show()

        return


def match_predictions_optim(pred_logits: torch.Tensor, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> tuple:
    """
        This function matches each ground truth mask with a single mask in the predictions
    """
    predictions = []
    for index, logits in enumerate(pred_logits):
        confidence, pred_class = torch.max(torch.softmax(logits, dim=0), 0)
        if pred_class == 0:
            predictions.append((
                pred_boxes[index], confidence.item(), index
            ))

    cost_matrix = [[0 for _ in range(len(predictions))] for _ in range(len(gt_boxes))]
    for i, gt_box in enumerate(gt_boxes):  # For each GT mask we find the best match
        for j, (pred_box, _, _) in enumerate(predictions):  # For each prediction mask we compare it
            iou_score = iou(
                pred_box,
                gt_box
            )
            cost_matrix[i][j] = iou_score

    cost_matrix = np.array(cost_matrix)
    match = linear_sum_assignment(cost_matrix, maximize=True)

    false_positives = []
    for index, (p, conf, real_index) in enumerate(predictions):
        if index not in match[1]:  # Prediction not assigned to any GT polygon
            false_positives.append((conf, real_index))

    return [[i, predictions[j][2], cost_matrix[i][j], predictions[j][1]] for i, j in zip(match[0], match[1])], false_positives


@torch.no_grad()
def evaluate_map(model, data_loader_val, device, args):
    assert args.pretrained_model != '', "Give path to pretrained model with --pretrained_model"

    print("######################")
    print("EVALUATION")

    if args.pretrained_model != "":
        state_dict = torch.load(args.pretrained_model)
    model.load_state_dict(state_dict)
    model.eval()

    results = pd.DataFrame({
        'img_id': [],
        'gt_id': [],
        'pred_id': [],
        'confidence': [],
        'outcome': [],
        'gate': []
    })
    image_objects = pd.DataFrame({
        'img_id': [],
        'num_objects': [],
        'gates': []
    })

    for iteration, (images, targets) in enumerate(data_loader_val):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        outputs = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # plot_prediction(images, outputs, targets)
        # continue

        for pred_logits, pred_boxes, target in zip(outputs['pred_logits'], outputs['pred_boxes'], targets):  # For each image in the dataset

            # print("pred_logits", pred_logits.shape)
            # print("pred_boxes", pred_boxes.shape)
            # print("target:\n" + '\n'.join('   ' + str(k) + ' -> ' + str(target[k].shape) for k in target))

            '''Scores is a list of tuples as long as the objects in the ground truth: (gt_index, pred_index, iou_score)'''
            scores, false_positives = match_predictions_optim(
                pred_logits=pred_logits,
                pred_boxes=pred_boxes,
                gt_boxes=target['boxes']
            )
            img_id = int(target['image_id'].item())

            row = {
                'img_id': [],
                'gt_id': [],
                'pred_id': [],
                'confidence': [],
                'outcome': [],
                'gate': []
            }

            for gt_index, pred_index, iou_score, confidence in scores:
                row['img_id'].append(img_id)
                row['gt_id'].append(int(gt_index))
                row['pred_id'].append(int(pred_index))
                row['confidence'].append(confidence)
                row['outcome'].append(
                    iou_score
                )
                row['gate'].append(pred_boxes[pred_index].tolist())
            for conf, i in false_positives:
                row['img_id'].append(img_id)
                row['gt_id'].append(-1)
                row['pred_id'].append(i)
                row['confidence'].append(conf)
                row['outcome'].append(0.0)
                row['gate'].append(pred_boxes[i].tolist())

            results = results.append(pd.DataFrame(row), ignore_index=True)
            image_objects = image_objects.append(pd.DataFrame({
                'img_id': [img_id],
                'num_objects': [len(target['boxes'])],
                'gates': [target['boxes'].cpu().tolist()]
            }), ignore_index=True)

        print(f"Iteration {iteration} of {len(data_loader_val)}")
        print(f"Dataframe size {len(results)}\n#####")

    folder_path = args.pretrained_model.replace(args.pretrained_model.split('/')[-1], '')
    model_name = args.pretrained_model.split('/')[-1].replace('.pth', '')

    res_path = join(
        folder_path,
        "EVAL_" + model_name + ".pkl"
    )
    ds_info_path = join(
        folder_path,
        "DS_INFO_" + model_name + '.pkl'
    )

    print("SAVING RESULTS TO" + res_path)
    results.to_pickle(res_path)
    # image_objects.to_pickle(ds_info_path)


@torch.no_grad()
def evaluate_toy_setting(model, data_loader_val, criterion, device, args):
    assert args.pretrained_model != '', "Give path to pretrained model with --pretrained_model"

    print("######################")
    print("EVALUATION")

    if "checkpoint" in args.pretrained_model:
        state_dict = torch.load(args.pretrained_model)["model"]
    else:
        state_dict = torch.load(args.pretrained_model)
    model.load_state_dict(state_dict)
    model.eval()
    criterion.eval()

    confusion_matrix = {
        'T': {
            'T': 0,
            'F': 0
        },
        'F': {
            'T': 0,
            'F': 0
        }
    }

    iou_sum = 0
    num_loss_checks = 0
    invalid_gates = 0
    wrong_coord_gates = 0
    num_predictions = 0

    iou_threshold = float(args.iou_treshold)

    len_data = len(data_loader_val)

    for iteration, (samples, targets) in enumerate(data_loader_val):  # For one epoch

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        outputs = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        plot_prediction(samples, outputs, targets)
        exit(0)
        indices = criterion.get_indices(outputs, targets)

        for pred_logits, pred_boxes, target, idx, sample in zip(outputs["pred_logits"], outputs["pred_boxes"], targets, indices, samples.tensors):

            real_boxes = target['boxes']
            idx = (idx[0].tolist(), idx[1].tolist())

            for i, (pred_logit, pred_box) in enumerate(zip(pred_logits, pred_boxes)):
                num_predictions += 1
                _, pred_class = torch.max(pred_logit, 0)
                # confidence, pred_class = torch.max(torch.softmax(pred_logit, dim=0), 0)

                if i in idx[0]:  # Matched with a gate
                    real_box = real_boxes[idx[1][idx[0].index(i)]]
                    if not valid_box(real_box):
                        invalid_gates += 1
                        continue

                    real_box = torch.clamp(real_box, 0, 1)

                    if pred_class == 0:

                        iou_score = iou(pred_box, real_box)

                        # If the IOU score is smaller than the threshold, it's a wrong prediction
                        if iou_score < iou_threshold:
                            if iou_score == -1 or max(corner_distances(pred_box, real_box)) > 0.02:
                                wrong_coord_gates += 1
                                # plot_single_sample(sample, pred_box, real_box, title=f"False positive (iou %.2f and distance %.4f)" % (iou_score, max(corner_distances(pred_box, real_box))))
                                confusion_matrix['F']['T'] += 1
                                continue
                        iou_sum += iou_score
                        # plot_single_sample(sample, pred_box, real_box, title=("True positive (IoU %.8f)" % iou_score))
                        confusion_matrix['T']['T'] += 1
                        num_loss_checks += 1
                    else:
                        # plot_single_sample(sample, pred_box, real_box, title="False negative")
                        confusion_matrix['F']['F'] += 1
                else:  # Not matched with a gate
                    if pred_class == 0:
                        # plot_single_sample(sample, pred_box, real_box, title=("False positive (matching) with confidence %.4f" % confidence))
                        confusion_matrix['F']['T'] += 1
                    else:
                        # plot_single_sample(sample, pred_box, real_box, title="True negative")
                        confusion_matrix['T']['F'] += 1

        if iteration % 20 == 0:
            print(f"[EVAL] Iteration {iteration} of {len_data}")

        # plot_prediction(samples, outputs, targets)

    print_confusion_matrix(confusion_matrix)
    print(f"OUT OF {num_predictions} PREDICTIONS")
    print("AVERAGE IOU for correct predictions: %.8f" % (float(iou_sum)/num_loss_checks))
    print("INVALID GATES:", invalid_gates)
    print("WRONG COORD GATES (these are also false positives)", wrong_coord_gates)

    print("COMPLETED EVALUATION")
    print("######################")

    # test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                       data_loader_val, base_ds, device, args.output_dir)
    # if args.output_dir:
    #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
