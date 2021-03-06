import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


def show_pred(img, mask):
    plt.show(img)
    plt.imshow(mask.squeeze(0), alpha=0.5, cmap='jet', interpolation='none')
    plt.show()


def collate(data):
    return tuple(zip(*data))


def iou(mask1, mask2) -> float:
    intersection = torch.logical_and(mask1, mask2).sum()
    if intersection == 0.0:
        return 0.0
    union = torch.logical_or(mask1, mask2).sum()
    return (float(intersection)/union).item()


def match_masks_optim(gt_masks, pred_masks):
    """
    This function matches each ground truth mask with a single mask in the predictions
    """
    cost_matrix = [[0 for _ in range(len(pred_masks))] for _ in range(len(gt_masks))]
    for i, gt_mask in enumerate(gt_masks):  # For each GT mask we find the best match
        for j, pred_mask in enumerate(pred_masks):  # For each prediction mask we compare it
            iou_score = iou(
                pred_mask,
                gt_mask
            )
            cost_matrix[i][j] = iou_score

    cost_matrix = np.array(cost_matrix)
    match = linear_sum_assignment(cost_matrix, maximize=True)

    return [[i, j, cost_matrix[i][j]] for i, j in zip(match[0], match[1])]


@torch.no_grad()
def evaluate(model, pkl_path, pretrained_model, ds_func):

    ds_path = "/home/andreaalf/Documents/thesis/datasets/gate_samples"
    save_results_to = "/home/andreaalf/Documents/thesis/detr/results/baseline_comparison/EVAL_maskrcnn_uniform8000_100epochs.pkl"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()

    pred_confidence_threshold = 0.5
    pixel_confidence_threshold = 0.5
    iou_threshold = 0.8

    ds = ds_func(
        ds_path,
        pkl_path=pkl_path,
        image_set='val'
    )

    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=collate)

    results = pd.DataFrame({
        'img_id': [],
        'gt_id': [],
        'pred_id': [],
        'confidence': [],
        'outcome': []
    })
    total_gt = 0

    for iteration, (images, targets) in enumerate(data_loader):
        images = list(i.to(device) for i in images)
        targets = [{k: v.to(device) for k, v in dictionary.items()} for dictionary in targets]

        output = model(images)
        for output_dict, img, target in zip(output, images, targets):  # For each image in the dataset

            # print('\n'.join(str(k) + ' -- ' + str(output_dict[k].shape) for k in output_dict))
            # print('\n')
            # print('\n'.join(str(k) + ' -- ' + str(target[k].shape) for k in target))

            total_gt += len(target['labels'])

            '''Scores is a list of tuples as long as the objects in the ground truth: (gt_index, pred_index, iou_score)'''
            scores = match_masks_optim(
                gt_masks=target['masks'],
                pred_masks=torch.where(output_dict['masks'] > 0.5, 1, 0)
            )
            img_id = int(target['image_id'].item())

            row = {
                'img_id': [],
                'gt_id': [],
                'pred_id': [],
                'confidence': [],
                'outcome': []
            }

            pred_indices = list(range(len(output_dict['labels'])))
            for gt_index, pred_index, iou_score in scores:
                row['img_id'].append(img_id)
                row['gt_id'].append(int(gt_index))
                row['pred_id'].append(int(pred_index))
                row['confidence'].append(output_dict['scores'][pred_index].item())
                row['outcome'].append(
                    'TP' if iou_score > iou_threshold else 'FP'
                )
                pred_indices.remove(int(pred_index))
            for unmatched_pred_index in pred_indices:
                row['img_id'].append(img_id)
                row['gt_id'].append(-1)
                row['pred_id'].append(unmatched_pred_index)
                row['confidence'].append(output_dict['scores'][unmatched_pred_index].item())
                row['outcome'].append('FN')

            results = results.append(pd.DataFrame(row), ignore_index=True)

        print(f"Iteration {iteration} of {len(data_loader)}")
        print(f"Dataframe size {len(results)}\n#####")
        # if iteration % int(len(data_loader)/10) == 0:
        #     print(f"Iteration {iteration} of {len(data_loader)}")
    print("SAVING RESULTS TO", save_results_to)
    results.to_pickle(save_results_to)