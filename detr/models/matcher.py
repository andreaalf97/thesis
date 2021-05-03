# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

CLASSES = {
    0: "<start>",
    1: "<point>",
    2: "<end-of-polygon>",
    3: "<end-of-computation>"
}


@torch.no_grad()
def get_polygons(logits_full: torch.Tensor, boxes: torch.Tensor) -> list:

    logits = logits_full.argmax(-1)

    i = 1
    polygons = []
    points = []
    while i < len(logits) and logits[i] != 3:  # Keep going until <end-of-computation> token
        if logits[i] == 1:  # If the prediction is a point, we add it to the point list of this polygon
            points.append(boxes[i])
        elif logits[i] in [2, 0]:  # <end-of-polygon> or <start>
            if len(points) > 0:
                polygons.append(torch.stack(points))
            points = []
        i += 1
    return polygons


@torch.no_grad()
def get_sequence(polygons: list, target: dict):
    """
        This function takes the list of prediction polygons and the list of ground truth polygons of the same batch
        and outputs the optimal target sequence after matching them.
        What we want to do is find a the optimal match between the first point of each prediction and the points of each polygon
    """

    num_tgt_polygons = target['boxes'].shape[0]

    # First we stack the FIRST point of the prediction polygons in a single tensor that has
    # final shape [num_pred_polygons, 2]
    poly = torch.stack([p[0] for p in polygons])

    # We also reshape the target polygons to have x, y in a single dimension.
    # The final shape of the tgt tensor is [num_tgt_polygons, num_points_pre_polygon(with padding), 2]
    tgt = target['boxes'].view(num_tgt_polygons, -1, 2)

    # What we w

    print('poly', poly.shape)
    print('poly', poly)
    print('tgt', tgt.shape)
    print('tgt', tgt)

    # dist has shape [num_tgt_polygons, 7 (points per polygon with padding), num_pred_polygons]
    dist = torch.cdist(tgt, poly, p=1)

    print("dist", dist.shape)
    print("dist", dist)

    dist, _ = torch.min(dist, dim=1)

    print("dist", dist.shape)
    print("dist", dist)

    indices = linear_sum_assignment(dist.cpu())

    print(indices)



    exit(0)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                 "aux_outputs" IS REMOVED

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # First, I extract the predicted polygons from the output sequence
        polygons = [get_polygons(log, box) for log, box in zip(outputs["pred_logits"], outputs["pred_boxes"])]

        # With the prediction polygons, we can match them to the tgt polygons and create our target sequence
        matched_sequences = [get_sequence(polygon, target) for polygon, target in zip(polygons, targets)]

        exit(0)

        print('outputs["pred_logits"]', outputs["pred_logits"].shape)
        print('outputs["pred_boxes"]', outputs["pred_boxes"].shape)
        print('[v["labels"] for v in targets]', [v["labels"] for v in targets])
        print('[v["boxes"] for v in targets]', [v["boxes"] for v in targets])
        exit(0)

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 8]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class # + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
