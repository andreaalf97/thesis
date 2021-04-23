# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


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

        """
            For RNN based DETR:
            OUTPUTS:
                pred_logits --> [2, 10, 2]
                pred_boxes --> [2, 10, 8, 3]
        """

        batch_indices = []
        for tgt, pred_logits, pred_boxes in zip(targets, outputs["pred_logits"], outputs["pred_boxes"]):
            tgt_boxes = tgt['boxes']
            tgt_labels = tgt['labels']

            # print("#####################################")
            # print("CREATING COST MATRIX FOR:")
            # print("pred_logits", pred_logits.shape)
            # print("pred_boxes", pred_boxes.shape)
            # print("tgt_boxes", tgt_boxes.shape)
            # print("tgt_labels", tgt_labels.shape)
            # print("#####################################")


            cost_matrix = torch.zeros(len(pred_logits), len(tgt_labels))

            for i in range(len(pred_logits)):
                for j in range(len(tgt_labels)):
                    cost_matrix[i, j] = self.get_cost(pred_logits[i], pred_boxes[i], tgt_labels[j], tgt_boxes[j], self.cost_class, self.cost_bbox)

            # print("#####################################")
            # print(" FINAL COST MATRIX")
            # print(cost_matrix.shape)
            # print(cost_matrix)
            # print("#####################################")

            pred_indices, tgt_indices = linear_sum_assignment(cost_matrix)
            # print("PREDICTION INDICES:", pred_indices)
            # print("TARGET INDICES:", tgt_indices)

            indices = torch.tensor([[torch.as_tensor(p, dtype=torch.int64), torch.as_tensor(t, dtype=torch.int64)] for p, t in zip(pred_indices, tgt_indices)])
            batch_indices.append(indices)

        return batch_indices

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, num_corners + padding, 3]

        print("out_prob", out_prob.shape)
        print("out_bbox", out_bbox.shape)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])  # [num_polygons]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [num_polygons, num_corners + padding]

        print("tgt_ids", tgt_ids.shape)
        print("tgt_bbox", tgt_bbox.shape)
        exit(0)

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

    def get_cost(self, a, b, c, d, cost_class, cost_bbox):

        pred_logits, pred_boxes, tgt_labels, tgt_boxes = \
            a.clone(), b.clone(), c.clone(), d.clone()

        # print("#########################")
        # print("COMPARING:")
        # print("pred_logits:\n", torch.softmax(pred_logits, dim=0))
        # print("pred_boxes:\n", pred_boxes)
        # print("tgt_labels:\n", tgt_labels)

        class_cost = 1 - torch.softmax(pred_logits, dim=0)[tgt_labels]

        # First we remove the points in target that are just for padding (-1 is used for padding)
        i = 0
        while i < len(tgt_boxes) and tgt_boxes[i] != -1:
            i += 1
        tgt_boxes = tgt_boxes[:i]

        # Then we reshape the target to have shape [num_points, 2]
        # and we add the 3rd element representing the "CONTINUE" class
        tgt_boxes = tgt_boxes.view(-1, 2)
        tgt_boxes = torch.cat([tgt_boxes, torch.zeros(tgt_boxes.shape[0], 1).to(tgt_boxes.device)], dim=1)

        # print("tgt_boxes:\n", tgt_boxes)
        # print("#########################")

        # If there are n corners in the polygon, we keep only the first n predictions of the RNN
        pred_boxes_points_only = pred_boxes[:len(tgt_boxes), :]

        # We initialize the point-to-point cost matrix for this polygon
        cost_matrix_coord = torch.zeros(len(pred_boxes_points_only), len(tgt_boxes))

        # The cost between two matching points is just the L1 loss
        for i in range(len(pred_boxes_points_only)):
            for j in range(len(tgt_boxes)):
                cost_matrix_coord[i][j] = torch.abs(pred_boxes_points_only[i] - tgt_boxes[j]).sum()

        # print("COST MATRIX for this polygon:")
        # print(cost_matrix_coord)

        ind_pred, ind_tgt = linear_sum_assignment(cost_matrix_coord)

        # print("Predition indices:", ind_pred)
        # print("Target indices:", ind_tgt)

        tgt_boxes = torch.cat([tgt_boxes[ind_tgt, :], torch.tensor([0.0, 0.0, 1.0]).to(tgt_boxes.device).expand(len(pred_boxes) - len(tgt_boxes), 3)], dim=0)

        # print("Final tgt_boxes:", tgt_boxes)
        # print("Final pred_boxes:", pred_boxes)

        # bbox_cost = torch.cdist(tgt_boxes, pred_boxes, 1).sum()
        bbox_cost = torch.abs(tgt_boxes - pred_boxes).sum()

        # print("FINAL COST:")
        # print(class_cost)
        # print(bbox_cost)

        return cost_class*class_cost + cost_bbox*bbox_cost


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
