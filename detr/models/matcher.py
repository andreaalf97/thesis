# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


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
def get_sequence(polygons: list, target: dict, max_sequence_len: int):
    """
    This is repeated for each batch
        This function takes the list of prediction polygons and the list of ground truth polygons of the same batch
        and outputs the optimal target sequence after matching them.
        What we want to do is find a the optimal match between the first point of each prediction and the points of each polygon
    """

    num_tgt_polygons = target['boxes'].shape[0]

    # tgt_poly_shapes is basically a list of the number of points of each polygon in the image
    tgt_poly_shapes = target['poly_shapes']

    # We also reshape the target polygons to have x, y in a single dimension.
    # The final shape of the tgt tensor is [num_tgt_polygons, num_points_per_polygon(with padding), 2]
    tgt = target['boxes'].view(num_tgt_polygons, -1, 2)

    if len(polygons) < 1:
        tgt_ids, pred_ids = [], []
    else:
        # First we stack the FIRST point of the prediction polygons in a single tensor that has
        # final shape [num_pred_polygons, 2]
        first_point_poly = torch.stack([p[0] for p in polygons])

        # What we want is to compute the cost matrix (p2 distance) between all first points of the prediction polygons
        # and all the points of the target polygons in a single operation

        # dist has shape [num_tgt_polygons, 7 (points per polygon with padding), num_pred_polygons] and
        # contains the p2 distances
        dist = torch.cdist(tgt, first_point_poly, p=2)

        # We find, for each prediction-target pair, the closest point. This distance is the cost of this pair
        # dist has shape [num_target_polygons, num_pred_polygons]
        dist, _ = torch.min(dist, dim=1)

        # linear_sum_assignment returns the correct target-prediction indices
        tgt_ids, pred_ids = linear_sum_assignment(dist.cpu())

    print('\n'.join([f"target {t} ({tgt_poly_shapes[t]} points)was matched with prediction {p} ({len(polygons[p])} points)" for t, p in zip(tgt_ids, pred_ids)]))

    # Now we start creating the optimal target sequences, one for classification and one for point coordinates
    target_sequence_class = torch.tensor([HungarianMatcher.CLASSES['<start>']], dtype=torch.float32, device=target['boxes'].device)
    # Eventually we want to have a zero loss for stuff that is not classified as <point>
    target_sequence_points = torch.tensor([-1, -1], dtype=torch.float32, device=target['boxes'].device).unsqueeze(0)
    for tgt_id, pred_id in zip(tgt_ids, pred_ids):
        num_tgt_points = tgt_poly_shapes[tgt_id]

        extend_class = torch.tensor([HungarianMatcher.CLASSES['<point>'] for _ in range(num_tgt_points)],
                                    device=target_sequence_class.device)
        target_sequence_class = torch.cat([
            target_sequence_class,
            extend_class,
            torch.tensor([HungarianMatcher.CLASSES['<end-of-polygon>']], dtype=torch.float32,
                         device=extend_class.device)
        ])

        # If the prediction has a different amount of points from the ground truth, we compute the loss with a
        # random permutation of the target points
        if len(polygons[pred_id]) != num_tgt_points:
            perm = torch.randperm(num_tgt_points)

            target_sequence_points = torch.cat([
                target_sequence_points,
                tgt[tgt_id][:num_tgt_points][perm],
                torch.tensor([-1, -1], dtype=torch.float32, device=target_sequence_points.device).unsqueeze(0)
            ])
        # Otherwise, we let the architecture output the points in the order it prefers
        else:
            point_to_point_cost = torch.cdist(tgt[tgt_id][:num_tgt_points], polygons[pred_id], p=2)
            tgt_points_ids, pred_points_ids = linear_sum_assignment(point_to_point_cost.cpu())

            target_sequence_points = torch.cat([
                target_sequence_points,
                tgt[tgt_id][:num_tgt_points][pred_points_ids],
                torch.tensor([-1, -1], dtype=torch.float32, device=target_sequence_points.device).unsqueeze(0)
            ])

    # We now need to add to the sequence the polygons that were not predicted by the model
    # Again, we add the polygons with a random permutation of the points
    for i in range(num_tgt_polygons):
        if i not in tgt_ids:
            num_tgt_points = tgt_poly_shapes[i]

            extend_class = torch.tensor([HungarianMatcher.CLASSES['<point>'] for _ in range(num_tgt_points)],
                                        device=target_sequence_class.device)
            target_sequence_class = torch.cat([
                target_sequence_class,
                extend_class,
                torch.tensor([HungarianMatcher.CLASSES['<end-of-polygon>']], dtype=torch.float32,
                             device=extend_class.device)
            ])

            perm = torch.randperm(num_tgt_points)

            target_sequence_points = torch.cat([
                target_sequence_points,
                tgt[i][:num_tgt_points][perm],
                torch.tensor([-1, -1], dtype=torch.float32, device=target_sequence_points.device).unsqueeze(0)
            ])

    # Finally, we close the sequence
    target_sequence_class = torch.cat([
        target_sequence_class,
        torch.tensor(
            [HungarianMatcher.CLASSES['<end-of-computation>'] for _ in range(max_sequence_len - len(target_sequence_class))], dtype=torch.float32, device=extend_class.device
        )
    ])
    target_sequence_points = torch.cat([
        target_sequence_points,
        torch.tensor([[-1, -1] for _ in range(max_sequence_len - len(target_sequence_points))], dtype=torch.float32, device=target_sequence_points.device).view(-1, 2)
    ])

    return target_sequence_class, target_sequence_points


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    CLASSES = {
        "<start>": 0,
        "<point>": 1,
        "<end-of-polygon>": 2,
        "<end-of-computation>": 3
    }

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

        class_seq = []
        coord_seq = []
        for polygon, target in zip(polygons, targets):
            # With the prediction polygons, we can match them to the tgt polygons and create our target sequence
            cl, coord = get_sequence(polygon, target, num_queries)
            class_seq.append(cl)
            coord_seq.append(coord)

        class_seq = torch.stack(class_seq)
        coord_seq = torch.stack(coord_seq)

        return class_seq, coord_seq

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
