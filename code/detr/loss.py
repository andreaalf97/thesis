import torch
from munkres import Munkres

"""
As input we have:
    labels: Tensor of shape [1, 2, 4, 2] = [batch_size, NUM_GATES, 4 corners, 2 coords per corner]
    pred_logits: Tensor of shape [1, 100, 2] = [batch_size, num_predictions, num_classes]
    pred_boxes: Tensor of shape [1, 100, 8] = [batch_size, num_predictions, 8 coords (4 corners x-y)

    The idea is to generate two matrices that match the same shape as pred_logits and pred_boxes, and
    then compute the loss as the point-wise difference (L1 or L2 or...) between the prediction and
    ground truth matrices
    
    0 is GATE class index
    1 is NO_OBJ class index
"""


@torch.no_grad()
def get_ordered_matrix(labels: torch.Tensor, pred_logits: torch.Tensor, pred_boxes: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """ For now, this only works with batches of 1 image """

    # Labels.shape == [*, 4, 2]
    if not list(labels.shape) == [0]:
        assert (list(labels.shape)[1] == 4 and \
           list(labels.shape)[2] == 2)

    assert list(pred_logits.shape) == [100, 2] and \
           list(pred_boxes.shape) == [100, 8]

    labels = labels.tolist()

    label_classes = []
    for _ in labels:
        label_classes.append(0)

    while len(labels) < list(pred_logits.shape)[0]:
        labels.append([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ])
        label_classes.append(1)

    #  pred_logits = [100, 2]
    #  pred_boxes = [100, 8]
    #  labels = [100, 4, 2]
    #  label_classes = [100, 1]
    bipartite_match = [[0 for _ in range(100)] for _ in range(100)]
    for i in range(len(pred_logits)):
        for j in range(len(labels)):
            bipartite_match[i][j] = gate_to_gate(pred_logits[i], pred_boxes[i], labels[j], label_classes[j])

    m = Munkres()

    indices = m.compute(bipartite_match)

    # Here we create 2 matrices:
    # output_logits has shape [100, 1] (100 predictions x 1 class index)
    # output_boxes has shape [100, 8] (100 predictions x 8 coordinates)
    # The loss is then just the cross entropy with output_logits and the l1 distance with output_boxes

    output_logits = torch.zeros([100, 1], dtype=torch.int32)
    output_boxes = torch.zeros([100, 8], dtype=torch.float64)
    for pred_index, label_index in indices:
        output_logits[pred_index] = label_classes[label_index]
        if label_classes[label_index] != 1:
            output_boxes[pred_index] = torch.flatten(torch.tensor(labels[label_index]))
        else:
            output_boxes[pred_index] = pred_boxes[pred_index]

    return output_logits, output_boxes


@torch.no_grad()
def gate_to_gate(pred_logit: torch.Tensor, pred_box: torch.Tensor, label_coord: list, label_class: int, coord_loss='l1') -> float:

    assert list(pred_logit.shape) == [2] and list(pred_box.shape) == [8]
    assert len(label_coord) == 4 and len(label_coord[0]) == 2
    assert label_class == 0 or label_class == 1

    no_obj = label_class == 1

    # print("-----\nComputing loss between:\nLogit: {}\ncoord: {}\n-----\nAND\nLabel: {}\nLabel class: {}\n-----".format(pred_logit, pred_box, label_coord, label_class))
    pred_logit = torch.softmax(pred_logit, dim=0)
    final_loss = -torch.log(pred_logit[label_class])

    if not no_obj:
        pred_box = pred_box.cpu()
        label_coord = torch.tensor(label_coord).flatten()

        if coord_loss == 'l1':
            loss_matr = torch.abs(pred_box - label_coord)
            final_loss += torch.sum(loss_matr)

    return final_loss.item()


def compute_prob_loss(pred_logits_batched: torch.Tensor, gt_logits_batched: torch.Tensor) -> torch.Tensor:

    loss = torch.tensor([0.0], dtype=torch.float64).to(torch.device(str(pred_logits_batched.device)))
    for pred_logits, gt_logits in zip(pred_logits_batched, gt_logits_batched):
        for prediction, gt in zip(pred_logits, gt_logits):
            softmax_prediction = torch.softmax(prediction, dim=0)
            loss += -torch.log(softmax_prediction[gt.item()])
    return loss


def compute_coord_loss(pred_boxes_batched: torch.Tensor, gt_boxes_batched: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(gt_boxes_batched - pred_boxes_batched))

if __name__ == '__main__':
    image = torch.load("test_tensors/image.pt")

    # plot_img = image.cpu()[0]
    # plt.imshow(plot_img.permute(1, 2, 0))
    # plt.show()

    labels = torch.load("test_tensors/labels.pt")
    pred_logits = torch.load("test_tensors/pred_logits.pt")
    pred_boxes = torch.load("test_tensors/pred_boxes.pt")

    gt_logits, gt_boxes = get_ordered_matrix(labels[0], pred_logits[0], pred_boxes[0])

    gt_logits = torch.unsqueeze(gt_logits, dim=0).cuda()
    gt_boxes = torch.unsqueeze(gt_boxes, dim=0).cuda()

    prob_loss = compute_prob_loss(pred_logits, gt_logits)
    box_loss = compute_coord_loss(pred_boxes, gt_boxes)

    print("PROB LOSS: {}\nBOX LOSS: {}\nTOTAL LOSS: {}".format(prob_loss, box_loss, prob_loss+box_loss))