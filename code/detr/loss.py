import torch
from munkres import Munkres, print_matrix

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
def get_ordered_matrices(labels: torch.Tensor, pred_logits_batched: torch.Tensor, pred_boxes_batched: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """ For now, this only works with batches of 1 image """

    # Labels.shape == [1, *, 4, 2]
    assert list(labels.shape)[0] == 1 and \
           list(labels.shape)[2] == 4 and \
           list(labels.shape)[3] == 2

    assert list(pred_logits_batched.shape) == [1, 100, 2] and \
           list(pred_boxes_batched.shape) == [1, 100, 8]

    print("###################")

    labels = labels[0].tolist()

    pred_logits = pred_logits_batched[0]
    pred_boxes = pred_boxes_batched[0]

    label_classes = []
    for _ in labels:
        label_classes.append(0)

    while len(labels) < list(pred_logits_batched.shape)[1]:
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
    print(indices)

    exit(-1)


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


if __name__ == '__main__':
    image = torch.load("test_tensors/image.pt")

    # plot_img = image.cpu()[0]
    # plt.imshow(plot_img.permute(1, 2, 0))
    # plt.show()

    labels = torch.load("test_tensors/labels.pt")
    pred_logits = torch.load("test_tensors/pred_logits.pt")
    pred_boxes = torch.load("test_tensors/pred_boxes.pt")

    matrix1, matrix2 = get_ordered_matrices(labels, pred_logits, pred_boxes)