from math import sqrt
from hungarian_algorithm import algorithm


def corner_to_corner_loss(corner1: tuple, corner2: tuple) -> float:
    assert len(corner1) == len(corner2) and len(corner1) == 2
    return sqrt((corner1[0] - corner2[0])**2 + (corner1[1] - corner2[1])**2)


def gate_to_gate_loss(predicted: list, groud_truth: list, ordered_corners=False) -> float:
    """"
    This compares two gates and returns the loss between the two
    :param predicted: a list of size 4 of the coordinates of the corners of the predicted gate
    :param groud_truth: a list of size 4 of the coordinates of the corners of the ground truth gate
    :return: a single loss value
    """

    assert len(predicted) == len(groud_truth) and len(predicted) == 4

    loss = 0.0
    if ordered_corners:
        for i in range(len(predicted)):
            corner_loss = corner_to_corner_loss(predicted[i], groud_truth[i])
            print("Corner loss between {} and {} is {}".format(predicted[i], groud_truth[i], corner_loss))
            loss += corner_loss

        return loss


    # Creating the bipartite graph

    string_names_predicted = ['one_p', 'two_p', 'three_p', 'four_p']
    string_names_ground = ['one_g', 'two_g', 'three_g', 'four_g']

    bipartite_weighted_graph = {}

    for p_i in range(len(predicted)):
        corner_p_i = predicted[p_i]
        p_i_dict = {}
        for p_g in range(len(groud_truth)):
            corner_p_g = groud_truth[p_g]
            loss = corner_to_corner_loss(corner_p_i, corner_p_g)
            p_i_dict[string_names_ground[p_g]] = loss
        bipartite_weighted_graph[string_names_predicted[p_i]] = p_i_dict

    # print(bipartite_weighted_graph)

    res = algorithm.find_matching(bipartite_weighted_graph, matching_type='min', return_type='total')

    return res


if __name__ == '__main__':
    predicted = [(80, 100), (90, 40), (140, 32), (150, 113)]
    ground_truth = [(83, 110), (83, 32), (142, 29), (142, 113)]

    loss = gate_to_gate_loss(predicted, ground_truth, ordered_corners=False)

    print("LOSS: {}".format(loss))