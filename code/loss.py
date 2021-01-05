from math import sqrt
from hungarian_algorithm import algorithm


def corner_to_corner_loss(corner1: tuple, corner2: tuple) -> float:
    assert len(corner1) == len(corner2) and len(corner1) == 2
    return sqrt((corner1[0] - corner2[0])**2 + (corner1[1] - corner2[1])**2)


def gate_to_gate_loss(predicted: list, ground_truth: list, ordered_corners=False) -> float:
    """"
    This compares two gates and returns the loss between the two
    :param predicted: a list of size 4 of the coordinates of the corners of the predicted gate
    :param ground_truth: a list of size 4 of the coordinates of the corners of the ground truth gate
    :return: a single loss value
    """

    assert len(predicted) == len(ground_truth) and len(predicted) == 4

    loss = 0.0
    if ordered_corners:
        for i in range(len(predicted)):
            corner_loss = corner_to_corner_loss(predicted[i], ground_truth[i])
            # print("Corner loss between {} and {} is {}".format(predicted[i], ground_truth[i], corner_loss))
            loss += corner_loss

        return loss


    # Creating the bipartite graph

    string_names_predicted = ['one_p', 'two_p', 'three_p', 'four_p', 'five_p', 'six_p', 'seven_p', 'eight_p', 'nine_p']
    string_names_ground = ['one_g', 'two_g', 'three_g', 'four_g', 'five_q', 'six_q', 'seven_q', 'eight_q', 'nine_q']

    bipartite_weighted_graph = {}

    for p_i in range(len(predicted)):
        corner_p_i = predicted[p_i]
        p_i_dict = {}
        for p_g in range(len(ground_truth)):
            corner_p_g = ground_truth[p_g]
            loss = corner_to_corner_loss(corner_p_i, corner_p_g)
            p_i_dict[string_names_ground[p_g]] = loss
        bipartite_weighted_graph[string_names_predicted[p_i]] = p_i_dict

    # print(bipartite_weighted_graph)

    res = algorithm.find_matching(bipartite_weighted_graph, matching_type='min', return_type='total')

    return res


def orderless_loss(predicted: list, ground_truth: list) -> float:

    for gate in predicted:
        assert len(gate) == 4
        for corner in gate:
            assert len(corner) == 2

    for gate in ground_truth:
        assert len(gate) == 4
        for corner in gate:
            assert len(corner) == 2

    string_names_predicted = ['one_p', 'two_p', 'three_p', 'four_p', 'five_p', 'six_p', 'seven_p', 'eight_p', 'nine_p']
    string_names_ground = ['one_g', 'two_g', 'three_g', 'four_g', 'five_q', 'six_q', 'seven_q', 'eight_q', 'nine_q']

    bipartite_weighted_graph = {}

    for p_i in range(len(predicted)):
        gate_p_i = predicted[p_i]
        p_i_dict = {}
        for p_g in range(len(ground_truth)):
            gate_p_g = ground_truth[p_g]
            l = gate_to_gate_loss(gate_p_i, gate_p_g, ordered_corners=True)
            if l == 0.0:
                l += 0.0001
            p_i_dict[string_names_ground[p_g]] = l
        bipartite_weighted_graph[string_names_predicted[p_i]] = p_i_dict

    print(bipartite_weighted_graph)

    for name_p in bipartite_weighted_graph:
        print("Dictionary associated with {}: {}".format(name_p.upper(), bipartite_weighted_graph[name_p]))


    res = algorithm.find_matching(bipartite_weighted_graph, matching_type='min', return_type='total')

    # print(res)

    return res



if __name__ == '__main__':
    # predicted = [(80, 100), (90, 40), (140, 32), (150, 113)]
    # ground_truth = [(83, 110), (83, 32), (142, 29), (142, 113)]

    predicted = [[(70, 69), (70, 55), (92, 46), (88, 65)], [(130, -14), (131, 0), (135, -15), (145, 10)], [(33, 105), (29, 15), (77, 28), (79, 85)], [(0,0), (0,0), (0,0), (0,0)]]
    ground_truth = [[(135, -14), (135, -16), (138, -15), (138, -15)], [(30, 100), (30, 15), (81, 29), (81, 85)], [(75, 62), (75, 48), (90, 49), (90, 61)], [(0,0), (0,0), (0,0), (0,0)]]

    loss = orderless_loss(predicted, ground_truth)

    print("LOSS: {}".format(loss))
