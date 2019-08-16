import numpy as np
from math import log2
from numpy import reshape

# for calculating entropy, this can be change based on requirement
def entropy(a, b):
    # a, b must have integer number

    if a == 0:
        return 0
    elif b == 0:
        return 0
    else:
        c = a + b
        e = -(a/c) * log2(a/c) - (b/c) * log2(b/c);
        return e

# checking how many points are lies in both side with a line, counting missclassification
def check_positive_negative(points, label):
    point_positive = 0
    point_negative = 0
    
    for i in points:
        if label[i] == 1:
            point_positive = point_positive + 1
        elif label[i] == 0:
            point_negative = point_negative + 1
        else:
            print(label[i])

    return point_positive, point_negative


# split data with linear combination of feature

def split_data(X, label, pos, neg, p, n):
    # X is all feature combine
    # label is true value/category/calss
    # pos positive values from X
    # neg negative value from X
    # p, n points from pos and neg from X

    # find a parameter that go through a line, a real value
    # slope of line or compute as  slopes = (y2 - y1) / (x2 - x1)
    v = p - n

    # normalization using np.linalg.norm(v) or this is L2 norm
    # provides vector magnitude or euclidean length of the vector
    # sqrt(a1^2 + a2^2 + a3^2) or    sqrt(sum(x.^2))
    # give single positive value using euclidean formula
    # method to keep the coefficients of the model small, and in turn, the model less complex.
    # https://machinelearningmastery.com/vector-norms-machine-learning/
    v /= np.linalg.norm(v)

    # find mid point for two categories, assumption that go through all the points
    center = (p + n) / 2

    # reshape in to (2*1)
    center = center.reshape((len(center), -1))

    #b = -mx + y
    bias = -v.dot(center)

    # all parameters as theta
    theta = np.append(v, bias)

    # new directed line shape must have (10, 2) (1, 10) with 10*1 dataset
    # ones for intercept, every time b*1 = b
    ones = np.ones(len(X))
    ones = ones.reshape(1, -1)
    with_one_combine = np.concatenate((X, ones.T), axis=1)

    theta = theta.reshape((len(theta), -1))
    # way of finding data from decision line
    # it remains as y_hat and finds left and right value with y = mx+b
    # now we have y_hat
    direction = with_one_combine.dot(theta)

    # print all calculation
    # print(v, center, bias, theta, direction)

    #checking purity of a line

    # all the position lies both side, it will only take index
    # limit is >=0: 1 and 0 rest
    pos_side = np.where(direction >= 0)
    neg_side = np.where(direction < 0)

    # checking miss classified points
    # we use pos_side, neg_side index to check on actual label
    # all the lies value are used on entropy calculation
    point_pos_pos, point_pos_neg = check_positive_negative(pos_side[0], label)
    point_neg_pos, point_neg_neg = check_positive_negative(neg_side[0], label)

    # print(point_pos_pos, point_pos_neg, point_neg_pos, point_neg_neg)

    # calculating total entropy
    total_entropy = entropy(len(pos), len(neg))

    # calculation entropy for each side of after split
    entropy_positive = entropy(point_pos_pos, point_pos_neg)
    entropy_negative = entropy(point_neg_pos, point_neg_neg)

    # print(entropy_positive, entropy_negative)

    # calculation information gain, from first split
    pos_fraction = len(pos_side[0]) / len(X)
    neg_fraction = len(neg_side[0]) / len(X)

    gain = total_entropy - pos_fraction * entropy_positive \
           - neg_fraction * entropy_negative

    return gain, pos_side[0], neg_side[0], theta

# def split_data(X, label, pos, neg, n, p):
#     # X is all feature combine
#     # label is true value/category/calss
#     # pos positive values from X
#     # neg negative value from X
#     # random_mvdt points for both pos and neg from X
#
#     total_entropy = entropy(len(pos), len(neg))
#
#     # find a parameter that go through a line
#     v = p-n
#     v /= np.linalg.norm(v)
#
#     # go through all the points
#     center = (p+n)/2
#     # reshape in to (2*1)
#     center = center.reshape((len(center), -1))
#
#     bias = -v.dot(center)
#
#     theta = np.append(v, bias)
#
#     # new directed line shape must have (10, 2) (1, 10) with 10*1 dataset
#     ones = np.ones(len(X))
#     ones = ones.reshape(1,-1)
#     with_one_combine = np.concatenate((X, ones.T), axis=1)
#     theta = theta.reshape((len(theta), -1))
#
#     direction = with_one_combine.dot(theta)
#
#     # print all calculation
#     print(v, center, bias, theta, direction)
#
#     # all the position lies both side
#     pos_side = np.where(direction >= 0)
#     neg_side = np.where(direction < 0)
#
#     # checking miss classified points for both labels,
#     point_pos_pos, point_pos_neg = check_positive_negative(pos_side[0], label)
#     point_neg_pos, point_neg_neg = check_positive_negative(neg_side[0], label)
#
#     print(point_pos_pos, point_pos_neg, point_neg_pos, point_neg_neg)
#
#     # calculating total entropy
#     total_entropy = entropy(len(pos), len(neg));
#
#     # calculation entropy for each side of after split
#     entropy_positive = entropy(point_pos_pos, point_pos_neg);
#     entropy_negative = entropy(point_neg_pos, point_neg_neg);
#
#     print(entropy_positive, entropy_negative)
#
#     # calculation information gain, from first split
#     pos_fraction = len(pos_side[0]) / len(X);
#     neg_fraction = len(neg_side[0]) / len(X);
#
#     gain = total_entropy - pos_fraction * entropy_positive \
#            - neg_fraction * entropy_negative;
#
#     return gain, pos_side[0], neg_side[0], theta