from dt_common.nodes import Leaf, DecisionNode
from dt_common.common import random_features_selection, check_purity, partition, entropy
import numpy as np

# others from topic presentation
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import matplotlib.pyplot as plt


def viz_data_with_line_np(theta, split_list):
    # this will show decision boundary as line with scatter plot
    
    theta = theta.reshape(split_list[0].shape[1] + 1, 1)
    
    # Ploting Line, decision boundary
    theta_f = list(theta.flat)
    
    # Calculating line values x and y
    # y = np.arange(-10, 10, 0.1)
    # x = (-theta_f[2] - theta_f[1] * y) / theta_f[0]

    # ref #https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
    # https://stackoverflow.com/questions/42704698/logistic-regression-plotting-decision-boundary-from-theta

    x1_line = np.arange(int(round(split_list[0].min())), int(round(split_list[0].max())), 0.1)
    x2_line = (-theta_f[2] - theta_f[0] * x1_line) / theta_f[1]
    # x2 = - (theta_f[2] + np.dot(theta_f[0], x)) / theta_f[1]
    plt.plot(x1_line, x2_line, label='Decision Boundary')

    # zooming plots
    x1 = split_list[0][:, 0]
    x2 = split_list[0][:, 1]
    # increase in direction
    # x1.min-1 left, x2.max+1: bottom, x1.max+1: right:, x2.min-7: top 
    X_min_max = [int(round(x1.min() - 2)), int(round(x1.max() + 2))]
    X2_min_max = [int(round(x2.min() - 6)), int(round(x2.max() + 2))]
    plt.xlim(X_min_max[0], X_min_max[1])
    plt.ylim(X2_min_max[0], X2_min_max[1])

    # scatter plot
    categories = split_list[1]
    colormap = np.array(['#277CB6', '#FF983E'])
    
    plt.scatter(x1, x2, c=colormap[categories])


#     plt.show()

def plot_data(theta, split_list):
    weights = theta
    inputs = split_list[0]
    targets = split_list[1] 
    # fig config
#     plt.figure(figsize=(8,8))
    plt.grid(True)

    #plot input samples(2D data points) and i have two classes. 
    #one is +1 and second one is -1, so it red color for +1 and blue color for -1
    for input,target in zip(inputs,targets):
        plt.plot(input[0],input[1],'ro' if (target == 1.0) else 'bo')

    # Here i am calculating slope and intercept with given three weights
    for i in np.linspace(np.amin(inputs[:,:1]),np.amax(inputs[:,:1])):
        theta = weights.reshape(inputs.shape[1] + 1, 1)
    
        # Ploting Line, decision boundary
        theta_f = list(weights.flat)

        x1_line = np.arange(int(round(inputs.min())), int(round(inputs[0].max())), 0.1)
        x2_line = (-theta_f[2] - theta_f[0] * x1_line) / theta_f[1]
    
        # x2 = - (theta_f[2] + np.dot(theta_f[0], x)) / theta_f[1]
        plt.plot(x1_line, x2_line, label='Decision Boundary')
        
def best_split(rows, epochs, noOfFeature, algo='logit'):
    # get random x
    
#     idx, selectedFeatures, types, rand_x = random_features_selection(rows, noOfFeature)
    
    # solvers ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    
    if algo == 'logit':
        clf = LogisticRegression(solver='liblinear', max_iter=epochs)
    elif algo == 'perceptron':
        clf = Perceptron(max_iter=epochs)
    elif algo == 'linearsvc':
        clf = LinearSVC(max_iter=epochs)
    elif algo == 'svc':
        clf = SVC(kernel='linear', max_iter=epochs)   
    else:
        print("Deafult linear classifier has choosen", algo)
    # rows[0]: features, rows[1]:label
    
    
    clf.fit(rows[0], rows[1])
    
    theta = np.append(clf.coef_, clf.intercept_)
    
#     print(theta)
#     if len(theta) == 0:
#         print("no decision boundary", theat)
    
    # pred = (clf.predict_proba(X_test)[:,1] >= 0.3).astype(bool)
    pred = clf.predict(rows[0])

    return clf, theta, pred

def build_tree(rows, epochs, algo, min_point=2, max_depth=4, depth=0, noOfFeature=2, count=0):
    # check for max depth of tree if yes then stop building
    label_1, label_0 = check_purity(rows[-1])
    # if max depth is given

    if max_depth != 0:
        if depth > max_depth:
            #print("from max depth reached, depth {} reached, 1:{}, 0:{}, min_point: {}".format(depth, label_1, label_0, min_point))
            return Leaf(rows[-1].reshape(len(rows[0]), 1))

    if label_1 > min_point and label_0 > min_point:

        # random feature pair selection, and host idx and rand_x
        idx, selectedFeatures, types, rand_x = random_features_selection(rows, noOfFeature)

        model, theta, pred = best_split([rand_x, rows[-1]], epochs, algo)
        
        # if no decision bound then change index

#         if len(theta) == 0:
#             print("out of decision boundy", theta)
#             build_tree(rows, epochs, algo)
                
        pos_side = np.where(pred > 0)
        neg_side = np.where(pred <= 0)
        
#         print(len(pos_side[0]), len(neg_side[0]))

        # getting left and right branch subset after split
        true_rows, false_rows = partition(rows, pos_side, neg_side)

        # entropy measure
        e = entropy(len(pos_side[0]), len(neg_side[0]))

        #print(len(true_rows[0]), len(false_rows[0]))
        # checking purity of each branch
        left_1, left_0 = check_purity(true_rows[-1])
        right_1, right_0 = check_purity(false_rows[-1])
#         print("depth {}, entropy: {}, total rows:{} left branch labels [1={}, 0={}],right branch labels [1={}, 0={}],"
#              " min_point: {} \n".format(depth, e, len(rows[0]),left_1, left_0, right_1, right_0, min_point))
        if e == 0:
#             print("out of boundary decision", e)
#             print("from label min point, depth {} reached, 1:{}, 0:{}, min_point: {}"
#                   .format(depth, label_1, label_0, min_point))
#           if total len is equal to one side branch either left or right then change random feature selection
#           because above min condition will automaticall checks if any postive and negive are in one side
            if len(rows[0]) == len(true_rows[0]) or len(rows[0]) == len(false_rows[0]):
                if count < 10:
#                 sakiya ba with linear function only
                    print("total rows: {}, left_rows: {}, right_rows: {}, epochs: {}, algo: {}, depth: {}, count: {}, idx: {}, question: {}"
                          .format(len(rows[0]), len(true_rows[0]), len(false_rows[0]), epochs, algo, depth, count, idx, theta))                
                    build_tree(rows, epochs, algo, depth=depth, noOfFeature=noOfFeature, count=count+1)
                
#                 I have tried updating feature index if entropy:0 with out any proper decision boundary but still selecting all the features gives better results. Therefore I proposed to first splitting using all features until if the e:0 then randomly select minimum features starting from 2 
# 
#                 if noOfFeature == train_data[0].shape[1]:
#               select min
#                     select_min = 0
#                     build_tree(rows, epochs, algo, depth=depth, noOfFeature=select_min+2, count=count+1)
                    
#             if depth 0 then is root node
            
            return Leaf(rows[-1].reshape(len(rows[0]), 1))
        else:
            
            if len(rows[0][0]) == 2:
                viz_data_with_line_np(theta, rows)
#                 plot_data(theta, rows)

            true_branch = build_tree(true_rows, epochs, algo, min_point=2, depth=depth + 1,
                                     noOfFeature=noOfFeature, count=0)
            false_branch = build_tree(false_rows, epochs, algo, min_point=2, depth=depth + 1,
                                      noOfFeature=noOfFeature, count=0)
            return DecisionNode(theta, true_branch, false_branch, rows, idx, [left_1, left_0], [right_1, right_0], depth
                                , e)
    else:
#         print("from label min point, depth {} reached, 1:{}, 0:{}, min_point: {}"
#               .format(depth, label_1, label_0, min_point))
        return Leaf(rows[-1].reshape(len(rows[0]), 1))
