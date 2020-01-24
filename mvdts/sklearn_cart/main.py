from sklearn.tree import DecisionTreeClassifier
import random


def build_tree(rows, min_point, max_depth=0):
    if max_depth == 0:
        tree = DecisionTreeClassifier(min_samples_leaf=min_point, criterion="entropy",
                                      random_state=random.randint(10, 999999))
    else:
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_point, criterion="entropy",
                                      random_state=random.randint(10, 999999))
    tree.fit(rows[0], rows[1])
    return tree
