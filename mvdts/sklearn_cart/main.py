from sklearn.tree import DecisionTreeClassifier

def build_tree(rows, max_depth):
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20)
    tree.fit(rows[0], rows[1])
    return tree
