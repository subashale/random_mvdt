class DecisionNode:
    # left = 1/true and right = 0/false
    # store depth and node information in dictionary
    def __init__(self, question, true_branch, false_branch, rows, indexes, left_instances, right_instances, depth):
        # question object stores col and val variables regarding the question of that node
        self.question = question  # question = theta
        # this stores the branch that is true
        self.true_branch = true_branch
        # this stores the false branch
        self.false_branch = false_branch
        # store split parts
        self.rows = rows
        # store index of feature
        self.indexes = indexes
        # instance of each branch in [] or label distribution
        self.true_purity = left_instances
        self.false_purity = right_instances
        self.depth = depth


# Leaf class is the one whichs tores leaf of trees
# it includes question as last decision boundary
# and prediction is based on predict() on LR algo

class Leaf:
    def __init__(self, label):
        self.predictions = class_counts(label)

# count value from label, rows takes only labels in (len(rows), 1) format
def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts
