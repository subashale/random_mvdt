   
class Question:
    # initialise column and value variables->
    # eg->if ques is ->is sepal_length>=1cm then
    # sepal_length==col and 1cm=value
    def __init__(self, question, value=0):
        self.question = question
        self.value = value

    #     def match(self, example):
    #         # Compare the feature value in an example to the
    #         # feature value in this question.
    #         val = example[self.column]
    #         if is_numeric(val):
    #             return val >= self.value
    #         else:
    #             return val == self.value

    def match(self, data):
        value = data[self.column]
        return value >= self.value

    # This is just a helper method to print
    # the question in a readable format.
    def __repr__(self):
        condition = ">="
        return "Is %s %s %s?" % (
            str(self.theta), condition, str(self.value))


# this class represents all nodes in the tree
class DecisionNode:
    def __init__(self, question, true_branch, false_branch, rows):
        # question object stores col and val variables regarding the question of that node
        self.question = question  # question = theta
        # this stores the branch that is true
        self.true_branch = true_branch
        # this stores the false branch
        self.false_branch = false_branch
        # store split parts
        self.rows = rows


# Leaf class is the one whichs tores leaf of trees
# it includes question as last decision boundary
# and prediction is based on predict() on LR algo
class Leaf():
    def __init__(self, question, rows):
        self.question = question
        self.rows = rows
