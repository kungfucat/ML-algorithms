import numpy as np
import pandas as pd
from TreeNode import *


# returns the gini index of the system, independent of the attributes available
def rootGiniIndex(Y):
    diction = {}
    # 0th row, stores the name of the attributes, as it would be useful in printing the decision tree finally
    # vectors are from 1 to len(Y)

    length = len(Y) - 1
    # starting from 1, because 0 has attribute name
    for i in range(1, len(Y)):
        item = Y[i]
        if (item in diction.keys()):
            diction[item] = diction[item] + 1
        else:
            diction[item] = 1

    gini = 1.00
    index = 0
    for key in diction.keys():
        val = diction[key]
        # length variable is excluding the attribute name
        gini = gini - (val / length) ** 2
        index += 1

    return gini


# here X contains a 1d array that we want to find the gini index for
# e.g if for weather we want to find gini index,
# this function returns values for sunny and rainy etc. on seperate calls
def findAttrGini(X, Y, Key, noOfOc):
    diction = {}
    for i in range(0, len(X)):
        if (X[i] == Key):
            item = Y[i]
            if (item in diction.keys()):
                diction[item] = diction[item] + 1
            else:
                diction[item] = 1

    cur_gini = 1.0
    for key in diction.keys():
        val = diction[key]
        cur_gini = cur_gini - (val / noOfOc) ** 2
    return cur_gini


# For a single attribute, returns its gini index, weighted
def distinctClasses(X, Y):
    diction = {}
    # Find distinct items for given atprinttribute
    for i in range(1, len(X)):
        item = X[i]
        if (item in diction.keys()):
            diction[item] = diction[item] + 1
        else:
            diction[item] = 1

    # for weather, seperated sunny, rainy etc and took their weighted sum
    # as that will be the gini index for the given attribute
    length = len(Y) - 1
    giniVal = 0.00
    for key in diction.keys():
        val = diction[key]
        cur = findAttrGini(X, Y, key, val)
        cur = cur * (val / length)
        giniVal = giniVal + cur

    return giniVal, diction


# This function returns the new dataset
# TODO: can optimse this
# for e.g if we choose weather as attribute, we want to seperate dataset for weather=sunny and solve it further
# similarly for all other values
def reduceForChild(X, y, key, indexAsRoot, val):
    X_red = np.ndarray(shape=(val + 1, len(X[0])), dtype=object)
    # y needed to be an nparray, not an np.ndarray
    lst_y = []
    # Appended names of attributes
    X_red[0] = X[0]
    lst_y.append(y[0])
    cntr = 1
    # Appended attributes corresponding to key
    for i in range(1, len(X[:, indexAsRoot])):
        if (X[i][indexAsRoot] == key):
            X_red[cntr] = X[i]
            lst_y.append(y[i])
            cntr += 1

    return X_red, np.array(lst_y)


# For solving a leaf node, if we are left with no attributes at all
# we simply return the maximum occurence of any given y
def maxOccurence(y):
    dictionary = {}
    for i in range(1, len(y)):
        item = y[i]
        if (item in dictionary):
            dictionary[item] += 1
        else:
            dictionary[item] = 1

    maxCnt = 0
    answer = ""
    for attr in dictionary:
        count = dictionary[attr]
        if (count > maxCnt):
            answer = attr
            maxCnt = count

    return answer


def solve(X, y):
    # calculate the gini index for the system
    datasetGiniIndex = rootGiniIndex(y)
    # an empty node
    node = NodeStruct()

    # only one type of answer remaining in the datset
    if (datasetGiniIndex == 0):
        node.setAnswer(y[1])
        return node
        # return node

    # for no attributes remaining
    if (len(X[0]) == 0):
        node.setAnswer(maxOccurence(y))
        return node
        # return node

    minGiniIndex = 2
    indexAsRoot = -1
    maxdiction = {}

    # find the attribute which minimises the gini index of the system
    for ind in range(0, len(X[0])):
        curGini, diction = distinctClasses(X[0:, ind], y[0:])

        if (curGini < minGiniIndex):
            minGiniIndex = curGini
            indexAsRoot = ind
            maxdiction = diction

    node.setName(X[0][indexAsRoot])
    # Delete the attribute considered currently
    # say we chose weather
    # then seperate the dataset for weather=sunny and other possible values
    # and remove the attribute 'weather' from it
    edgeList = {}
    for key in maxdiction:
        X_red, y_red = reduceForChild(X, y, key, indexAsRoot, maxdiction[key])
        edgeList[key] = solve(np.delete(X_red, indexAsRoot, 1), y_red)

    node.edges = edgeList
    return node


if __name__ == "__main__":
    # Importing the dataset
    dataset = pd.read_csv('dataset.csv', header=None)
    X = dataset.iloc[:, 0:3].values
    y = dataset.iloc[:, 3].values

    # solve returns the root of the decision tree
    node = solve(X, y)
    # prettify the tree,and then print that tree
    ppNode = getPrettyTree(node)
    prettyPrinter(ppNode)
