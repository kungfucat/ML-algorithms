# TreeNode represents a single node in the tree
# name is empty if the node is a leaf node, otherwise, it will definitely have a name
from pptree import *


class NodeStruct:
    def __init__(self, name="", dict={}):
        self.name = name
        self.edges = dict
        self.answer = "No_answer"

    def add(self, key, root):
        self.edges[key] = root

    def setAnswer(self, val):
        self.answer = val

    def setName(self, name):
        self.name = name

    def setEdges(self, edges):
        self.edges = edges

def getPrettyTree(node, parent=None):
    if (node.name == ""):
        newNode = Node(node.answer, parent)
        return newNode

    newNode = Node(node.name, parent)
    for key in node.edges.keys():
        e_child = Node(key,newNode)
        getPrettyTree(node.edges[key], e_child)

    return newNode


def prettyPrinter(node):
    print_tree(node)
