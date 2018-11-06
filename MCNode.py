from random import choice
from math import sqrt, log

class MCNode():
    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent
        self.children = []
        self.c = 1
        self.totalScore = 0
        self.numberOfSimulations = 1


    def getChildNodes(self):
        """Returns a list containing all possible children of the node"""
        childNodes = []
        for childState in self.state.getChildStates():
            childNodes.append(MCNode(childState, self))
        return childNodes

    def expandNode(self):
        """Adds children to the node if not already expanded"""
        if len(self.children):
            raise Exception("Trying to expand a node that is already expanded")
        self.children = self.getChildNodes()

    def updateNodeValue(self, score):
        """Updates the value of the node"""
        self.totalScore += score
        self.numberOfSimulations += 1


    def ucb1(self, child, minTurn):
        """Returns the UCB value from this node to the given child
        minTurn = True if Q value should be minimized
        """
        qsa = child.totalScore/child.numberOfSimulations
        if minTurn:
            qsa *= -1
        usa = self.c * sqrt( log(self.numberOfSimulations) / (1+child.numberOfSimulations) )
        return qsa + usa

    def getBestUcbChild(self, minTurn):
        """Returns the child with the highest UCB value
        max(-qsa+usa) if minTurn.
        """
        if not len(self.children):
            raise Exception("Trying to get best UCB child from a node with no children")
        return max(self.children, key = lambda x:self.ucb1(x, minTurn))

    def getBestValueChild(self, minTurn):
        """Returns the child with highest Q value if not minTurn
        Lowest Q value if minturn
        """
        if not len(self.children):
            raise Exception("Trying to get best value child from a node with no children")
        if minTurn:
            return min(self.children, key = lambda x: x.totalScore/x.numberOfSimulations)
        return max(self.children, key = lambda x: x.totalScore/x.numberOfSimulations)

    def getBestVisitChild(self):
        """Returns the child with the highest amount of visits"""
        if not len(self.children):
            raise Exception("Trying to get best visit child from a node with no children")
        return max(self.children, key = lambda x: x.numberOfSimulations)

    def getRandomChild(self):
        """Returns a random child of this node"""
        tmp = self.children
        if not len(tmp):
            tmp = self.getChildNodes()
        return choice(tmp)