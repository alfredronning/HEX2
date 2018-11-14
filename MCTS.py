import numpy as np
from sklearn import preprocessing
from copy import deepcopy
from HexState import HexState
from MCNode import MCNode

class MCST():
    def __init__(self, startNode, anet, replayBuffer, numberOfSimulations):
        self.rootNode = startNode
        self.startingPlayer = self.rootNode.state.player
        self.numberOfSimulations = numberOfSimulations

        self.anet = anet
        self.replayBuffer = replayBuffer

    def findNextMove(self, currentNode):
        """Finds the next move for the actual game"""
        for _ in range(self.numberOfSimulations):
            #selection with UCB untill unvisited node
            selectedNode = self.threeSearch(currentNode)

            #expand node if needed
            self.expand(selectedNode)

            if len(selectedNode.children):
                selectedNode = selectedNode.getRandomChild()

            #simulate a single rollout
            #score = self.rollout(selectedNode) #27.7s
            score = self.rollout2(selectedNode) #20.9s
            #score = self.randomRollout(selectedNode) #5.0s
            #score = self.randomRollout2(selectedNode) #3.9s

            #backpropogate the score from the rollout from the selected node up to root
            self.backPropagate(selectedNode, score)

        self.addToReplayBuffer(currentNode)
        return currentNode.getBestVisitChild()

    def threeSearch(self, currentNode):
        """Returns the first unvisited node with UCB policy"""
        tmpNode = currentNode
        if tmpNode is not None:
            while(len(tmpNode.children)):
                minTurn = tmpNode.state.player != self.startingPlayer
                tmpNode = tmpNode.getBestUcbChild(minTurn)
                if tmpNode.numberOfSimulations == 1:
                    return tmpNode
        return tmpNode

    def expand(self, node):
        node.expandNode()


    def rollout(self, selectedNode):
        """Plays out random untill terminal state"""
        while(not selectedNode.state.isOver()):
            neuralState = selectedNode.state.getNeuralRepresentation()
            feeder = {self.anet.input: [neuralState]}
    
            legalMoves = selectedNode.state.legalMoves
        
            anetOutput = self.anet.current_session.run(self.anet.output, feed_dict=feeder)[0]
            for i in range(len(anetOutput)):
                anetOutput[i] = anetOutput[i] * legalMoves[i]
            anetOutput = [float(i)/sum(anetOutput) for i in anetOutput]
            index = anetOutput.index(max(anetOutput))
            #need to remove one index for each illegal move, because there is no child for that move
            for i in range(index):
                if legalMoves[i] == 0:
                    index -= 1
            selectedNode = selectedNode.getChildNodes()[index]
        return 1 if selectedNode.state.getWinner() == self.startingPlayer else -1

    def rollout2(self, selectedNode):
        stateCopy = HexState(selectedNode.state.player, selectedNode.state.hexSize, selectedNode.state.legalMoves[:], [row[:] for row in selectedNode.state.board])
        """Plays out random untill terminal state"""
        while(not stateCopy.isOver()):
            neuralState = stateCopy.getNeuralRepresentation()
            feeder = {self.anet.input: [neuralState]}
            anetOutput = self.anet.current_session.run(self.anet.output, feed_dict=feeder)[0]

            legalMoves = stateCopy.legalMoves
            for i in range(len(anetOutput)):
                anetOutput[i] = anetOutput[i] * legalMoves[i]
            index = np.where(anetOutput == max(anetOutput))[0][0]
            #need to remove one index for each illegal move, because there is no child for that move
            stateCopy.makeMove(index)
        return 1 if stateCopy.getWinner() == self.startingPlayer else -1

    def randomRollout(self, selectedNode):
        while not selectedNode.state.isOver():
            selectedNode = selectedNode.getRandomChild()
        return 1 if selectedNode.state.getWinner() == self.startingPlayer else -1

    def randomRollout2(self, selectedNode):
        state = HexState(selectedNode.state.player, selectedNode.state.hexSize, selectedNode.state.legalMoves[:], [row[:] for row in selectedNode.state.board])
        while not state.isOver():
            state.playRandom()
        return 1 if state.getWinner() == self.startingPlayer else -1
            

    def backPropagate(self, selectedNode, score):
        """Update all parents with score"""
        while(selectedNode is not None):
            selectedNode.updateNodeValue(score)
            selectedNode = selectedNode.parent


    def addToReplayBuffer(self, node):
        """Adds neural representation of the board as input, """
        """and softmaxed visit counts of children as target to the replayBuffer"""
        inp = node.state.getNeuralRepresentation()
        children = node.children
        Dpre = []
        lMoves = node.state.legalMoves
        for node in children:
            Dpre.append(node.numberOfSimulations)
        Dnorm = [0]*(node.state.hexSize * node.state.hexSize)
        for move in range(len(lMoves)):
            if lMoves[move] == 1:
                Dnorm[move] = Dpre.pop(0)
        Dnorm = [float(i)/sum(Dnorm) for i in Dnorm]
        case = [inp, Dnorm]
        self.replayBuffer.append(case)