from MCNode import MCNode
#from TOPP import TOPP
from copy import deepcopy
import tflowtools as TFT
import numpy as np


class MCST():
    def __init__(self, startState, anet, numberOfGames, numberOfSimulations, verbose = True, mixed = False, k = None):
        self.rootNode = MCNode(startState)
        self.numberOfGames = numberOfGames
        self.numberOfSimulations = numberOfSimulations
        self.verbose = verbose
        self.mixed = mixed
        self.startingPlayer = self.rootNode.state.player

        self.anet = anet
        self.replayBuffer = anet.case_manager.cases

        #self.TOPP = TOPP()
        self.k = k


    def run(self):
        """Runs the batch"""
        print("Starting up playing "+str(self.numberOfGames)+" games: ")
        winsPlayer1 = 0
        winsPlayer2 = 0
        startPlayer1 = 0


        self.anet.setupSession()
        self.anet.error_history = []
        self.anet.validation_history = []

        for i in range(self.numberOfGames):

            currentNode = deepcopy(self.rootNode)

            print("\nGame "+str(i))
            while not currentNode.state.isOver():

                playerToMove = currentNode.state.player

                nextNode = self.findNextMove(currentNode)
                if self.verbose: self.printMove(currentNode, nextNode)
                
                if nextNode.state.isOver():
                    if self.verbose:
                        print("\nPlayer " + str(playerToMove) + " wins \n")
                    if playerToMove == 1:
                        winsPlayer1 += 1
                    else:
                        winsPlayer2 += 1

                currentNode = nextNode
                currentNode.parent = None

            #************** training of anet ************
            np.random.shuffle(self.replayBuffer)
            inputs = [case[0] for case in self.replayBuffer]; targets = [case[1] for case in self.replayBuffer] 
            feeder = {self.anet.input: inputs, self.anet.target: targets}

            gvars = [self.anet.error]   

            _, error, _ = self.anet.run_one_step(
                [self.anet.trainer],
                grabbed_vars = gvars,
                session=self.anet.current_session,
                feed_dict=feeder
                )
            if self.verbose: print("error: "+str(error[0]))
            self.anet.error_history.append((i, error[0]))
            #*********************************************

            #saving the sessions

            #if 

        print("player 1 wins {} out of {} games: {} percent".format(winsPlayer1, self.numberOfGames, 100*winsPlayer1/self.numberOfGames))
        print("player 2 wins {} out of {} games: {} percent".format(winsPlayer2, self.numberOfGames, 100*winsPlayer2/self.numberOfGames))

        TFT.plot_training_history(self.anet.error_history, self.anet.validation_history,xtitle="Game",ytitle="Error",
                                   title="",fig=True)

        self.anet.close_current_session(view=False)

        #loop to keep program from closing at the end so we can view the graph
        x = ""
        while x == "":
            x = str(input("enter any key to quit"))
   
    def findNextMove(self, currentNode):
        """Finds the next move for the actual game"""
        for i in range(self.numberOfSimulations):

            #selection with UCB untill unvisited node
            selectedNode = self.selectNode(currentNode)

            #expand node if needed
            selectedNode.expandNode()

            if len(selectedNode.children):
                selectedNode = selectedNode.getRandomChild()

            #simulate a single rollout
            score = self.rollout(selectedNode)

            #backpropogate the score from the rollout from the selected node up to root
            self.backPropagate(selectedNode, score)


        self.addToReplayBuffer(currentNode)
        return currentNode.getBestVisitChild()

    def selectNode(self, currentNode):
        """Returns the first unvisited node with UCB policy"""
        tmpNode = currentNode
        if tmpNode is not None:
            while(len(tmpNode.children)):
                minTurn = tmpNode.state.player != self.startingPlayer
                tmpNode = tmpNode.getBestUcbChild(minTurn)
                if tmpNode.numberOfSimulations == 1:
                    return tmpNode
        return tmpNode

    def rollout(self, selectedNode):
        """Plays out random untill terminal state"""
        while(not selectedNode.state.isOver()):
            neuralState = selectedNode.state.getNeuralRepresentation()
            feeder = {self.anet.input: [neuralState]}
    
            legalMoves = selectedNode.state.legalMoves
        
            anetOutput = self.anet.current_session.run(self.anet.output, feed_dict=feeder)
            anetOutput = anetOutput[0]
            for i in range(len(anetOutput)):
                anetOutput[i] = anetOutput[i] * legalMoves[i]
            anetOutput = [float(i)/sum(anetOutput) for i in anetOutput]
            index = anetOutput.index(max(anetOutput))
            indexLen = index
            for i in range(indexLen):
                if legalMoves[i] == 0:
                    index -= 1
            selectedNode = selectedNode.getChildNodes()[index]
        return 1 if selectedNode.state.getWinner() == self.startingPlayer else 0

    def backPropagate(self, selectedNode, score):
        """Update all parents with score"""
        while(selectedNode is not None):
            selectedNode.updateNodeValue(score)
            selectedNode = selectedNode.parent


    def addToReplayBuffer(self, node):
        """Adds neural representation of the board as input, """
        """and softmaxed visit counts of children as target to the replayByffer"""
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
        if(sum(Dnorm) != 0):
            Dnorm = [float(i)/sum(Dnorm) for i in Dnorm]
        case = [inp, Dnorm]
        self.replayBuffer.append(case)


    def printMove(self, fromNode, toNode):
        """Prints out the move from node to node"""
        toNode.state.printBoard()