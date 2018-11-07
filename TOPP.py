import tensorflow as tf
from ANET import ANET, CaseManager
from HexState import HexState
import tflowtools as TFT
import numpy as np

class TOPP:
    def __init__(self, layerDims, hexSize, numberOfAgents, games, loadPath = "netsaver/topp/agent", verbose = False):
        self.verbose = verbose
        self.hexSize = hexSize
        self.gamesNum = games
        self.agents = self.createAgents(layerDims, loadPath, numberOfAgents)


    def createAgents(self, dims, path, numberOfAgents):
        agents = []
        for i in range(numberOfAgents):
            agents.append(HexAgent(dims, self.hexSize, path, i))
        return agents

    def playTournament(self):
        for i in range(len(self.agents)-1):
            for j in range(len(self.agents), i+1, -1):
                startingplayer = 1
                for _ in range(self.gamesNum):
                    self.playoutGame(self.agents[i], self.agents[j-1], startingplayer)
                    startingplayer = 3 - startingplayer

    def printResults(self):
        for agent in self.agents:
            print(agent.name+" won "+str(agent.wins)+" games")
    
    def playoutGame(self, agent1, agent2, startingplayer):
        game = HexState(1, self.hexSize)
        agents = [agent1, agent2]
        currentplayer = startingplayer
        if self.verbose: print(agent1.name+" vs "+agent2.name+", "+agents[startingplayer-1].name+" starts")
        while not game.isOver():
            move = agents[currentplayer-1].giveMoveFromState(game.getNeuralRepresentation())
            game = game.getChildStates()[move]
            currentplayer = 3 - currentplayer
            if self.verbose: game.printBoard()

        winner = game.getWinner()
        if startingplayer == 2:
            winner = 3-winner
        agents[winner-1].wins += 1
        if self.verbose: print(agents[winner-1].name+" wins")

    



class HexAgent:
    def __init__(self, layerDims, hexSize, loadPath, globalStep):
        self.anet = None
        self.hexSize = hexSize
        self.name = "agent-"+str(globalStep)
        self.wins = 0

        self.loadParams(layerDims, loadPath, globalStep)


    def loadParams(self, layerDims, loadPath, globalStep):
        self.anet = ANET(
        layer_dims = layerDims,
        softmax=True,
        case_manager = CaseManager([]))

        session = TFT.gen_initialized_session(dir="probeview")
        self.anet.current_session = session
        state_vars = []
        for m in self.anet.layer_modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.anet.state_saver = tf.train.Saver(state_vars)
        self.anet.state_saver.restore(self.anet.current_session, loadPath+"-"+str(globalStep))

    def giveMoveFromState(self, neuralRepresentation):
        feeder = {self.anet.input: [neuralRepresentation]}
        moves = self.anet.current_session.run(self.anet.output, feed_dict=feeder)
        legalMoves = [1] * (self.hexSize * self.hexSize)
        for i in range(0, len(neuralRepresentation)-2, 2):
            if neuralRepresentation[i] == 1 or neuralRepresentation[i+1] == 1:
                legalMoves[int(i/2)] = 0
        moves = moves * legalMoves
        moves = moves[0]
        bestMove = np.where(moves == max(moves))[0][0]
        #decrementing to make moveIndex match childIndex
        for i in range(bestMove):
            if legalMoves[i] == 0:
                bestMove -= 1
        return bestMove

def main():
    size = 3

    topp = TOPP(layerDims=[size*size*2+2, size*size*4+4, size*size*2+2, size*size],
        hexSize = size,
        numberOfAgents = 5,
        games = 2,
        loadPath = "netsaver/topp/agent",
        verbose = True)

    topp.playTournament()

    topp.printResults()

    #agent = HexAgent([size*size*2+2, size*size*2+2, size*size], size, "netsaver/topp/agent", 3)


if __name__ == '__main__':
    main()