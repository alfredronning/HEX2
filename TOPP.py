import tensorflow as tf
from ANET import ANET, CaseManager
import tflowtools as TFT
import numpy as np

class TOPP:
    def __init__(self, layerDims, numberOfAgents, games, loadPath = "netsaver/topp/agent", verbose = True):
        self.agents = self.createAgents(layerDims, loadPath, numberOfAgents)

    def createAgents(self, dims, path, numberOfAgents):
        agents = []
        for i in range(numberOfAgents):
            agents.append(HexAgent(dims, path, i))
        return agents


class HexAgent:
    def __init__(self, layerDims, loadPath, globalStep):
        self.anet = None
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
        move = self.anet.current_session.run(self.anet.output, feed_dict=feeder)
        return move[0]

def main():
    size = 3

    agent4 = HexAgent([size*size*2+2, size*size*2+2, size*size], "netsaver/topp/agent", 4)
    agent1 = HexAgent([size*size*2+2, size*size*2+2, size*size], "netsaver/topp/agent", 1)

    #move4 = agent4.giveMoveFromState([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    move1 = agent1.giveMoveFromState([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    print("-----------")
    #print(move4)
    #print(np.where(move4 == max(move4))[0][0]+1)
    print("-------------")
    print(move1)
    print(sum(move1))
    print(np.where(move1 == max(move1))[0][0]+1)


if __name__ == '__main__':
    main()