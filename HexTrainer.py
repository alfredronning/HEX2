from HexState import HexState
from MCNode import MCNode
from MCTS import MCST
from ANET import ANET, CaseManager
from copy import deepcopy
from random import choice
import tflowtools as TFT
import numpy as np


class HexTrainer():
    def __init__(self, startState, anet, numberOfGames, numberOfSimulations, batchSize, verbose = True,
            mixed = False, savedGames = None, saveFolder = "netsaver/topp/"):

        self.rootNode = MCNode(startState)
        self.numberOfGames = numberOfGames
        self.numberOfSimulations = numberOfSimulations
        self.batchSize = batchSize
        self.verbose = verbose
        self.mixed = mixed
        self.startingPlayer = self.rootNode.state.player

        self.anet = anet
        self.replayBuffer = anet.case_manager.cases

        self.savedGames = savedGames
        self.savePath = saveFolder+"agent"


    def run(self):
        """Runs the batch"""
        print("Starting up playing "+str(self.numberOfGames)+" games: ")
        winsPlayer1 = 0
        winsPlayer2 = 0

        self.anet.setupSession()
        self.anet.error_history = []
        self.anet.validation_history = []
        self.replayBuffer = []
        
        #saving the pilicy for 0 episodes
        self.anet.save_session_params(self.savePath, self.anet.current_session, 0)
        print("saving game after "+str(0)+"episodes as "+ str(0))
        for i in range(self.numberOfGames):

            currentNode = deepcopy(self.rootNode)
            self.replayBuffer = []

            mcst = MCST(currentNode, self.anet, self.replayBuffer, self.numberOfSimulations)

            print("\nGame "+str(i))
            while not currentNode.state.isOver():

                playerToMove = currentNode.state.player

                nextNode = mcst.findNextMove(currentNode)
                if self.verbose: nextNode.state.printBoard()
                
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
                feed_dict=feeder,
                display_interval=0
                )
            #if self.verbose: 
            print("error: "+str(error[0]))
            self.anet.error_history.append((i, error[0]))
            #*********************************************

            #************* Saving session params **********
            if self.savedGames is not None:
                saveInterval = self.numberOfGames/(self.savedGames-1)
                if (i+1) % saveInterval == 0:

                    savedGameNum = int((i+1)/saveInterval)
                    print("saving game after "+str(i+1)+"episodes as "+ str(savedGameNum))
                    self.anet.save_session_params(self.savePath, self.anet.current_session, savedGameNum)
                    
                
        print("player 1 wins {} out of {} games: {} percent".format(winsPlayer1, self.numberOfGames, 100*winsPlayer1/self.numberOfGames))
        print("player 2 wins {} out of {} games: {} percent".format(winsPlayer2, self.numberOfGames, 100*winsPlayer2/self.numberOfGames))

        TFT.plot_training_history(self.anet.error_history, self.anet.validation_history,xtitle="Game",ytitle="Error",
                                   title="",fig=True)

        self.anet.close_current_session(view=False)

        #loop to keep program from closing at the end so we can view the graph
        x = ""
        while x == "":
            x = str(input("enter any key to quit"))

    

def main():
    size = 3

    startState = HexState(player = 1, hexSize = size)

    anet = ANET(
        layer_dims = [size*size*2+2, size*size*4+4, size*size*2+2, size*size],
        case_manager = CaseManager([]),
        learning_rate=0.001,
        display_interval=None,
        minibatch_size=10,
        validation_interval=None,
        softmax=True,
        error_function="ce",
        hidden_activation_function="relu",
        optimizer="adam",
        w_range=[0.0, 0.1],
        grabvars_indexes=[],
        grabvars_types=[],
        lr_freq = None, bs_freq = None, early_stopping=False, target_accuracy=None
        )

    trainer = HexTrainer(startState = startState,
        anet = anet,
        numberOfGames = 60,
        numberOfSimulations = 500,
        batchSize = 20,
        verbose = False,
        savedGames = 5,
        saveFolder = "netsaver/topp/")

    trainer.run()

if __name__ == '__main__':
    main()