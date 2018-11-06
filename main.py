from HexState import HexState
from MCNode import MCNode
from MCTS import MCST
from ANET import ANET, CaseManager

size = 3

startState = HexState(player = 1, hexSize = size)

anet = ANET(
    layer_dims = [size*size*2+2, size*size*2+2, size*size],
    case_manager = CaseManager([]),
    learning_rate=0.001,
    display_interval=None,
    minibatch_size=10,
    validation_interval=None,
    softmax=False,
    error_function="mse",
    hidden_activation_function="relu",
    optimizer="adam",
    w_range=[0.0, 0.1],
    grabvars_indexes=[],
    grabvars_types=[],
    lr_freq = None, bs_freq = None, early_stopping=False, target_accuracy=None
    )

mcts = MCST(startState = startState,
    anet = anet,
    numberOfGames = 100,
    numberOfSimulations = 500,
    verbose = True)

mcts.run()
