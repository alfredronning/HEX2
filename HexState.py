from copy import deepcopy
from math import floor
from random import choice

class HexState:
    def __init__(self, player = 1, hexSize = 2, legalMoves = None, board = None):
        self.player = player
        self.hexSize = hexSize
        self.winner = None
        self.player = player

        if board is not None:
            self.board = board
        else:
            self.board = [[0 for i in range(self.hexSize)] for j in range(self.hexSize)]
        
        if legalMoves is not None:
            self.legalMoves = legalMoves
        else:
            self.legalMoves = [1]*hexSize*hexSize
        

    def isOver(self):
        """Returns true if one player have won the game"""

        whiteSideRight = []; blackSideRight = []
        unvisitedWhiteCells = []; unvisitedBlackCells = []
        visitedWhiteCells = []; visitedBlackCells = []
        size = self.hexSize
        board = self.board
        #Checks if white have won
        for i in range(size):
            if self.board[i][0] == 1:
                unvisitedWhiteCells.append((i, 0))
            whiteSideRight.append((i, size - 1))
        while unvisitedWhiteCells:
            checkCell = unvisitedWhiteCells.pop(0)
            for neighbor in self.getNeighbours(checkCell):
                if board[neighbor[0]][neighbor[1]] == 1 and neighbor not in unvisitedWhiteCells and neighbor not in visitedWhiteCells:
                    unvisitedWhiteCells.append(neighbor)
            visitedWhiteCells.append(checkCell)
            if checkCell in whiteSideRight:
                self.winner = 1
                return True
        #Checks if black have won
        for i in range(size):
            if self.board[size-1][i] == 2:
                unvisitedBlackCells.append((size - 1, i))
            blackSideRight.append((0, i))
        while unvisitedBlackCells:
            checkCell = unvisitedBlackCells.pop(0)
            for neighbor in self.getNeighbours(checkCell):
                if board[neighbor[0]][neighbor[1]] == 2 and neighbor not in unvisitedBlackCells and neighbor not in visitedBlackCells:
                    unvisitedBlackCells.append(neighbor)
            visitedBlackCells.append(checkCell)
            if checkCell in blackSideRight:
                self.winner = 2
                return True

        return False

    def getNeighbours(self, tup):
        neighbours = []
        maxIndex = self.hexSize - 1
        if tup[0] - 1 >= 0:
            neighbours.append((tup[0]-1, tup[1]))
      
        if tup[0] - 1 >= 0 and tup[1]  +1 <= maxIndex:
            neighbours.append((tup[0]-1,tup[1]+1))

        if tup[1]-1 >= 0:
            neighbours.append((tup[0], tup[1]-1))

        if tup[1] + 1 <= maxIndex:
            neighbours.append((tup[0], tup[1]+1))

        if tup[0] + 1 <= maxIndex and tup[1] - 1 >= 0:
            neighbours.append((tup[0]+1, tup[1]-1))

        if tup[0] + 1 <= maxIndex:
            neighbours.append((tup[0]+1, tup[1]))
        return neighbours


    def getWinner(self):
        """Returns the winner of this state. None for unfinnished states"""
        if self.winner is None:
            self.isOver()
        return self.winner


    def getChildStates(self):
        """Returns a list of all possible child states derived from this state"""
        childStates = []
        for rowIndex in range(self.hexSize):
            for colIndex in range(self.hexSize):
                if self.board[rowIndex][colIndex] == 0:
                    childState = HexState(player=3-self.player,
                        hexSize = self.hexSize,
                        legalMoves=self.legalMoves[:],
                        board = [row[:] for row in self.board]
                        )
                    childState.board[rowIndex][colIndex] = self.player
                    childState.legalMoves[rowIndex*self.hexSize+colIndex] = 0
                    childStates.append(childState)
        return childStates

    def makeMove(self, index):
        """Makes move on this board state"""
        self.board[int(index/self.hexSize)][index%self.hexSize] = self.player
        self.legalMoves[index] = 0
        self.player = 3 - self.player

    def playRandom(self):
        moves = [i for i, x in enumerate(self.legalMoves) if x == 1]
        move = choice(moves)
        self.board[int(move/self.hexSize)][move%self.hexSize] = self.player
        self.legalMoves[move] = 0
        self.player = 3 - self.player

    # returns the vector-representation of the boardstate
    def getNeuralRepresentation(self):
        """Resturns a list in neural input format of this board state"""
        neuralRepr = []
        for row in self.board:
            for col in row:
                if col == 1:
                    neuralRepr.append(float(1)); neuralRepr.append(float(0))
                elif col == 2:
                    neuralRepr.append(float(0)); neuralRepr.append(float(1))
                else:
                    neuralRepr.append(float(0)); neuralRepr.append(float(0))
        if self.player == 1:
            neuralRepr.append(float(1)); neuralRepr.append(float(0))
        else:
            neuralRepr.append(float(0)); neuralRepr.append(float(1))
        return neuralRepr


    def coordinateIsInBoard(self, iRow, iCol):
        maxIndex = self.hexSize - 1
        return iRow >= 0 and iRow <= maxIndex and iCol >= 0 and iCol <= maxIndex

    #prints the board
    def printBoard(self):
        maxIndex = self.hexSize - 1
        lines = []
        metaColIndex = 0
        metaRowIndex = -1

        # organize board into hex shape
        for i in range(0, self.hexSize*2 - 1):
            if i > maxIndex: metaColIndex += 1
            if i <= maxIndex: metaRowIndex += 1
            rowIndex = metaRowIndex
            colIndex = metaColIndex
            line = []
            while self.coordinateIsInBoard(rowIndex, colIndex):
                line.append(self.board[rowIndex][colIndex])
                rowIndex -= 1
                colIndex += 1
            lines.append(line)

        stringSpace = "   "
        spaceOffset = "                  "
        spaceController = self.hexSize
        spaceDecrease = True
        print(stringSpace + spaceOffset + "------"*maxIndex)
        print(stringSpace + spaceOffset + "W" + "------"*maxIndex + "B")

        #actually print board in hex format
        for i in range(0, self.hexSize*2 - 1):
            space = spaceOffset + stringSpace * spaceController
            for cell in lines[i]:
                printValue = "?"
                if cell == 1:
                    printValue = "W"
                if cell == 2:
                    printValue = "B"
                if cell == 0:
                    printValue = "O"
           
                space += printValue + "     "

            print(space)

            #control when to switch from removing space to adding space
            if i >= maxIndex: spaceDecrease = False
            
            #add or remove space before printing values
            if spaceDecrease:
                spaceController -= 1
            else:
                spaceController += 1

        print(stringSpace + spaceOffset + "B" + "------"*maxIndex + "W")
        print(stringSpace + spaceOffset + "------"*maxIndex)



