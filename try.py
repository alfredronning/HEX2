from copy import deepcopy
from math import floor

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
        whiteSideLeft, whiteSideRight, blackSideLeft, blackSideRight = self.getHexSides()

        unvisitedWhiteNodes = []; unvisitedBlackNodes = []
        visitedWhiteNodes = []; visitedBlackNodes = []
        #Checks if white have won
        for node in whiteSideLeft:
            if node.value == 1: unvisitedWhiteNodes.append(node)
        while unvisitedWhiteNodes:
            checkNode = unvisitedWhiteNodes[0]
            for neighbor in checkNode.neighbours:
                if neighbor.value == 1 and neighbor not in unvisitedWhiteNodes and neighbor not in visitedWhiteNodes:
                    unvisitedWhiteNodes.append(neighbor)
            visitedWhiteNodes.append(unvisitedWhiteNodes.pop(0))
            if checkNode in whiteSideRight:
                self.winner = 1
                return True
        #Checks if black have won
        for node in blackSideLeft:
            if node.value == 2: unvisitedBlackNodes.append(node)
        while unvisitedBlackNodes:
            checkNode = unvisitedBlackNodes[0]
            for neighbor in checkNode.neighbours:
                if neighbor.value == 2 and neighbor not in unvisitedBlackNodes and neighbor not in visitedBlackNodes:
                    unvisitedBlackNodes.append(neighbor)
            visitedBlackNodes.append(unvisitedBlackNodes.pop(0))
            if checkNode in blackSideRight:
                self.winner = 2
                return True

        return False

    """ def isOver(self):
        """#Returns true if one player have won the game
        """
        size = self.hexSize
        for i in range(size):
            if self.pathLength(1, i, 0, 0) == size:
                self.winner = 1
                return True
        for i in range(size):
            if self.pathLength(2, 0, i, 0) == size:
                self.winner = 2
                return True
        return False """

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

    """ def pathLength(self, player, row, col, sameRowIndex):
        """#Returns length from this location on the board to the end of the board
        """
        if row < 0 or row > self.hexSize-1 or col < 0 or col > self.hexSize-1:
            return 0
        if self.board[row][col] != player:
            return 0
        if player == 1:
            pathLengths = [self.pathLength(player, row, col+1, 0), self.pathLength(player, row-1, col+1, 0)]
            if sameRowIndex!=-1:
                pathLengths.append(self.pathLength(player, row+1, col, 1)-1)
            if sameRowIndex!=1:
                pathLengths.append(self.pathLength(player, row-1, col, -1)-1)
            return 1 + max(pathLengths)
        #if player = 2
        pathLengths = [self.pathLength(player, row+1, col, 0), self.pathLength(player, row+1, col-1, 0)]
        if sameRowIndex!=-1:
            pathLengths.append(self.pathLength(player, row, col+1, 1)-1)
        if sameRowIndex!=1:
            pathLengths.append(self.pathLength(player, row, col-1, -1)-1)
        return 1 + max(pathLengths) """

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



