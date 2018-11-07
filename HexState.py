from copy import deepcopy

class HexState:
    def __init__(self, player = 1, hexSize = 2, legalMoves = None, board = None):
        self.player = player
        self.hexSize = hexSize
        self.winner = None
        self.player = player
        self.board = board

        if board is None:
            self.buildBoard()

        if legalMoves is None:
            self.legalMoves = [1] * (hexSize * hexSize)
        else:
            self.legalMoves = legalMoves
        

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
                if self.board[rowIndex][colIndex].value == 0:
                    childState = HexState(player=3-self.player,
                        hexSize=self.hexSize,
                        legalMoves=self.legalMoves.copy(),
                        board = deepcopy(self.board)
                        )
                    
                    childState.board[rowIndex][colIndex].value = self.player

                    #remove one legal move            
                    childState.legalMoves[rowIndex*self.hexSize + colIndex] = 0
                
                    childStates.append(childState)
        return childStates


    # returns the vector-representation of the boardstate
    def getNeuralRepresentation(self):
        neuralRepr = []
        board = self.board
        for row in board:
            for col in row:
                if col.value == 1:
                    neuralRepr.append(float(1)); neuralRepr.append(float(0))
                elif col.value == 2:
                    neuralRepr.append(float(0)); neuralRepr.append(float(1))
                else:
                    neuralRepr.append(float(0)); neuralRepr.append(float(0))
        if self.player == 1:
            neuralRepr.append(float(1)); neuralRepr.append(float(0))
        else:
            neuralRepr.append(float(0)); neuralRepr.append(float(1))
        return neuralRepr

                

    # builds the board with inital cells
    def buildBoard(self):
        self.board = [[HexCell() for i in range(self.hexSize)] for i in range(self.hexSize)]
        # goes through every cell in the board and connects them to their neighbours
        for rowIndex, row in enumerate(self.board):
            for colIndex, cell in enumerate(row):
                self.addCellNeighbours(cell, rowIndex, colIndex)


    #adds all possible neighboors of cell
    def addCellNeighbours(self, cell, rowIndex, colIndex):
        maxIndex = self.hexSize - 1
        if rowIndex-1 >= 0:
            cell.neighbours.append(self.board[rowIndex-1][colIndex])
      
        if rowIndex -1 >= 0 and colIndex + 1 <= maxIndex:
            cell.neighbours.append(self.board[rowIndex-1][colIndex+1])

        if colIndex - 1 >= 0:
            cell.neighbours.append(self.board[rowIndex][colIndex - 1])

        if colIndex + 1 <= maxIndex:
            cell.neighbours.append(self.board[rowIndex][colIndex + 1])

        if rowIndex + 1 <= maxIndex and colIndex - 1 >= 0:
            cell.neighbours.append(self.board[rowIndex +1][colIndex - 1])

        if rowIndex + 1 <= maxIndex:
            cell.neighbours.append(self.board[rowIndex + 1][colIndex])


    def getHexSides(self):
        #white will get (0,0), (1, 0), (2,0)  and (0, n), (1, n), (2, n) etc
        whiteNodesLeft = []
        whiteNodesRight = []
        #black will get (0, 0), (0, 1), (0, 2)  and (n, 0), (n, 1), (n, 2) etc
        blackNodesLeft = []
        blackNodesRight = []

        for i in range(self.hexSize):
            whiteNodesLeft.append(self.board[i][0]); whiteNodesRight.append(self.board[i][self.hexSize - 1])
            blackNodesRight.append(self.board[0][i]); blackNodesLeft.append(self.board[self.hexSize - 1][i])

        return whiteNodesLeft, whiteNodesRight, blackNodesLeft, blackNodesRight


    #Returns true if coordinates is not out of bounds of the board
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
                if cell.value == 1:
                    printValue = "W"
                if cell.value == 2:
                    printValue = "B"
                if cell.value == 0:
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


#class representing a single cell in the board
class HexCell:
    def __init__(self, value=0):
        self.value = value
        self.neighbours = []

