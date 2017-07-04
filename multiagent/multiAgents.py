# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[
            index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"
        "To directly use the distance to express Score, we want to choose the minimum distance"
        bestScore = min(scores)
        bestIndices = [index for index in range(len(scores)) if scores[
            index] == bestScore]
        "Avoid Stop, if it is in our bestActions"
        chosenIndex = bestIndices[0]
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        GhostPositions = successorGameState.getGhostPositions()
        PacmanPosition = newPos
        nowFood = currentGameState.getFood()
        FoodList = nowFood.asList()
        FleeDistance = []
        FoodDistance = []
        for Food in FoodList:
            FoodDistance.append(util.manhattanDistance(PacmanPosition, Food))
        for GhostPosition in GhostPositions:
            FleeDistance.append(util.manhattanDistance(
                PacmanPosition, GhostPosition))

        # to eat food in order
        if len(FoodDistance) >= 1:
            Seek = FoodDistance[0]

        # eat the nearest food
        if min(FoodDistance) <= 1:
            Seek = min(FoodDistance)

        # Flee from the Ghost
        Flee = min(FleeDistance)
        if Flee <= 1:
            Flee = 1
        Score = 10 / Flee + Seek
        return Score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        Depth = self.depth
        return self.MinimaxSearch(Depth, gameState, 0)

    def MinimaxSearch(self, currentdepth, state, agent):
        NumberOfAgents = state.getNumAgents()
        currentstate = state
        # PacmanAction is the returned parameter
        PacmanAction = None
        # check the state
        if currentdepth == 0 or currentstate.isWin() or currentstate.isLose():
            return self.evaluationFunction(currentstate)
        # update depth and agent
        newdepth = currentdepth
        NextAgent = agent + 1
        if NextAgent >= NumberOfAgents:
            NextAgent = 0
            newdepth = currentdepth - 1
        # get the children nodes
        Actions = currentstate.getLegalActions(agent)
        FutureStates = []
        for actionIndex in range(len(Actions)):
            FutureStates.append(currentstate.generateSuccessor(
                agent, Actions[actionIndex]))

        # do the iterable process
        FutureScores = [self.MinimaxSearch(
            newdepth, FutureState, NextAgent) for FutureState in FutureStates]
        # return the final action
        if agent == 0 and currentdepth == self.depth:
            value = -99999
            for actionIndex in range(len(Actions)):
                if FutureScores[actionIndex] > value:
                    value = FutureScores[actionIndex]
                    currentstate = FutureStates[actionIndex]
                    PacmanAction = Actions[actionIndex]
            return PacmanAction
        # max or min return
        elif agent == 0:
            return max(FutureScores)
        else:
            return min(FutureScores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        Depth = self.depth
        Alpha = -99999
        Beta = 99999
        return self.AlphaBetaSearch(Depth, gameState, 0, Alpha, Beta)

    def AlphaBetaSearch(self, currentdepth, state, agent, alpha, beta):
        NumberOfAgents = state.getNumAgents()
        currentstate = state
        PacmanAction = None
        if currentdepth == 0 or currentstate.isWin() or currentstate.isLose():
            return self.evaluationFunction(currentstate)
        # update depth and agent
        newdepth = currentdepth
        NextAgent = agent + 1
        if NextAgent >= NumberOfAgents:
            NextAgent = 0
            newdepth = currentdepth - 1
        Actions = currentstate.getLegalActions(agent)

        # return the result
        if agent == 0 and currentdepth == self.depth:
            value = -99999
            for action in Actions:
                FutureScore = self.AlphaBetaSearch(
                    newdepth, currentstate.generateSuccessor(agent, action), NextAgent, alpha, beta)
                # choose the maximum
                if FutureScore > value:
                    value = FutureScore
                    PacmanAction = action
                # check and cut the branch
                if value > beta:
                    return PacmanAction
                # update the alpha, and this alpha will be considered in next
                # depth
                else:
                    alpha = max(alpha, value)
            return PacmanAction
        elif agent == 0:
            value = -99999
            for action in Actions:
                FutureScore = self.AlphaBetaSearch(
                    newdepth, currentstate.generateSuccessor(agent, action), NextAgent, alpha, beta)
                if FutureScore > value:
                    value = FutureScore
                if value > beta:
                    return value
                else:
                    alpha = max(alpha, value)
            return value
        else:
            value = 99999
            for action in Actions:
                FutureScore = self.AlphaBetaSearch(
                    newdepth, currentstate.generateSuccessor(agent, action), NextAgent, alpha, beta)
                if FutureScore < value:
                    value = FutureScore
                if value < alpha:
                    return value
                else:
                    beta = min(beta, value)
            return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        Depth = self.depth
        return self.ExpectimaxSearch(Depth, gameState, 0)

    def ExpectimaxSearch(self, currentdepth, state, agent):
        NumberOfAgents = state.getNumAgents()
        currentstate = state
        PacmanAction = None
        if currentdepth == 0 or currentstate.isWin() or currentstate.isLose():
            return self.evaluationFunction(currentstate)
        # update depth and agent
        newdepth = currentdepth
        NextAgent = agent + 1
        if NextAgent >= NumberOfAgents:
            NextAgent = 0
            newdepth = currentdepth - 1
        LegalActions = currentstate.getLegalActions(agent)
        Actions = [i for i in LegalActions if i != 'Stop']
        FutureStates = []
        for actionIndex in range(len(Actions)):
            FutureStates.append(currentstate.generateSuccessor(
                agent, Actions[actionIndex]))

        FutureScores = [self.ExpectimaxSearch(
            newdepth, FutureState, NextAgent) for FutureState in FutureStates]
        if agent == 0 and currentdepth == self.depth:
            value = -99999
            for actionIndex in range(len(Actions)):
                if FutureScores[actionIndex] > value:
                    value = FutureScores[actionIndex]
                    currentstate = FutureStates[actionIndex]
                    PacmanAction = Actions[actionIndex]
            return PacmanAction
        elif agent == 0:
            return max(FutureScores)
        # return the mean score of this branch
        else:
            return sum(FutureScores) / len(FutureScores)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    # get the current situation
    GhostPositions = currentGameState.getGhostPositions()
    GhostStates = currentGameState.getGhostStates()
    PacmanPosition = currentGameState.getPacmanPosition()
    CapsuleList = currentGameState.getCapsules()
    Food = currentGameState.getFood()
    FoodList = Food.asList()
    FleeDistance = []
    FoodDistance = []
    CapsuleDistance = []

    # calculate the distance
    for GhostPosition in GhostPositions:
        FleeDistance.append(util.manhattanDistance(
            PacmanPosition, GhostPosition))
    for CapsulePosition in CapsuleList:
        CapsuleDistance.append(util.manhattanDistance(
            PacmanPosition, CapsulePosition))
    # use BFS to get the mazedistance of some food
    FoodDistance = BreadthFirstSearch(currentGameState)

    # To eat the nearest food, if too far, seek the first food
    if len(FoodDistance) >= 1:
        Seek = min(FoodDistance) + 20 * len(FoodDistance)
    else:
        Seek = 0

    # To eat the capsule if it is near here
    Seek = Seek + 10000 * len(CapsuleDistance)

    # To escape from the ghost
    Flee = min(FleeDistance)
    if Flee == 0:
        Flee = 0.01
    if Flee < 1:
        Score = 100 / Flee + Seek
    else:
        Score = Seek

    # To eat the ghost if it's Scared
    Scare = []
    for GhostState in GhostStates:
        if GhostState.scaredTimer >= 2:
            Score = 50 * Flee + Seek

    # we want to choose the minimum Score, so minus Score
    Score = -Score
    return Score


# to return the true distance between Pacman and food near here
def BreadthFirstSearch(gameState):
    PacmanPosition = gameState.getPacmanPosition()
    FoodState = gameState.getFood()
    WallState = gameState.getWalls()
    FoodList = FoodState.asList()
    WallList = WallState.asList()
    FoodDistance = []
    # to reduce the calculation, we only return the true distance near pacman,
    # else return manhhatan distance
    for FoodPosition in FoodList:
        if util.manhattanDistance(PacmanPosition, FoodPosition) >= 3 and len(FoodList) > 5:
            FoodDistance.append(util.manhattanDistance(
                PacmanPosition, FoodPosition))
            continue
        ExploredSet = set()
        OurPaths = util.Queue()
        OurPaths.push([PacmanPosition])
        while not OurPaths.isEmpty():
            ThisPath = OurPaths.pop()
            OurPosition = ThisPath[-1]
            if OurPosition == FoodPosition:
                FoodDistance.append(len(ThisPath) - 1)
                break
            if OurPosition not in ExploredSet:
                ExploredSet.add(OurPosition)
                LegalPositions = getLegalSuccessorPositions(
                    OurPosition, WallList)
                for i in LegalPositions:
                    PathToThisSuccessor = ThisPath + [i]
                    OurPaths.push(PathToThisSuccessor)
    return FoodDistance


def getLegalSuccessorPositions(Position, Walls):
    East = (Position[0] + 1, Position[1])
    West = (Position[0] - 1, Position[1])
    North = (Position[0], Position[1] + 1)
    South = (Position[0], Position[1] - 1)
    SuccessorPositions = [East, West, North, South]
    LegalPositions = [
        SuccessorPosition for SuccessorPosition in SuccessorPositions if SuccessorPosition not in Walls]
    return LegalPositions

# Abbreviation
better = betterEvaluationFunction
