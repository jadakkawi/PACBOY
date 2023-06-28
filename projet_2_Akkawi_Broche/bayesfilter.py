# Complete this class for all parts of the project

from pacman_module.game import Agent
import numpy as np
from pacman_module import util
from scipy.stats import binom


class BeliefStateAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        """
            Variables to use in 'update_belief_state' method.
            Initialization occurs in 'get_action' method.

            XXX: DO NOT MODIFY THE DEFINITION OF THESE VARIABLES
            # Doing so will result in a 0 grade.
        """

        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance

        self.p = 0.5
        self.n = int(self.sensor_variance/(self.p*(1-self.p)))

        # XXX: Your code here
        # NB: Adding code here is not necessarily useful, but you may.
        # XXX: End of your code

    def _get_sensor_model(self, pacman_position, evidence):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The sensor model represented as a 2D numpy array of
        size [width, height].
        The element at position (w, h) is the probability
        P(E_t=evidence | X_t=(w, h))
        """
        sensor_model = np.zeros((self.walls.width, self.walls.height))
        for i in range(self.walls.width):
            for j in range(self.walls.height):
                sensor_model[i][j] = binom.pmf(evidence - util.manhattanDistance((i, j), pacman_position)
                                               + self.n * self.p, self.n, self.p)
        return sensor_model

    def _get_transition_model(self, pacman_position):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The transition model represented as a 4D numpy array of
        size [width, height, width, height].
        The element at position (w1, h1, w2, h2) is the probability
        P(X_t+1=(w1, h1) | X_t=(w2, h2))
        """
        transition_model = np.zeros((self.walls.width, self.walls.height, self.walls.width, self.walls.height))
        k = 1
        if self.ghost_type == "scared":
            k = 3
        if self.ghost_type == "afraid":
            k = 1
        if self.ghost_type == "confused":
            k = 0
        normalizer = dict()
        for w1 in range(self.walls.width):
                for h1 in range(self.walls.height):

                    if not self.walls[w1][h1]:
                        t_plus_1_distance = util.manhattanDistance(pacman_position, (w1, h1))

                        for w2 in range(self.walls.width):
                            for h2 in range(self.walls.height):
                                normalizer.setdefault((w2, h2), 0)
                                if ((w1 == w2+1 and h1 == h2) or (w1 == w2-1 and h1 == h2) or
                                    (w1 == w2 and h1 == h2+1) or (w1 == w2 and h1 == h2-1)) and (not self.walls[w2][h2]):
                                    t_distance = util.manhattanDistance(pacman_position, (w2, h2))
                                    if t_plus_1_distance > t_distance:
                                        transition_model[w1][h1][w2][h2] = np.power(2, k)
                                        normalizer[(w2, h2)] += np.power(2, k)
                                    else:
                                        transition_model[w1][h1][w2][h2] = 1
                                        normalizer[(w2, h2)] += 1
                                else:
                                    transition_model[w1][h1][w2][h2] = 0
        for w1 in range(self.walls.width):
            for h1 in range(self.walls.height):
                for w2 in range(self.walls.width):
                    for h2 in range(self.walls.height):
                        if not self.walls[w2][h2] and normalizer[(w2, h2)]!= 0:
                            transition_model[w1][h1][w2][h2] = transition_model[w1][h1][w2][h2]/normalizer[(w2, h2)]
        return transition_model

    def _get_updated_belief(self, belief, evidences, pacman_position, ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        and the previous belief states before receiving the evidences,
        returns the updated list of belief states about ghosts positions

        Arguments:
        ----------
        - `belief`: A list of Z belief states at state x_{t-1}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze.
               Matrices filled with zeros must be returned for eaten ghosts.
        """

        # XXX: Your code here
        trans_model = self._get_transition_model(pacman_position)

        for e in range(len(belief)):
            push = np.zeros((self.walls.width, self.walls.height))
            if ghosts_eaten[e] == 0:
                sensor_model = self._get_sensor_model(pacman_position, evidences[e])
                for i in range(self.walls.width):
                    for j in range(self.walls.height):
                        if (not self.walls[i][j]) and (not pacman_position == (i, j)):
                            for u in range(self.walls.width):
                                for v in range(self.walls.height):
                                    push[i][j] += trans_model[i][j][u][v] * belief[e][u][v]
                belief[e] = sensor_model * push
                alpha = np.sum(belief[e])
                if alpha != 0:
                    belief[e] = np.divide(belief[e], alpha)
            else:
                belief[e] = np.zeros((self.walls.width, self.walls.height))
        # XXX: End of your code

        return belief

    def update_belief_state(self, evidences, pacman_position, ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        belief = self._get_updated_belief(self.beliefGhostStates, evidences,
                                          pacman_position, ghosts_eaten)
        self.beliefGhostStates = belief
        return belief

    def _get_evidence(self, state):
        """
        Computes noisy distances between pacman and ghosts.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.


        Return:
        -------
        - A list of Z noised distances in real numbers
          where Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for pos in positions:
            true_distance = util.manhattanDistance(pos, pacman_position)
            noise = binom.rvs(self.n, self.p) - self.n*self.p
            noisy_distances.append(true_distance + noise)

        return noisy_distances

    def _record_metrics(self, belief_states, state):
        """
        Use this function to record your metrics
        related to true and belief states.
        Won't be part of specification grading.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.
        - `belief_states`: A list of Z
           N*M numpy matrices of probabilities
           where N and M are respectively width and height
           of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze
        """

        records = open("confidence_quality_metrics walls scared 10"+".txt", "a")
        max_position_array = np.where(belief_states[0].max() == belief_states[0])
        max_position = (max_position_array[0][0], max_position_array[1][0])

        ghost_position = state.getGhostPosition(1)
        M,N = belief_states[0].shape
        mean_position = np.zeros(2)
        for i in range(M):
            for j in range(N):
                mean_position = np.add(mean_position, np.array([i, j])*belief_states[0][i][j])
        mean_position = [round(u) for u in mean_position]
        mean_position = (mean_position[0], mean_position[1])
        quality = util.manhattanDistance(mean_position, ghost_position)
        # print(f"ghost pos: {ghost_position}")
        # print(f"max pos: {max_position}")
        # print(f"mean pos: {mean_position} \n")
        variance = 0
        for i in range(M):
            for j in range(N):
                variance += (util.manhattanDistance(mean_position, (i,j))**2)*belief_states[0][i][j]
        std = (variance)**0.5

        records.write(str(round(std, 4))+"\t"+str(quality)+"\n")
        records.close()

    def get_action(self, state):
        """
        Given a pacman game state, returns a belief state.

        Arguments:
        ----------
        - `state`: the current game state.
                   See FAQ and class `pacman.GameState`.

        Return:
        -------
        - A belief state.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        evidence = self._get_evidence(state)
        newBeliefStates = self.update_belief_state(evidence,
                                                   state.getPacmanPosition(),
                                                   state.data._eaten[1:])
        self._record_metrics(self.beliefGhostStates, state)

        return newBeliefStates, evidence
