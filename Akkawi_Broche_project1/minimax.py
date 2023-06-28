# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions


def key(state):
    """
    Returns a key that uniquely identifies a Pacman game state.

    NOTE:
        state.getGhostPosition(1) because we are assuming only ONE ghost is present in this game

    Arguments:
    ----------
    - `state`: the current game state. See FAQ and class
               `pacman.GameState`.

    Return:
    -------
    - A hashable key object that uniquely identifies a Pacman game state.
    """
    return state.getPacmanPosition(), state.getFood(), state.getGhostPosition(1)


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
                - a dictionry of keys with their corresponding actions as values
        """
        self.actions_taken = dict()

    def get_action(self, state):
        """
        Given a pacman game state, returns a legal move.

        NOTE:
            Get_action is calculated at each state given that Pacman doesn't know how the ghost will behave,
            So Minimax will find the optimal action of PacMan but the ghost might react to this action differently
            to what is expected.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """
        my_visited_states = dict()
        my_action_dict = dict()

        utility = self.initial_maximize_utility(state, my_visited_states, my_action_dict)
        self.actions_taken[key(state)] = utility
        return my_action_dict[utility]

    def initial_maximize_utility(self, state, visited, action_dict):
        """
            Implementation of the alpha-beta search pseudo code of lecture. (without pruning)
            NOTE:
                Here we consider that Pacman (MAX player) is starting the game and his actions are to be recorded.

            Arguments:
            ----------
            state: the game state under study
            visited: dictionary that stores utility value for each key(state)
            action_dict: dictionary that stores the Action to take for each corresponding utility value

            Return:
            -------
            maximum utility value

            Void:
            -----
            Fills action dictionary
            Fills visited states dictionary
        """
        current = key(state)
        uti_val = float('-inf')
        uti_action = 0

        for next_state, action in state.generatePacmanSuccessors():
            my_max = self.minimize_utility(next_state, visited)
            if uti_val < my_max:
                uti_val = my_max
                uti_action = action

        action_dict[uti_val] = uti_action
        visited[current] = uti_val
        return uti_val

    def maximize_utility(self, state, visited):
        """
            Implementation of the alpha-beta search pseudo code of lecture. (without pruning)
            maximize utility value while expecting MIN player to minimize it
            avoids cycles by checking actions_taken dictionary
        """
        current = key(state)

        if state.isWin() or state.isLose():
            return state.getScore()
        elif current in visited:
            return visited[current]
        elif current in self.actions_taken:
            return float('-inf')
        else:
            uti_val = float('-inf')
            visited[current] = uti_val
            for next_state, action in state.generatePacmanSuccessors():
                uti_val = max(uti_val, self.minimize_utility(next_state, visited))

            visited[current] = uti_val
            return uti_val

    def minimize_utility(self, state, visited):
        """
            Implementation of the alpha-beta search pseudo code of lecture. (without pruning)
            minimize utility while expecting MAX player to maximize it
            avoids cycles by checking actions_taken dictionary
        """
        current = key(state)

        if state.isWin() or state.isLose():
            return state.getScore()
        elif current in visited:
            return visited[current]
        elif current in self.actions_taken:
            return float('inf')
        else:
            uti_val = float('inf')
            visited[current] = uti_val
            for next_state, action in state.generateGhostSuccessors(1):
                uti_val = min(uti_val, self.maximize_utility(next_state, visited))
            visited[current] = uti_val
            return uti_val




