# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions
from pacman_module.util import manhattanDistance


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


def split_grid(_food_Grid, _my_splitter, all_distances):
    """
        Given a food_grid, an initial food dot splitter and an empty list of distances, does a series
        of scans to fill the list with dot-to-dot distances as described in report.

        Arguments:
        ----------
        - `_food_grid`: the food grid at the cut-off state.
        - '_my_splitter': initial food point splitter
        - 'all_distances': the list to fill

        Return:
        -------
        - Void.
        - Fills a given list with manhattan distances between food dots
    """
    my_food_counter = {"N_W": [], "N_E": [], "S_W": [], "S_E": []}

    for i in range(_food_Grid.width):
        for j in range(_food_Grid.height):
            if _food_Grid[i][j]:
                if i <= _my_splitter[0] and j < _my_splitter[1]:
                    my_food_counter["N_W"].append((i, j))
                elif i > _my_splitter[0] and j <= _my_splitter[1]:
                    my_food_counter["N_E"].append((i, j))
                elif i >= _my_splitter[0] and j > _my_splitter[1]:
                    my_food_counter["S_E"].append((i, j))
                elif i < _my_splitter[0] and j >= _my_splitter[1]:
                    my_food_counter["S_W"].append((i, j))
    my_zone = []
    for direction in my_food_counter:
        my_zone.append(my_food_counter[direction])
    if sum([len(zone) for zone in my_zone]) == 0:
        return
    my_zone.sort(key=len)
    manhat = [manhattanDistance(_my_splitter, foody) for foody in my_zone[-1]]
    all_distances.append(min(manhat))
    _my_new_splitter = my_zone[-1][manhat.index(min(manhat))]
    _food_Grid[_my_splitter[0]][_my_splitter[1]] = False
    split_grid(_food_Grid, _my_new_splitter, all_distances)


def eval_function(state):

    """
        Given a state (AT CUTOFF or WIN/LOSE)
        Hminimax2 is Same as Hminimax1.

        Returns an evaluation(heuristic) of a given state.

        Arguments:
        ----------
        - 'state': the current game state.


        Return:
        -------
        - The value of the evaluation.
    """

    pacman_position = state.getPacmanPosition()
    ghost_position = state.getGhostPosition(1)
    food_Grid = state.getFood()

    # distance between PacMan and Ghost
    dist_Pacman_Ghost = manhattanDistance(pacman_position, ghost_position)

    # Sum of all distances from pacman to food dots
    distances = []
    food_location = []
    for i in range(food_Grid.width):
        for j in range(food_Grid.height):
            if food_Grid[i][j]:
                distances.append(manhattanDistance((i, j), pacman_position))
                food_location.append((i, j))
    # distance between Pacman and closest Food dot and food path created by split grid
    if distances:
        dist_Pacman_food = min(distances)
        my_splitter = food_location[distances.index(dist_Pacman_food)]
        all_distances = []
        _foofood_grid = food_Grid.copy()
        split_grid(_foofood_grid, my_splitter, all_distances)
        return state.getScore() - dist_Pacman_food + dist_Pacman_Ghost*(state.isWin() is False)/2 - sum(all_distances)
    else:
        dist_Pacman_food = 0
        return state.getScore() - dist_Pacman_food + dist_Pacman_Ghost*(state.isWin() is False)/2


class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
                - depth of minimax added pacman-agent class level
                - a dictionry of keys with their corresponding actions as values
        """
        self.max_depth = 5
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

        utility = self.initial_maximize_value(state, my_visited_states, my_action_dict)
        self.actions_taken[key(state)] = utility
        return my_action_dict[utility]

    def initial_maximize_value(self, state, visited, action_dict):
        """
            Implementation of the alpha-beta search pseudo code of lecture. (without pruning)
            NOTE:
                Here we consider that Pacman (MAX player) is starting the game and his actions are to be recorded.

            Arguments:
            ----------
            state: the game state under study
            visited: dictionary that stores eval value for each key(state)
            action_dict: dictionary that stores the Action to take for each corresponding eval value

            Return:
            -------
            maximum eval value

            Void:
            -----
            Fills action dictionary
            Fills visited states dictionary
        """
        current = key(state)
        uti_val = float('-inf')
        current_depth = 0
        uti_action = None

        for next_state, action in state.generatePacmanSuccessors():
            my_max = self.minimize_value(next_state, visited, current_depth + 1)
            if uti_val < my_max:
                uti_val = my_max
                uti_action = action

        action_dict[uti_val] = uti_action
        visited[current] = uti_val
        return uti_val

    def cutoff_test(self, state, depth):
        return depth == self.max_depth or state.isWin() or state.isLose()

    def maximize_value(self, state, visited, current_depth):
        """
            Implementation of the alpha-beta search pseudo code of lecture. (without pruning)
            maximize eval value while expecting MIN player to minimize it
            avoids cycles by checking actions_taken dictionary
        """
        current = key(state)

        if self.cutoff_test(state, current_depth):
            return eval_function(state)
        elif current in visited:
            return visited[current]
        elif current in self.actions_taken:
            return float('-inf')
        else:
            uti_val = float('-inf')
            visited[current] = uti_val
            for next_state, action in state.generatePacmanSuccessors():
                uti_val = max(uti_val, self.minimize_value(next_state, visited, current_depth + 1))

            visited[current] = uti_val
            return uti_val

    def minimize_value(self, state, visited, current_depth):
        """
            Implementation of the alpha-beta search pseudo code of lecture. (without pruning)
            minimize eval value while expecting MAX player to maximize it
            avoids cycles by checking actions_taken dictionary
        """

        current = key(state)

        if self.cutoff_test(state, current_depth):
            return eval_function(state)
        elif current in visited:
            return visited[current]
        elif current in self.actions_taken:
            return float('inf')
        else:
            uti_val = float('inf')
            visited[current] = uti_val
            for next_state, action in state.generateGhostSuccessors(1):
                uti_val = min(uti_val, self.maximize_value(next_state, visited, current_depth + 1))
            visited[current] = uti_val
            return uti_val




