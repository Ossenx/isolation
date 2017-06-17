"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import isolation
class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    ## Similar to center score, this bases score off of how far a piece is from
    ## an its too closest edges in comparison to its opponent.

    w, h = game.width, game.height
    my, mx = game.get_player_location(player)
    oy, ox = game.get_player_location(game.get_opponent(player))
    # I square the distance here to eliminate any negative values
    score = ((oy-h/2) ** 2 + (ox-w/2) **2) - ((my-h/2) ** 2 + (mx-w/2) **2)

    return float(score)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    opponent = game.get_opponent(player)
    my, mx = game.get_player_location(player)
    oy, ox = game.get_player_location(opponent)
    open_spaces = game.get_blank_spaces()

    open_spaces_near_me = 0
    open_spaces_near_opp = 0
    # The idea here is to find how many open spaces are within 2 spaces of
    # each player because any open spaces within two squares are possible
    # subsequent legal moves as well.
    for opx, opy in open_spaces:
        if abs(opx - mx) <= 2 or abs(opy - my) <= 2:
            open_spaces_near_me += 1
        if abs(opx - ox) <= 2 or abs(opy - oy) <= 2:
            open_spaces_near_opp += 1

    return float(open_spaces_near_me - open_spaces_near_opp)



def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    my_legal_moves = game.get_legal_moves(player)
    opponent = game.get_opponent(player)
    opp_legal_moves = game.get_legal_moves(opponent)
    my_total = 0
    opp_total = 0
    for mx, my in my_legal_moves:
        # Make negative, since being closer to the edges should be penalized
        value = (-1 * (w - mx)**2 + (h - my)**2)
        my_total += value
    for ox, oy in opp_legal_moves:
        # Make negative, since being closer to the edges should be penalized
        value = (-1 * (w - ox)**2 + (h - oy)**2)
        opp_total += value

        # This function, in its current state has not been weighted to work with
        # other heuristics, since the score returned is often negative
    return float(my_total - opp_total)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def find_score_of_minimizing_move(game, depth):
            # Make sure to include timeout in any helper functions
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            moves = game.get_legal_moves()
            # Check for all different cases that the game wouldn't be able to
            # forecast any future moves (end game)
            if depth <= 0 or len(moves) == 0 or game.utility(self) != 0:
                return self.score(game, self)
            best_score = float('inf')
            for move in moves:
                clone = game.forecast_move(move)
                score = find_score_of_maximizing_move(clone, depth - 1)
                if score < best_score:
                    best_score = score
            return best_score

        def find_score_of_maximizing_move(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            moves = game.get_legal_moves()
            # Check for all different cases that the game wouldn't be able to
            # forecast any future moves (end game)
            if depth <= 0 or len(moves) == 0 or game.utility(self) != 0:
                return self.score(game, self)
            best_score = float('-inf')
            for move in moves:
                clone = game.forecast_move(move)
                score = find_score_of_minimizing_move(clone, depth - 1)
                if score > best_score:
                    best_score = score

            return best_score


        moves = game.get_legal_moves()
        best_score = float('-inf')
        # This is simply to initialize best_move. (-1, -1) should only be used
        # as an illegal move, never a premature forfeit.
        best_move = (-1, -1)
        # This initializes best_move to be essentially random. (Unless there is
        # an unclear rhyme or reason to the order in which legal moves are
        # returned)
        if len(moves) > 0:
            best_move = moves[0]

        for move in moves:
            clone = game.forecast_move(move)
            score = find_score_of_minimizing_move(clone, depth - 1)
            if score > best_score:
                best_move = move
                best_score = score
        return best_move




class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)
        legal_moves = game.get_legal_moves()
        if len(legal_moves) > 0:
            best_move = legal_moves[0]
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout


        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.

            depth = 0
            # Iterative deepening will continuously run, one level deeper
            # each time until there is a winner or timeout
            while depth < float("inf"):
                depth += 1
                best_move = self.alphabeta(game, depth, best_move)



        except SearchTimeout:
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, best_move, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        def min_play(game, alpha, beta, depth):
            ## End if time runs out
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            ## End if depth limited search reaches full depth or if game ends
            moves = game.get_legal_moves()
            if depth <= 0 or len(moves) <= 0 or game.utility(self) != 0:
                return self.score(game, self)
            score = float('inf')
            for move in moves:
                clone = game.forecast_move(move)
                # max_play is called here since we're alternate players
                score = min(score, max_play(clone, alpha, beta, depth - 1))
                ## Check each node to see if we have a new min lower than our current alpha
                if score <= alpha:
                    return score
                ## Set beta to be whichever is smaller
                beta = min(beta, score)
            return score

        def max_play(game, alpha, beta, depth):
            ## End if time runs out
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            ## End if depth limited search reaches full depth or if game ends
            moves = game.get_legal_moves()
            if depth <= 0 or len(moves) <= 0 or game.utility(self) != 0:
                return self.score(game, self)
            score = float('-inf')
            for move in moves:
                clone = game.forecast_move(move)
                # min_play is called here since we're alternate players
                score = max(score, min_play(clone, alpha, beta, depth - 1))
                if score >= beta:
                    return score
                # Set alpha to whichever is larger
                alpha = max(alpha, score)
            return score

        # End if time runs out
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Initialize best_move to (-1, -1) if this is the first iteration of ID
        if not best_move:
            best_move = (-1, -1)
        moves = game.get_legal_moves()
        best_score = float('-inf')
        for move in moves:
            clone = game.forecast_move(move)
            # Start with min_play, since opponent turn comes after our
            # hypothetical move.
            score = min_play(clone, best_score, beta, depth - 1)
            # Keep track of the score that is returned from each tree
            if score > best_score:
                best_move = move
                best_score = score
        return best_move
