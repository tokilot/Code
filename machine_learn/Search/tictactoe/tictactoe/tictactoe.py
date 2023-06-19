"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    Xcount = 0
    Ocount = 0
    for row in board:
        Xcount += row.count(X)
        Ocount += row.count(O)
    #初始状态判断存在问题
    if Xcount <= Ocount:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    action = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                action.add((i,j))
    return action


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    turn = player(board)
    new_board = [[EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY],
                [EMPTY, EMPTY, EMPTY]]
    for i in range(3):
        for j in range(3):
            new_board[i][j] = board[i][j]
    if new_board[action[0]][action[1]]!=None:
        raise Exception
    else:
        new_board[action[0]][action[1]] = turn
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    values = set()
    board = sum(board,[])
    board = [' ' if i is None else i for i in board]
    for i in range(3):
        values.add("".join(board[i:i+3]))
        values.add("".join(board[i::3]))
    values.add("".join(board[0::4]))
    values.add("".join(board[2::4]))
    if "XXX" in values:
        return X
    elif "OOO" in values:
        return O
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board)!=None:
        return True
    for i in board:
        if EMPTY in i:
            return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board)==X:
        return 1
    elif winner(board)==O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    def max_value(board):
        optimal_move = ()
        if terminal(board):
            return utility(board), optimal_move
        else:
            v = -5
            for action in actions(board):
                minval = min_value(result(board, action))[0]
                if minval > v:
                    v = minval
                    optimal_move = action
            return v, optimal_move

    def min_value(board):
        optimal_move = ()
        if terminal(board):
            return utility(board), optimal_move
        else:
            v = 5
            for action in actions(board):
                maxval = max_value(result(board, action))[0]
                if maxval < v:
                    v = maxval
                    optimal_move = action
            return v, optimal_move

    curr_player = player(board)

    if terminal(board):
        return None

    if curr_player == X:
        return max_value(board)[1]

    else:
        return min_value(board)[1]
